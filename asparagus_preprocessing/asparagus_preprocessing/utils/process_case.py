import logging
import math
import nibabel as nib
import numpy as np
import os
from asparagus_preprocessing.utils.bbox import get_bbox_for_foreground
from asparagus_preprocessing.utils.crop import crop_to_box
from asparagus_preprocessing.utils.nifti import (
    apply_nifti_preprocessing_and_return_numpy,
)
from asparagus_preprocessing.utils.pad import pad_case_to_size
from asparagus_preprocessing.utils.resample import resample_and_normalize_case
from asparagus_preprocessing.utils.saving import save_data_and_metadata
from dataclasses import asdict
from typing import List, Optional, Union


def process_mri_case(path, image_save_path, preprocessing_config, saving_config):
    # TODO: if no metadata saving, do not check (look at saving_config)
    ext = ".pt" if saving_config.save_as_tensor else ".nii.gz"
    is_file = os.path.isfile(image_save_path + ext)

    if (is_file and not saving_config.save_file_metadata) or (
        is_file and saving_config.save_file_metadata and os.path.isfile(image_save_path + ".pkl")
    ):
        return
    try:
        image = nib.load(path)
        case, image_props = preprocess_case_without_label(images=[image], **asdict(preprocessing_config), strict=False)
        save_data_and_metadata(case, image_props, image_save_path, saving_config)
        del case, image, image_props
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")
    except Exception as e:
        logging.error(f"Unexpected error {e}: {path}")


def process_dwi_case(
    path,
    bvals_path,
    bvecs_path,
    image_save_path,
    preprocessing_config,
    saving_config,
    use_trace_computation=False,
    strict=True,
):
    try:
        image = nib.load(path)
        method_name = "trace computation" if use_trace_computation else "averaging"
        logging.debug(f"Processing {path} using {method_name} method")

        if len(image.shape) == 4 and image.shape[-1] != 1:
            bvals = np.loadtxt(bvals_path)
            bvecs = np.loadtxt(bvecs_path)

            # Check if bvecs need transposing to (3, N) format
            if bvecs.shape[1] == 3 and bvecs.shape[0] > 3:
                # Convert from (N, 3) to (3, N) format
                bvecs = bvecs.T
            elif bvecs.shape[0] != 3:
                logging.error(f"Invalid bvecs shape: {bvecs.shape}. Expected (3, N) or (N, 3)")
                return

            if not os.path.exists(bvals_path) or not os.path.exists(bvecs_path):
                logging.error(f"SKIPPED: Missing bval or bvec for: {path}")
                return

            images, bvals = extract_3ddwi_from_4ddwi(
                image,
                bvals,
                bvecs,
                use_trace_computation=use_trace_computation,
                strict=strict,
            )
        else:
            if len(image.shape) == 4:  # Must be shape[-1] == 1
                image = nib.Nifti1Image(np.squeeze(image.get_fdata(), axis=-1), image.affine, image.header)
            images = [image]
            bvals = [""]

        for idx, image in enumerate(images):
            # TODO: convoluted/hardcoded way. Find a nice way here
            if image_save_path.endswith(".nii.gz"):
                base = image_save_path[:-7]  # Remove .nii.gz
            elif image_save_path.endswith(".nii"):
                base = image_save_path[:-4]  # Remove .nii
            else:
                base = image_save_path
            if use_trace_computation and bvals[idx] != "" and int(bvals[idx]) > 0:
                filename = f"{base}_bval_{bvals[idx]}_trace"
            else:
                filename = f"{base}_bval_{bvals[idx]}"
            ext = ".pt" if saving_config.save_as_tensor else ".nii.gz"

            if (os.path.isfile(filename + ext) and not saving_config.save_file_metadata) or (
                os.path.isfile(filename + ext) and saving_config.save_file_metadata and os.path.isfile(filename + ".pkl")
            ):
                continue

            case, image_props = preprocess_case_without_label(images=[image], **asdict(preprocessing_config), strict=strict)

            save_data_and_metadata(
                case,
                image_props,
                filename,
                saving_config,
            )
            del case, image, image_props
    except AssertionError as e:
        logging.error(f"AssertionError {e}: {path}")
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")
    except Exception as e:
        logging.error(f"Unexpected error {e}: {path}")


def process_pet_case(path, image_save_path, preprocessing_config, saving_config, strict=True):
    ext = ".pt" if saving_config.save_as_tensor else ".nii.gz"

    if (os.path.isfile(image_save_path + ext) and not saving_config.save_file_metadata) or (
        os.path.isfile(image_save_path + ext) and saving_config.save_file_metadata and os.path.isfile(image_save_path + ".pkl")
    ):
        return
    try:
        image = nib.load(path)
        if len(image.shape) == 4:
            image = extract_3dpet_from_4dpet(image, strict=strict)
        case, image_props = preprocess_case_without_label(images=[image], **asdict(preprocessing_config), strict=False)
        save_data_and_metadata(case, image_props, image_save_path, saving_config)
        del case, image, image_props
    except AssertionError as e:
        logging.error(f"AssertionError {e}: {path}")
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")
    except Exception as e:
        logging.error(f"Unexpected error {e}: {path}")


def process_perf_case(
    path,
    image_save_path,
    m0scan_patterns,
    preprocessing_config,
    saving_config,
    strict=True,
):
    ext = ".pt" if saving_config.save_as_tensor else ".nii.gz"

    if (os.path.isfile(image_save_path + ext) and not saving_config.save_file_metadata) or (
        os.path.isfile(image_save_path + ext) and saving_config.save_file_metadata and os.path.isfile(image_save_path + ".pkl")
    ):
        return
    try:
        image = nib.load(path)

        # Determine if this is an M0 scan based on filename patterns
        filename = os.path.basename(path).lower()
        is_m0scan = any(pattern.lower() in filename for pattern in m0scan_patterns)

        if len(image.shape) == 4:
            image = extract_3dperfusion_from_4dperfusion(image, is_m0scan=is_m0scan, strict=strict)
        case, image_props = preprocess_case_without_label(images=[image], **asdict(preprocessing_config), strict=strict)
        save_data_and_metadata(case, image_props, image_save_path, saving_config)
        del case, image_props
    except AssertionError as e:
        logging.error(f"AssertionError {e}: {path}")
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")
    except Exception as e:
        logging.error(f"Unexpected error {e}: {path}")


def preprocess_case_without_label(
    images: List[Union[np.ndarray, nib.Nifti1Image]],
    normalization_operation: list,
    background_pixel_value: int = 0,
    crop_to_nonzero: bool = True,
    keep_aspect_ratio_when_using_target_size: bool = False,
    image_properties: Optional[dict] = {},
    intensities: Optional[List] = None,
    target_orientation: Optional[str] = "RAS",
    target_size: Optional[List] = None,
    target_spacing: Optional[List] = None,
    strict: bool = True,
    supposed_to_be_3D: bool = True,
    remove_nans=True,
    min_slices=15,
):
    images, label, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images=images,
        original_size=np.array(images[0].shape),
        target_orientation=target_orientation,
        label=None,
        include_header=True,
        strict=strict,
    )

    images = [safe_squeeze(image) for image in images]
    verify_3D_image_is_valid(
        images,
        supposed_to_be_3D=supposed_to_be_3D,
        remove_nans=remove_nans,
        min_slices=min_slices,
    )
    original_size = images[0].shape

    # Cropping is performed to save computational resources. We are only removing background.
    if crop_to_nonzero:
        nonzero_box = get_bbox_for_foreground(images[0], background_label=background_pixel_value)
        image_properties["crop_to_nonzero"] = nonzero_box
        for i in range(len(images)):
            images[i] = crop_to_box(images[i], nonzero_box)
    else:
        image_properties["crop_to_nonzero"] = crop_to_nonzero

    resample_target_size, final_target_size, new_spacing = determine_target_size(
        images=images,
        original_spacing=np.array(image_properties["nifti_metadata"]["original_spacing"]),
        target_size=target_size,
        target_spacing=target_spacing,
        keep_aspect_ratio=keep_aspect_ratio_when_using_target_size,
    )

    images = resample_and_normalize_case(
        case=images,
        target_size=resample_target_size,
        norm_op=normalization_operation,
        intensities=intensities,
        resample_method="yucca",
    )

    if final_target_size is not None:
        images = pad_case_to_size(case=images, size=final_target_size, label=None)

    image_properties["new_size"] = list(images[0].shape)
    image_properties["foreground_locations"] = []
    image_properties["original_spacing"] = image_properties["nifti_metadata"]["original_spacing"]
    image_properties["original_size"] = original_size
    image_properties["original_orientation"] = image_properties["nifti_metadata"]["original_orientation"]
    image_properties["new_spacing"] = new_spacing
    image_properties["new_direction"] = image_properties["nifti_metadata"]["final_direction"]
    return images, image_properties


def preprocess_case_with_label(
    images: List[Union[np.ndarray, nib.Nifti1Image]],
    label: List[Union[np.ndarray, nib.Nifti1Image]],
    normalization_operation: list,
    background_pixel_value: int = 0,
    crop_to_nonzero: bool = True,
    keep_aspect_ratio_when_using_target_size: bool = False,
    image_properties: Optional[dict] = {},
    intensities: Optional[List] = None,
    target_orientation: Optional[str] = "RAS",
    target_size: Optional[List] = None,
    target_spacing: Optional[List] = None,
    strict: bool = True,
    supposed_to_be_3D: bool = True,
    remove_nans=True,
    min_slices=15,
):
    image_properties["pad_box"] = []
    image_properties["crop_box"] = []

    images, label, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images=images,
        original_size=np.array(images[0].shape),
        target_orientation=target_orientation,
        label=label,
        include_header=True,
        strict=strict,
    )

    images = [safe_squeeze(image) for image in images]
    verify_3D_image_is_valid(
        images,
        supposed_to_be_3D=supposed_to_be_3D,
        remove_nans=remove_nans,
        min_slices=min_slices,
    )
    original_size = images[0].shape

    # Cropping is performed to save computational resources. We are only removing background.
    if crop_to_nonzero:
        nonzero_box = get_bbox_for_foreground(images[0], background_label=background_pixel_value)
        for i in range(len(images)):
            images[i] = crop_to_box(images[i], nonzero_box)
        label = crop_to_box(label, nonzero_box)
        image_properties["crop_box"] = [int(i) for i in nonzero_box]

    image_properties["size_before_resample"] = images[0].shape
    resample_target_size, final_target_size, new_spacing = determine_target_size(
        images=images,
        original_spacing=np.array(image_properties["nifti_metadata"]["original_spacing"]),
        target_size=target_size,
        target_spacing=target_spacing,
        keep_aspect_ratio=keep_aspect_ratio_when_using_target_size,
    )

    images, label = resample_and_normalize_case(
        case=images,
        label=label,
        target_size=resample_target_size,
        norm_op=normalization_operation,
        intensities=intensities,
        resample_method="yucca",
    )

    if final_target_size is not None:
        image_properties["shape_before_pad"] = images[0].shape
        images, pad_box = pad_case_to_size(case=images, size=final_target_size, label=label)
        image_properties["pad_box"] = pad_box

    image_properties["foreground_locations"] = get_foreground_locations(label=label, per_class=True)
    image_properties["new_size"] = list(images[0].shape)
    image_properties["original_spacing"] = image_properties["nifti_metadata"]["original_spacing"]
    image_properties["original_size"] = original_size
    image_properties["original_orientation"] = image_properties["nifti_metadata"]["original_orientation"]
    image_properties["new_spacing"] = new_spacing
    image_properties["new_direction"] = image_properties["nifti_metadata"]["final_direction"]
    return images, label, image_properties


def verify_3D_image_is_valid(
    images: list,
    supposed_to_be_3D: bool = True,
    remove_nans: bool = True,
    min_slices: int = 15,
):
    for image in images:
        if supposed_to_be_3D and len(image.shape) != 3:
            raise ValueError(f"image is not 3D. Shape: {image.shape}.  ")
        if np.min(image.shape) < min_slices:
            raise ValueError(f"image is too small. Shape: {image.shape}. ")
        if np.count_nonzero(image) < 1:
            raise ValueError("image is all zeros. ")
        if remove_nans:
            if np.any(np.isnan(image)):
                raise ValueError("image contains NaN values. ")


def extract_3dpet_from_4dpet(image, strict=True):
    if strict:
        assert np.min(image.shape) == image.shape[-1], (
            f"Min shape not last dimension for PET. Set strict to False to allow this. Found shape {image.shape}"
        )
    image_arr = np.mean(image.get_fdata(), axis=-1)
    header = image.header.copy()
    header.set_data_shape(image_arr.shape)
    image = nib.Nifti1Image(image_arr, image.affine, header=header)
    return image


def extract_3ddwi_from_4ddwi(image, bvals, bvecs, bval_tolerance=50, strict=True, use_trace_computation=False):
    """
    Extract 3D DWI from 4D DWI data.
    The use_trace_computation parameter determines the processing method for the entire image.
    """
    if strict:
        assert np.min(image.shape) == image.shape[-1], (
            f"Min shape not last dimension for DWI. Set strict to False to allow this. Found shape {image.shape}"
        )

    old_header = image.header.copy()

    # Group similar b-values
    bval_groups = group_bvalues(bvals, tolerance=bval_tolerance)

    dwis = []
    group_bvals = []

    for group_bval, indices in bval_groups:
        if group_bval == 0:
            # B0 images are always processed the same way
            dwi = get_data_for_bval_group(group_bval, indices, bvals, bvecs, image.get_fdata(), use_trace=False)
        else:
            dwi = get_data_for_bval_group(
                group_bval,
                indices,
                bvals,
                bvecs,
                image.get_fdata(),
                use_trace=use_trace_computation,
                bval_groups=bval_groups,
            )

        if dwi is not None:  # Only add if we got data (not skipped)
            new_header = old_header.copy()
            new_header.set_data_shape(dwi.shape)
            dwi = nib.Nifti1Image(dwi, image.affine, header=new_header)
            dwis.append(dwi)
            group_bvals.append(str(int(round(group_bval))))
        else:
            logging.error(f"Skipped b-value group: {group_bval} (processing failed)")

    return dwis, group_bvals


def get_best_basis(bvecs):
    """Finds the indices of the three bvecs closest to X, Y, and Z directions."""
    standard_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    best_match = []
    max_cosines = []

    for std_vec in standard_basis:
        best_idx = None
        max_cosine = -1
        for i in range(bvecs.shape[1]):
            norm_bvec = np.linalg.norm(bvecs[:, i])
            if norm_bvec == 0:
                continue
            cos_sim = np.dot(bvecs[:, i], std_vec) / norm_bvec
            if cos_sim > max_cosine:
                max_cosine = cos_sim
                best_idx = i
        best_match.append(best_idx)
        max_cosines.append(max_cosine)

    if None in best_match:
        logging.warning(f"Could not find basis vector for some directions. Found indices: {best_match}")

    assert len(best_match) == 3
    return best_match


def group_bvalues(bvals, tolerance=50, b0_threshold=5):
    """
    Group similar b-values together. First groups b0s (including near-zero values), then remaining values.

    Args:
        bvals: Array of b-values
        tolerance: Tolerance for grouping similar non-zero b-values
        b0_threshold: B-values <= this threshold are treated as b0
    """
    groups = []

    # Find b0 and near-b0 indices (b-values <= b0_threshold)
    b0_indices = np.where(bvals <= b0_threshold)[0]
    if len(b0_indices) > 0:
        # Use the first b0/near-b0 volume as reference
        groups.append((0, b0_indices[:1]))
        logging.debug(f"Found {len(b0_indices)} b0/near-b0 volumes (b<={b0_threshold}), using first one as reference")

    # Process non-b0 values
    non_b0_mask = bvals > b0_threshold
    non_b0_bvals = bvals[non_b0_mask]

    if len(non_b0_bvals) > 0:
        unique_non_b0 = np.unique(non_b0_bvals)
        used_indices = set(b0_indices[:1]) if len(b0_indices) > 0 else set()

        for bval in sorted(unique_non_b0):
            similar_indices = np.where((bvals >= bval - tolerance) & (bvals <= bval + tolerance) & non_b0_mask)[0]
            available_indices = [idx for idx in similar_indices if idx not in used_indices]

            if len(available_indices) >= 3:
                group_representative = np.mean(bvals[available_indices])
                groups.append((group_representative, np.array(available_indices)))
                used_indices.update(available_indices)

    if len(groups) == 0:
        logging.warning("No valid b-value groups found")
    elif len(groups) == 1 and groups[0][0] == 0:
        logging.debug("Only b0/near-b0 volumes found, no orthogonal diffusion directions")

    return groups


def get_data_for_bval_group(group_bval, indices, bvals, bvecs, data, use_trace=False, bval_groups=None):
    """Extract data for a group of similar b-values."""
    if group_bval == 0:
        return data[..., indices[0]]

    if len(indices) < 3:
        logging.warning(f"Insufficient volumes for b-value group {group_bval}: {len(indices)} < 3")
        return None

    try:
        basis_indices = get_best_basis(bvecs[:, indices])
        if None in basis_indices:
            logging.warning(f"Failed to find complete basis for b-value group {group_bval}")
            return None

        selected_volumes = data[..., [indices[i] for i in basis_indices]]

        if not use_trace:
            # Original method: simple averaging
            logging.debug(f"Using averaging method for b-value {group_bval}")
            return np.mean(selected_volumes, axis=-1)
        else:
            # Trace computation method
            logging.debug(f"Using trace computation method for b-value {group_bval}")
            # Pass the individual b-values for each selected volume
            individual_bvals = [bvals[indices[i]] for i in basis_indices]
            return compute_trace_adc(selected_volumes, individual_bvals, bval_groups, data)

    except Exception as e:
        logging.warning(f"Error processing b-value group {group_bval}: {e}")
        return None


def extract_3dperfusion_from_4dperfusion(image, is_m0scan=False, strict=True):
    if strict:
        assert np.min(image.shape) == image.shape[-1], (
            f"Min shape not last dimension for DWI. Set strict to False to allow this. Found shape {image.shape}"
        )

    data = image.get_fdata()
    tp = data.shape[-1]  # number of time points

    if is_m0scan:
        # For M0 scans, take the average across all the time points
        logging.debug(f"Processing M0 scan with {tp} time points - taking average")
        processed_data = np.mean(data, axis=-1)
    else:
        if tp <= 3:
            # For tp <= 3, take the difference between first and last
            logging.debug(f"Processing ASL scan with {tp} time points - taking difference between first and last")
            processed_data = data[..., -1] - data[..., 0]
        else:
            # For tp > 3, take the average
            logging.debug(f"Processing ASL scan with {tp} time points - taking average")
            processed_data = np.mean(data, axis=-1)

    # Create new 3D image with updated header
    header = image.header.copy()
    header.set_data_shape(processed_data.shape)
    processed_image = nib.Nifti1Image(processed_data, image.affine, header=header)

    return processed_image


def compute_trace_adc(selected_volumes, individual_bvals, bval_groups, full_data):
    """
    Compute trace ADC using the formula: Trace = ADCx + ADCy + ADCz
    where ADC = -(1/b) * ln(S/S0)

    Args:
        selected_volumes: 4D array with shape (..., 3) containing the 3 basis directions
        individual_bvals: List of 3 individual b-values for each direction
        bval_groups: List of (bval, indices) tuples from grouping
        full_data: Full 4D data array
    """
    try:
        # Find b0/near-b0 data (b-value group with value 0, which includes near-zero values)
        b0_data = None
        if bval_groups is not None:
            for bval, indices in bval_groups:
                if bval == 0:
                    b0_data = full_data[..., indices[0]]
                    break

        if b0_data is None:
            logging.warning("No b0/near-b0 found for trace computation, reverting to averaging")
            return np.mean(selected_volumes, axis=-1)

        # Compute ADC for each direction (x, y, z)
        adc_components = []

        for i in range(selected_volumes.shape[-1]):  # For each direction
            S = selected_volumes[..., i]  # Signal for this direction
            S0 = b0_data  # Reference b0/near-b0 signal
            bval = individual_bvals[i]  # Individual b-value for this direction

            # Avoid division by zero and negative/zero values for log
            # Create mask for valid computations
            valid_mask = (S0 > 0) & (S > 0) & (S <= S0)

            # Initialize ADC array
            adc = np.zeros_like(S, dtype=np.float32)

            # Compute ADC only for valid voxels
            if np.any(valid_mask):
                ratio = S[valid_mask] / S0[valid_mask]
                # Ensure ratio is positive and <= 1 for log
                ratio = np.clip(ratio, 1e-10, 1.0)
                adc[valid_mask] = -(1.0 / bval) * np.log(ratio)  # Use individual bval

            # Handle invalid voxels
            invalid_mask = ~valid_mask
            if np.any(invalid_mask):
                logging.debug(f"Found {np.sum(invalid_mask)} invalid voxels for direction {i} with b-value {bval}")
                # Set invalid voxels to 0 or small positive value
                adc[invalid_mask] = 0.0

            adc_components.append(adc)

        # Compute trace as sum of ADC components
        trace_adc = np.sum(adc_components, axis=0)

        logging.debug(
            f"Computed trace ADC using individual b-values {individual_bvals}, "
            f"mean trace: {np.mean(trace_adc[trace_adc > 0]):.6f}"
        )

        return trace_adc

    except Exception as e:
        logging.error(f"Error in trace computation: {e}")
        logging.info("Reverting to averaging method")
        return np.mean(selected_volumes, axis=-1)


def determine_target_size(
    images: list,
    original_spacing,
    target_size,
    target_spacing,
    keep_aspect_ratio,
):
    image_shape = np.array(images[0].shape)

    # We do not want to change the aspect ratio so we resample using the minimum alpha required
    # to attain 1 correct dimension, and then the rest will be padded.
    # Additionally we make sure each dimension is divisible by 16 to avoid issues with standard pooling/stride settings
    if target_size is not None:
        resample_target_size, final_target_size, new_spacing = determine_resample_size_from_target_size(
            current_size=image_shape,
            current_spacing=original_spacing,
            target_size=target_size,
            keep_aspect_ratio=keep_aspect_ratio,
        )

    # Otherwise we need to calculate a new target shape, and we need to factor in that
    # the images will first be transposed and THEN resampled.
    # Find new shape based on the target spacing
    elif target_spacing is not None:
        target_spacing = np.array(target_spacing, dtype=float)
        resample_target_size, final_target_size, new_spacing = determine_resample_size_from_target_spacing(
            current_size=image_shape,
            current_spacing=original_spacing,
            target_spacing=target_spacing,
        )
    else:
        resample_target_size = image_shape
        final_target_size = None
        new_spacing = original_spacing.tolist()
    return resample_target_size, final_target_size, new_spacing


def determine_resample_size_from_target_size(current_size, current_spacing, target_size, keep_aspect_ratio: bool = False):
    if keep_aspect_ratio:
        resample_target_size = np.array(current_size * np.min(target_size / current_size)).astype(int)
        final_target_size = target_size
        final_target_size = [math.ceil(i / 16) * 16 for i in final_target_size]
    else:
        resample_target_size = target_size
        resample_target_size = [math.ceil(i / 16) * 16 for i in resample_target_size]
        final_target_size = None
    new_spacing = (
        (np.array(resample_target_size).astype(float) / current_size.astype(float)) * np.array(current_spacing).astype(float)
    ).tolist()
    return resample_target_size, final_target_size, new_spacing


def determine_resample_size_from_target_spacing(
    current_size: np.ndarray,
    current_spacing: np.ndarray,
    target_spacing: np.ndarray,
) -> tuple[np.ndarray, None, list[float]]:
    resample_target_size = np.round((current_spacing / target_spacing) * current_size).astype(int)
    return resample_target_size, None, target_spacing.tolist()


def get_foreground_locations(
    label: np.ndarray,
    per_class: bool = False,
    max_locs_total: int = 100_000,
) -> dict[str, list]:
    foreground_locations: dict[str, list] = {}

    if not per_class:
        locs = np.array(np.nonzero(label)).T[::10].tolist()
        if locs:
            if len(locs) > max_locs_total:
                step = round(len(locs) / max_locs_total)
                locs = locs[::step]
            foreground_locations["1"] = locs
    else:
        classes = np.unique(label)[1:]
        if len(classes) == 0:
            return foreground_locations
        max_per_class = max_locs_total // len(classes)
        for c in classes:
            locs = np.array(np.where(label == int(c))).T[::10]
            if len(locs) > 0:
                if len(locs) > max_per_class:
                    locs = locs[:: round(len(locs) / max_per_class)]
                foreground_locations[str(int(c))] = locs
    return foreground_locations


def safe_squeeze_leading(array):
    if array.shape[0] == 1:
        return array.squeeze(axis=0)
    return array


def safe_squeeze_trailing(array):
    if array.shape[-1] == 1:
        return array.squeeze(axis=-1)
    return array


def safe_squeeze(array):
    array = safe_squeeze_leading(array)
    array = safe_squeeze_trailing(array)
    return array
