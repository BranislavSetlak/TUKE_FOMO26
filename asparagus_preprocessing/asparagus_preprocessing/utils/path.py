import json
import logging
import os

logger = logging.getLogger(__name__)


def prepare_target_dir(target_dir: str, save_as_tensor: bool) -> None:
    """Create the target directory, raising an error if a previous run used a different format.

    Checks whether a ``dataset.json`` already exists in *target_dir*.  If it
    does, the ``saving_config.save_as_tensor`` value recorded there must match
    the current *save_as_tensor* flag.  A mismatch means the user is about to
    mix ``.pt`` and ``.nii.gz`` files under the same task name, which is not
    allowed.
    """
    dataset_json_path = os.path.join(target_dir, "dataset.json")
    if os.path.isfile(dataset_json_path):
        with open(dataset_json_path, "r") as f:
            existing = json.load(f)
        existing_flag = existing.get("saving_config", {}).get("save_as_tensor")
        if existing_flag is not None and existing_flag != save_as_tensor:
            existing_fmt = ".pt" if existing_flag else ".nii.gz"
            requested_fmt = ".pt" if save_as_tensor else ".nii.gz"
            raise ValueError(
                f"Format conflict in '{target_dir}': existing dataset.json has "
                f"save_as_tensor={existing_flag} ({existing_fmt}) but this run "
                f"requests save_as_tensor={save_as_tensor} ({requested_fmt}). "
                f"Delete the target directory or use a different task name to "
                f"reprocess with a different format."
            )
        logger.info("Resuming preprocessing in '%s' (save_as_tensor=%s)", target_dir, save_as_tensor)
    os.makedirs(target_dir, exist_ok=True)


def get_image_output_paths(files, source_dir, target_dir, file_suffix=[".nii", ".nii.gz"]):
    """Generates image file paths files and metadata paths with the input structure in the output directory"""
    files_out = []
    for f in files:
        file_out = f.replace(source_dir, target_dir)
        # Replace any matching file suffix with the image suffix
        for suffix in file_suffix:
            if file_out.endswith(suffix):
                file_out = file_out[: -len(suffix)]
                break
        files_out.append(file_out)

    return files_out
