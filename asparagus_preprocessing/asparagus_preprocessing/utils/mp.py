import logging
from asparagus_preprocessing.utils.process_case import (
    process_dwi_case,
    process_mri_case,
    process_perf_case,
    process_pet_case,
)
from itertools import repeat
from multiprocessing.pool import Pool


def multiprocess_mri_dwi_pet_perf_cases(
    files_standard,
    files_standard_out,
    files_DWI,
    bvals_DWI,
    bvecs_DWI,
    files_DWI_out,
    files_PET,
    files_PET_out,
    files_Perf,
    files_Perf_out,
    patterns_m0,
    preprocessing_config,
    saving_config,
    strict=True,
    processes=12,
    chunksize=1,
):
    logging.info(f"Starting multiprocessing for MRI/standard. Number of files: {len(files_standard)}")
    if len(files_standard) > 0:
        process_mri_case(
            files_standard[0],
            files_standard_out[0],
            preprocessing_config,
            saving_config,
        )

    with Pool(processes) as pool:
        pool.starmap(
            process_mri_case,
            zip(
                files_standard,
                files_standard_out,
                repeat(preprocessing_config),
                repeat(saving_config),
            ),
            chunksize=chunksize,
        )

    logging.info(f"Starting multiprocessing for DWI. Number of files: {len(files_DWI)}")
    # TODO: Maybe have a % split?
    # Create deterministic 50/50 split for DWI processing methods
    # Sort files to ensure reproducible ordering, then alternate methods
    sorted_files_with_indices = sorted(enumerate(files_DWI), key=lambda x: x[1])

    # Alternate: even positions get averaging, odd positions get trace computation
    trace_flags_ordered = [False] * len(files_DWI)

    for i, (orig_idx, _) in enumerate(sorted_files_with_indices):
        if i % 2 == 1:
            trace_flags_ordered[orig_idx] = True

    trace_count = sum(trace_flags_ordered)
    avg_count = len(trace_flags_ordered) - trace_count
    logging.info(f"DWI processing: {avg_count} files will use averaging, {trace_count} files will use trace computation")
    if len(files_DWI) > 0:
        process_dwi_case(
            files_DWI[0],
            bvals_DWI[0],
            bvecs_DWI[0],
            files_DWI_out[0],
            preprocessing_config,
            saving_config,
            trace_flags_ordered[0],
            strict,
        )
    with Pool(processes) as pool:
        pool.starmap(
            process_dwi_case,
            zip(
                files_DWI,
                bvals_DWI,
                bvecs_DWI,
                files_DWI_out,
                repeat(preprocessing_config),
                repeat(saving_config),
                trace_flags_ordered,
                repeat(strict),
            ),
            chunksize=chunksize,
        )

    logging.info(f"Starting multiprocessing for PET. Number of files: {len(files_PET)}")
    if len(files_PET) > 0:
        process_pet_case(files_PET[0], files_PET_out[0], preprocessing_config, saving_config, strict)
    with Pool(processes) as pool:
        pool.starmap(
            process_pet_case,
            zip(
                files_PET,
                files_PET_out,
                repeat(preprocessing_config),
                repeat(saving_config),
                repeat(strict),
            ),
            chunksize=chunksize,
        )

    logging.info(f"Starting multiprocessing for Perfusion. Number of files: {len(files_Perf)}")
    if len(files_Perf) > 0:
        process_perf_case(
            files_Perf[0],
            files_Perf_out[0],
            patterns_m0,
            preprocessing_config,
            saving_config,
            strict,
        )
    with Pool(processes) as pool:
        pool.starmap(
            process_perf_case,
            zip(
                files_Perf,
                files_Perf_out,
                repeat(patterns_m0),
                repeat(preprocessing_config),
                repeat(saving_config),
                repeat(strict),
            ),
            chunksize=chunksize,
        )


def process_dataset_without_table(
    process_fn,
    files_in,
    files_out,
    dataset_config,
    preprocessing_config,
    saving_config,
    processes,
):
    process_fn(files_in[0], files_out[0], dataset_config, preprocessing_config, saving_config)  # dry run to catch errors early
    with Pool(processes) as pool:
        pool.starmap(
            process_fn,
            zip(
                files_in,
                files_out,
                repeat(dataset_config),
                repeat(preprocessing_config),
                repeat(saving_config),
            ),
        )


def process_dataset_with_table(
    process_fn,
    files_in,
    files_out,
    dataset_config,
    preprocessing_config,
    saving_config,
    table_in,
    table_out,
    processes,
):
    for i in range(5):
        process_fn(
            files_in[i],
            files_out[i],
            dataset_config,
            preprocessing_config,
            saving_config,
            table_in,
            table_out,
        )

    with Pool(processes) as pool:
        pool.starmap(
            process_fn,
            zip(
                files_in,
                files_out,
                repeat(dataset_config),
                repeat(preprocessing_config),
                repeat(saving_config),
                repeat(table_in),
                repeat(table_out),
            ),
        )
