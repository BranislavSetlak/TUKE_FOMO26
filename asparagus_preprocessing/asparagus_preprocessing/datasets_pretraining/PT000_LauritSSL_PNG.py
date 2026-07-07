import os
import torch
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils.dataset_registration import register_dataset
from asparagus_preprocessing.utils.path import prepare_target_dir
from itertools import repeat
from multiprocessing.pool import Pool
from PIL import Image


def main(
    path: str = "",
    subdir: str = "",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=False,
):
    task_name = "PT000_LauritSSL_PNG"
    target_dir = os.path.join(get_data_path(), task_name)
    prepare_target_dir(target_dir, False)

    torch.manual_seed(421890)

    p = Pool(processes)
    p.starmap(
        generate_random_ptcase,
        zip(
            range(250),
            repeat(target_dir),
        ),
    )
    p.close()
    p.join()
    register_dataset(
        task_name=task_name,
        n_classes=1,
        n_modalities=1,
        data_dir=target_dir,
        extension=".png",
        n_folds=1,
    )


def generate_random_ptcase(i, target_dir):
    dims = (
        torch.randint(low=13, high=480, size=(1,)),
        torch.randint(low=60, high=360, size=(1,)),
    )
    image = torch.FloatTensor(*dims).uniform_(-1.1e16, 1.1e16).numpy()
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image.save(os.path.join(target_dir, f"case_{i:03d}.png"))


if __name__ == "__main__":
    main()
