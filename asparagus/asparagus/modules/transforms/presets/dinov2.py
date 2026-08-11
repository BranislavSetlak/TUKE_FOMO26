from asparagus.modules.transforms.DinoV2 import DINOv2Augmentation
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from torchvision.transforms import Compose


def dinov2(global_view_size, local_view_size, patch_size):
    """
    CPU preprocessing and multi-crop augmentation for 3D MRI DINO training.

    RandScaleCrop interprets scale as a fraction of each spatial dimension,
    rather than as a volume fraction. The values below therefore approximate:

        global volume fraction: 0.32--1.00
        local volume fraction:  0.05--0.32
    """
    return Compose(
        [
            Torch_Normalize(normalize=True),
            DINOv2Augmentation(
                global_view_scale=[0.68, 1.0],
                global_view_size=global_view_size,
                local_view_scale=[0.37, 0.68],
                local_view_size=local_view_size,
                num_local_views=4,
            ),
        ]
    )
