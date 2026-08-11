import copy
import torch
from asparagus.modules.networks.vision_transformer import MaskedVisionTransformer
from asparagus.modules.transforms.blockmask import random_block_mask
from functools import partial
from lightly.models.modules.heads import DINOv2ProjectionHead
from lightly.utils.scheduler import cosine_schedule
from monai.networks.nets.vit import ViT
from torch import nn


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


def freeze_eval_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2(nn.Module):
    """
    3D DINOv2 model with teacher/student ViT backbones and DINO/iBOT heads.
    Supports block masking and teacher-student EMA updates.
    """

    def __init__(
        self,
        input_channels: int = 1,
        img_size: tuple = (64, 64, 64),
        patch_size: tuple = (16, 16, 16),
        hidden_size: int = 768,
        norm_last_layer: bool = False,
        ibot_separate_head: bool = False,
        freeze_last_layer: int = -1,
        projection_dim: int = 65536,
        mask_ratio: float = 0.3,
    ):
        """
        Initialize DINOv2_3D model.
        Args: see model config for details.
        """
        super().__init__()
        self.projection_dim = projection_dim
        self.norm_last_layer = norm_last_layer
        self.ibot_separate_head = ibot_separate_head

        self.hidden_size = hidden_size
        vit = partial(
            ViT,
            in_channels=input_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=3072,
            num_layers=8,
            num_heads=12,
            proj_type="conv",
            classification=True,
            spatial_dims=len(img_size),
        )

        # Initialize the student first. The EMA teacher must start as an exact
        # copy; independently initialized targets destabilize early training.
        self.student_backbone = MaskedVisionTransformer(vit=vit())
        # Unfreeze student
        for param in self.student_backbone.parameters():
            param.requires_grad = True
        self.student_backbone.train()

        def make_projection_head():
            return DINOv2ProjectionHead(
                input_dim=self.hidden_size,
                output_dim=projection_dim,
            )

        self.student_dino_head = make_projection_head()

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        # DINOv2ProjectionHead uses legacy torch weight_norm. Such modules
        # cannot be deep-copied on PyTorch 2.6 because the computed weight is
        # not a graph leaf. Instantiate an independent module, then copy its
        # complete parameter/buffer state instead.
        self.teacher_dino_head = make_projection_head()
        self.teacher_dino_head.load_state_dict(self.student_dino_head.state_dict())
        freeze_eval_module(self.teacher_backbone)
        freeze_eval_module(self.teacher_dino_head)

        if self.ibot_separate_head:
            self.student_ibot_head = make_projection_head()
            self.teacher_ibot_head = make_projection_head()
            self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())
            freeze_eval_module(self.teacher_ibot_head)
        else:
            self.teacher_ibot_head = self.teacher_dino_head
            self.student_ibot_head = self.student_dino_head

        self.random_block_mask = random_block_mask(
            grid_size=self.student_backbone.grid_size,
            sequence_length=self.student_backbone.sequence_length,
            mask_ratio=mask_ratio,
        )

    def train(self, mode: bool = True):
        """Train the student while keeping all EMA teacher modules in eval mode."""
        super().train(mode)
        self.teacher_backbone.eval()
        self.teacher_dino_head.eval()
        self.teacher_ibot_head.eval()
        return self

    def forward_teacher(self, x):
        """Forward pass for the EMA (teacher) backbone without masking."""
        features = self.teacher_backbone(x)
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:] if features.shape[1] > 1 else features
        return cls_token, patch_tokens

    def forward_student(self, x, mask=None):
        """Forward pass for the student backbone, keeping patch grid structure."""
        features = self.student_backbone(x, mask=mask)
        cls_tokens = features[:, 0]
        patch_tokens = features[:, 1:] if features.shape[1] > 1 else features
        return cls_tokens, patch_tokens

    def update_teacher(self, global_step: int, max_steps: int) -> None:
        """Update teacher using EMA with cosine momentum schedule."""
        momentum = cosine_schedule(step=global_step, max_steps=max_steps, start_value=0.992, end_value=1.0)

        # Remove problematic device movement logic
        # In DDP, parameters should already be on correct devices
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_dino_head, self.teacher_dino_head, m=momentum)
        if self.ibot_separate_head:
            update_momentum(self.student_ibot_head, self.teacher_ibot_head, m=momentum)

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel gradients in the last layer during warmup."""
        self.student_dino_head.cancel_last_layer_gradients(current_epoch)
        if self.ibot_separate_head:
            self.student_ibot_head.cancel_last_layer_gradients(current_epoch)

    def forward(self, global_views: list[torch.Tensor], local_views: list[torch.Tensor] = None) -> dict:
        """
        Forward pass for DINOv2 3D with block masking and multi-view augmentations.
        Args:
            global_views: List of augmented 3D tensors (global views)
            local_views: List of augmented 3D tensors (local views)
        Returns:
            Dict of teacher/student outputs for loss computation
        """
        device = global_views[0].device
        global_views = torch.cat(global_views)

        n_local_views = len(local_views) if local_views is not None else 0
        if n_local_views > 0:
            local_views = torch.cat(local_views)
        block_mask = self.random_block_mask(batch_size=global_views.shape[0], device=device)
        patch_mask = block_mask.flatten(start_dim=1)
        mask = torch.zeros((global_views.shape[0], self.student_backbone.sequence_length), device=device, dtype=torch.bool)
        mask[:, 1:] = patch_mask

        with torch.no_grad():
            teacher_cls_token, teacher_patch_tokens = self.forward_teacher(global_views)
            teacher_cls_token = self.teacher_dino_head(teacher_cls_token)
            teacher_patch_tokens = self.teacher_ibot_head(teacher_patch_tokens[patch_mask])

        # Student forward
        student_global_cls_token, student_global_patch_tokens = self.forward_student(global_views, mask=mask)
        student_global_cls_token = self.student_dino_head(student_global_cls_token)
        student_global_patch_tokens = self.student_ibot_head(student_global_patch_tokens[patch_mask])

        # Local views
        if local_views is not None:
            student_local_cls_token, _ = self.forward_student(local_views, mask=None)
            student_local_cls_token = self.student_dino_head(student_local_cls_token)
            student_cls_token = torch.cat([student_global_cls_token, student_local_cls_token], dim=0)
        else:
            student_cls_token = student_global_cls_token

        out = {
            "teacher_cls_token": teacher_cls_token,
            "student_cls_token": student_cls_token,
            "teacher_patch_tokens": teacher_patch_tokens,
            "student_patch_tokens": student_global_patch_tokens,
            "student_glob_cls_token": student_global_cls_token,
            "mask": patch_mask,
            "n_local_views": torch.tensor(n_local_views, device=device),
        }

        return {"pred": out}

    def encode(self, x: torch.Tensor):
        """
        Simple encoding method that returns the raw CLS token features.
        Useful for feature extraction or as input to downstream tasks.
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            CLS token features of shape (B, hidden_size)
        """
        backbone_features = self.student_backbone(x, mask=None)
        return backbone_features[:, 0]  # Return CLS token only


def vit_test(input_channels=1, img_size=(1, 1, 1), patch_size=(0, 0, 0), hidden_size=120):
    return DINOv2(
        input_channels=input_channels,
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        norm_last_layer=False,
        ibot_separate_head=False,
        projection_dim=1024,
    )


def vit_x(
    input_channels=1,
    img_size=(92, 92, 92),
    patch_size=(8, 8, 8),
    hidden_size=192,
    ibot_separate_head=False,
    projection_dim=65536,
    mask_ratio=0.3,
):
    return DINOv2(
        input_channels=input_channels,
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        norm_last_layer=False,
        ibot_separate_head=ibot_separate_head,
        projection_dim=projection_dim,
        mask_ratio=mask_ratio,
    )