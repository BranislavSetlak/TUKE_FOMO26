import torchvision
import torch
from asparagus.functional.loading import load_image_file
from asparagus.functional.sequence_labels import sequence_class_id
from torch.utils.data import Dataset
from typing import Mapping, Optional


class PretrainDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
        sequence_raw_to_class: Optional[Mapping[str, int]] = None,
        sequence_ignored: Optional[Mapping[str, int]] = None,
        sequence_other_class_id: Optional[int] = None,
    ):
        super().__init__()

        self.files = files
        self.transforms = transforms
        self.sequence_raw_to_class = dict(sequence_raw_to_class or {})
        self.sequence_ignored = dict(sequence_ignored or {})
        self.sequence_other_class_id = sequence_other_class_id

        sequence_args = (
            bool(self.sequence_raw_to_class),
            self.sequence_other_class_id is not None,
        )
        if any(sequence_args) and not all(sequence_args):
            raise ValueError(
                "sequence_raw_to_class and sequence_other_class_id must either both be set or both be omitted"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = load_image_file(file)
        data_dict = {"file_path": file, "image": data, "transforms_applied": {}}
        if self.sequence_raw_to_class:
            data_dict["sequence_label"] = torch.tensor(
                sequence_class_id(
                    file,
                    raw_to_class=self.sequence_raw_to_class,
                    ignored_sequences=self.sequence_ignored,
                    other_class_id=self.sequence_other_class_id,
                ),
                dtype=torch.long,
            )
        data_dict = self._transform(data_dict)  # CPU transforms only here
        return data_dict

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict
