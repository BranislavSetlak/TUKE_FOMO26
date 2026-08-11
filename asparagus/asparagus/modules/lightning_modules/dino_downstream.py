"""Small downstream Lightning additions specific to the DINO evaluation."""

import torch

from asparagus.modules.lightning_modules.clsreg_module import ClassificationModule


class DINOClassificationModule(ClassificationModule):
    """Classification module that also writes probabilities for test AUROC."""

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        probabilities = torch.softmax(outputs.float(), dim=1)
        prediction = outputs.argmax(1).long()
        label = batch["CLSREG_label"]
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "probabilities": probabilities.squeeze(0).detach().cpu().tolist(),
            "label": label.item(),
        }
        self.predictions.append(prediction)
        self.labels.append(label)
