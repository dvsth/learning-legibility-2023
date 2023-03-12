import torch
from transformers import Trainer
from classes import LegibilityModel

class MultiTaskTrainer(Trainer):
    '''
    Overrides the default HuggingFace Trainer class
    '''
    
    def training_step(self, model: LegibilityModel.LegibilityModel, inputs):
        # get the input images and target label from the data dictionary
        imgs0, imgs1, labels = inputs['img0'], inputs['img1'], inputs['choice']
        # run the model
        scores0 = model(imgs0)
        scores1 = model(imgs1)
        # compute the loss
        loss = self.compute_loss(model, scores0, scores1, labels)
        loss.backward()

        return loss.detach()

    def eval_step(self, model: LegibilityModel.LegibilityModel, inputs):
        with torch.no_grad():
            # get the input images and target label from the data dictionary
            imgs0, imgs1, labels = inputs['img0'], inputs['img1'], inputs['choice']
            # run the model
            scores0 = model(imgs0)
            scores1 = model(imgs1)
            # compute the loss
            loss = self.compute_loss(model, scores0, scores1, labels)
            return loss

    def prediction_step(self, model: LegibilityModel.LegibilityModel, inputs, prediction_loss_only=True, ignore_keys=None):
       with torch.no_grad():
            # get the input images and target label from the data dictionary
            imgs0, imgs1, labels = inputs['img0'], inputs['img1'], inputs['choice']
            # run the model
            scores0 = model(imgs0)
            scores1 = model(imgs1)
            # compute the loss
            loss = self.compute_loss(model, scores0, scores1, labels)
            return loss, [scores0, scores1], labels

    def compute_loss(self, model: LegibilityModel.LegibilityModel, scores0: torch.Tensor, scores1: torch.Tensor, labels: torch.Tensor, return_outputs=False):
        # labels:
        # 0 or 1: word 0 or 1 is more legible, other unknown
        # 2: both words are equally legible
        # 3: neither word is legible

        contrastive_term = torch.binary_cross_entropy_with_logits(
            scores0 - scores1, (labels == 0).type(torch.float))
        word0_term = torch.binary_cross_entropy_with_logits(
            scores0, torch.logical_or(labels == 0, labels == 2).type(torch.float))
        word1_term = torch.binary_cross_entropy_with_logits(
            scores1, torch.logical_or(labels == 1, labels == 2).type(torch.float))

        # mask out terms which are not relevant for the loss
        mask_c = labels < 2
        mask_0 = torch.logical_or(torch.logical_or(labels == 0, labels == 2), labels == 3)
        mask_1 = labels > 0

        # compute the loss
        loss = mask_c * contrastive_term + mask_0 * word0_term + mask_1 * word1_term
        return loss.mean()