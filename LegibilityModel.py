import torch.nn as nn
from transformers import VisionEncoderDecoderModel, PreTrainedModel, AutoConfig

class LegibilityModel(PreTrainedModel):
    def __init__(self, config):
        config = AutoConfig.from_pretrained("microsoft/trocr-base-handwritten")
        super(LegibilityModel, self).__init__(config=config)

        # base model architecture
        self.model = VisionEncoderDecoderModel(config).encoder

        # change dropout during training
        self.stack = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(768, 1)
        )

    # choice, img0, img1 are not used by the model, but are passed by the trainer
    def forward(self, img_batch, choice, img0, img1):
        output = self.model(img_batch)
        # average the output of the last hidden layer
        output = output.last_hidden_state.mean(dim=1)
        scores = self.stack(output)
        return scores