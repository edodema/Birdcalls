from torch import nn
from src.pl_modules.model import Extraction


class BirdcallModel(nn.Module):
    def __init__(self, out_features: int, **kwargs):
        super().__init__()

        self.ext = Extraction(
            image_shape=(1, 128, 3751),
            att_channels=[1, 3, 1],
            att_kernels=[1, 3, 1],
            att_paddings=[0, 1, 0],
            att_strides=[1, 1, 1],
            att_num_heads=1,
            pool_att=1,
            res_kernels=[5, 5, 5],
            pool=3,
        )

        self.gru = nn.GRU(
            input_size=4960,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            dropout=0,  # Dropout should be 0 when there is only one layer.
        )

        self.fc = nn.Linear(in_features=1024, out_features=out_features)

    def forward(self, xb):
        # Feature extraction backbone.
        out = self.ext(xb)

        # Reshape.
        b, c, w, h = out.shape
        out = out.reshape(b, 1, c * w * h).transpose(0, 1)

        # Prediction head.
        # out, weights = self.att(out, out, out)
        out, _ = self.gru(out)
        logits = self.fc(out.squeeze(0))
        return logits
