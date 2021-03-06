from torch import nn
from typing import List, Tuple
from src.pl_modules.model import CNNRes


class Extraction(nn.Module):
    """
    Feature extraction backbone.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        res_kernels: List,
        pool: int,
    ):
        super(Extraction, self).__init__()
        self.level = len(res_kernels)

        # Residual blocks and convolutions in between them used to change dimensionality/filter sizes.
        # We can fix out_channels since all audio data has originally one channel.
        self.res = nn.Sequential()
        in_channels = image_shape[0]
        for n, kernel in enumerate(res_kernels):
            res = CNNRes(in_channels=in_channels, kernel_size=kernel)
            cnn = nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=kernel,
                stride=(2, 2),
            )

            self.res.add_module(name=f"CNNRes{n+1}", module=res)
            self.res.add_module(name=f"CNN{n+1}", module=cnn)

            in_channels *= 2

        # Output pooling.
        self.pool = nn.AvgPool2d(kernel_size=pool)

    def forward(self, xb):
        # Residuals.
        out = self.res(xb)

        # Pooling.
        out = self.pool(out)

        return out


class JointModel(nn.Module):
    def __init__(self, out_features: int, **kwargs):
        super().__init__()

        self.ext = Extraction(
            image_shape=(1, 128, 313),
            res_kernels=[3, 5],
            pool=1,
        )

        self.gru = nn.GRU(
            input_size=9120,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            dropout=0,
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
