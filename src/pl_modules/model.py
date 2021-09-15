from typing import Tuple, List

import torch
from torch import nn
from src.common.utils import (
    cnn_size,
    pad_size,
    fc_params,
    cnn_params,
    cnnxfc_params,
    pool_size,
    cnn_kernel,
    multiple_cnn_size,
)

# N.B. It is recommended to choose odd kernels or things could break up.


class CNNAtt(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        channels: List,
        kernels: List,
        paddings: List,
        strides: List,
        num_heads: int,
    ):
        """
        A layer with some CNNs and attention.
        :param image_shape: The shape of the input image (channels, width, height).
        :param channels: The number of output channels for the CNNs, the last one has to be 1.
        :param kernels: A tuple with kernels for first, second and third convolution.
        :param paddings: A tuple with paddings for first, second and third convolution.
        :param strides: A tuple with strides for first, second and third convolution.
        :param num_heads: Number of heads to use in the attention layer, by default one to avoid prime numbers' issues.
        """
        super(CNNAtt, self).__init__()
        c, w, h = image_shape

        # Are used three CNNs since is the minimum needed to have a bottleneck and return to the original channel size,
        # yet is still possible to learn the identity function as a composition of f and its inverse.

        # Query
        self.cnn_q = self.get_seq(
            name="CNNQuery",
            channels=channels,
            kernels=kernels,
            paddings=paddings,
            strides=strides,
        )

        # Key
        self.cnn_k = self.get_seq(
            name="CNNKey",
            channels=channels,
            kernels=kernels,
            paddings=paddings,
            strides=strides,
        )

        # Value
        self.cnn_v = self.get_seq(
            name="CNNValue",
            channels=channels,
            kernels=kernels,
            paddings=paddings,
            strides=strides,
        )

        # Measure output sizes.
        seq, embed = self.count_outputs(
            input=(w, h), kernels=kernels, paddings=paddings, strides=strides
        )

        # Attention layers, one for each image dimension.
        self.att1 = nn.MultiheadAttention(
            embed_dim=embed, num_heads=num_heads, batch_first=True
        )
        self.att2 = nn.MultiheadAttention(
            embed_dim=seq, num_heads=num_heads, batch_first=True
        )

    def forward(self, xb):
        q = self.cnn_q(xb).squeeze(dim=1)
        k = self.cnn_k(xb).squeeze(dim=1)
        v = self.cnn_v(xb).squeeze(dim=1)

        # Transpose since batch first is usually faster, in att2 to swap row with columns.
        att1, _ = self.att1(q, k, v)
        att2, _ = self.att2(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))

        # Add one dimension and concatenate as a 2-channels image.
        att1 = att1.unsqueeze(1)
        att2 = att2.transpose(1, 2).unsqueeze(1)
        out = torch.cat((att1, att2), dim=1)

        return out

    def get_seq(
        self,
        name: str,
        channels: List,
        kernels: List,
        paddings: List,
        strides: List,
    ):
        """
        This function returns a list of modules that build up a basic block. Each one will be used as query, key or value
        in an attention layer.
        :param name: Name of the layer.
        :param channels: Channels of each CNN layer, the last one has to be 1.
        :param kernels: Kernel size for each CNN.
        :param paddings: Paddings for each CNN.
        :param strides: Strides for each CNN.
        :return: A nn.Sequential object.
        """
        assert channels[-1] == 1, "The attention layer can accept one channel only!."

        # By default audio data has only one channel.
        in_channels = 1
        seq = nn.Sequential()

        for n, (out_channels, kernel, padding, stride) in enumerate(
            zip(channels, kernels, paddings, strides)
        ):
            cnn = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                padding=padding,
                stride=stride,
            )
            seq.add_module(name=f"{name}{n}", module=cnn)
            in_channels = out_channels

        return seq

    def count_outputs(
        self,
        input: Tuple[int, int],
        kernels: List,
        paddings: List,
        strides: List,
    ):
        """
        Just count the output of an image obtained by a sequence of convolutions.
        :param input: Input image dimension.
        :param kernels: List of kernels to apply.
        :param paddings: List of paddings.
        :param strides: List of strides.
        :return: The size of the feature image.
        """
        image = input
        for kernel, padding, stride in zip(kernels, paddings, strides):
            image = cnn_size(input=image, kernel=kernel, padding=padding, stride=stride)
        return image


class CNNRes(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int):
        super(CNNRes, self).__init__()

        # This padding is added to keep dimensionality the same, it is recommended to choose even kernels.
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=pad,
        )

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=pad,
        )

        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=pad,
        )

        self.conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=pad,
        )

        self.conv5 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=pad,
        )

        self.conv6 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=pad,
        )

    def forward(self, res):
        out = self.conv1(res)
        res = self.conv1(out) + res
        out = self.conv1(res)
        res = self.conv1(out) + res
        out = self.conv1(res)
        res = self.conv1(out)
        return res


class Extraction(nn.Module):
    """
    Feature extraction backbone.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        att_channels: List,
        att_kernels: List,
        att_paddings: List,
        att_strides: List,
        att_num_heads: int,
        pool_att: int,
        res_kernels: List,
        pool: int,
    ):
        super(Extraction, self).__init__()
        self.level = len(res_kernels)

        # CNNAtt blocks.
        self.cnnatt1 = CNNAtt(
            image_shape=image_shape,
            channels=att_channels,
            kernels=att_kernels,
            paddings=att_paddings,
            strides=att_strides,
            num_heads=att_num_heads,
        )

        self.cnnatt2 = CNNAtt(
            image_shape=image_shape,
            channels=att_channels,
            kernels=att_kernels,
            paddings=att_paddings,
            strides=att_strides,
            num_heads=att_num_heads,
        )

        # CNN for the concatenated attentions, it returns us an image of the same shape as the input and
        # can be used for residuals. The number of input channels is fixed to 4 due to it being 1*2*2:
        # - 1 the number of out channels in CNNAtt.
        # - 2 due to the concatenation of attentions in CNNAtt.
        # - 2 due to the concatenation of attentions self.forward().
        self.cnn_att = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1, 1))

        # Pooling before blocks, can be useful to lighten computation.
        self.pool_att = nn.AvgPool2d(kernel_size=pool_att)
        # count_pool_0 = pool_size(input=image_shape[1:], pooling=pool_0)

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
        # Get attentions.
        att1 = self.cnnatt1(xb)
        att2 = self.cnnatt1(xb)
        out = torch.cat((att1, att2), dim=1)

        # Conv to use residuals on.
        out = self.cnn_att(out) + xb
        out = self.pool_att(out)

        # Residuals.
        out = self.res(out)

        # Pooling.
        out = self.pool(out)

        return out


class SoundscapeModel(nn.Module):
    # Shape of the input image (c,h,w)
    shape = torch.Size((1, 128, 313))

    def __init__(self, **kwargs):
        super().__init__()

        # Feature extraction backbone.
        self.ext = Extraction(
            image_shape=(1, 128, 313),
            att_channels=[1, 3, 1],
            att_kernels=[1, 3, 1],
            att_paddings=[0, 1, 0],
            att_strides=[1, 1, 1],
            att_num_heads=1,
            pool_att=1,
            res_kernels=[3, 3],
            pool=3,
        )

        # Prediction head.
        self.att = nn.MultiheadAttention(embed_dim=1000, num_heads=5, dropout=0.2)

        self.lstm = nn.LSTM(
            input_size=1000,
            hidden_size=250,
            num_layers=1,
            bidirectional=True,
            dropout=0,  # Dropout should be 0 when there is only one layer.
        )

        self.fc = nn.Linear(in_features=500, out_features=1)

    def forward(self, xb):
        # Feature extraction backbone.
        out = self.ext(xb)

        # Reshape.
        b, c, w, h = out.shape
        out = out.reshape(b, 1, c * w * h).transpose(0, 1)

        # Prediction head.
        out, weights = self.att(out, out, out)
        out, (h_n, c_n) = self.lstm(out)
        out = self.fc(out.squeeze())

        return out


class BirdcallModel(nn.Module):
    def __init__(self, out_features: int, **kwargs):
        super().__init__()

        self.ext = Extraction(
            image_shape=(1, 128, 313),  # 3751 for Birdcalls and 313 for Joint.
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
            input_size=384,  # 4960 or 384
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
        logits = self.fc(out.squeeze())
        return logits


class JointModel(nn.Module):
    def __init__(self, out_features: int, **kwargs):
        super().__init__()

        self.ext = Extraction(
            image_shape=(1, 128, 313),
            att_channels=[1, 3, 1],
            att_kernels=[1, 3, 1],
            att_paddings=[0, 1, 0],
            att_strides=[1, 1, 1],
            att_num_heads=1,
            pool_att=1,
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
        logits = self.fc(out.squeeze())
        return logits
