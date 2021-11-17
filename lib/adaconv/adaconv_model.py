from argparse import ArgumentParser

from torch import nn

from lib.adaconv.adaconv import AdaConv2d, KernelPredictor
from lib.vgg import VGGEncoder


class AdaConvModel(nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--style-img-size', type=int, default=256)
        parser.add_argument('--style-descriptor-depth', type=int, default=512)
        parser.add_argument('--pred-kernel-size', type=int, default=3)
        return parser

    def __init__(self, style_img_size, style_descriptor_depth, pred_kernel_size):
        super().__init__()
        self.encoder = VGGEncoder()
        style_in_shape = (self.encoder.out_channels,
                          style_img_size // self.encoder.scale_factor,
                          style_img_size // self.encoder.scale_factor)
        style_out_shape = (style_descriptor_depth,
                           pred_kernel_size,
                           pred_kernel_size)
        self.style_encoder = GlobalStyleEncoder(in_shape=style_in_shape, out_shape=style_out_shape)
        self.decoder = AdaConvDecoder(style_channels=style_descriptor_depth, pred_kernel_size=pred_kernel_size)

    def forward(self, content, style, return_embeddings=False):
        self.encoder.freeze()

        # Encode -> Decode
        content_embeddings, style_embeddings = self._encode(content, style)
        output = self._decode(content_embeddings[-1], style_embeddings[-1])

        # Return embeddings if training
        if return_embeddings:
            output_embeddings = self.encoder(output)
            embeddings = {
                'content': content_embeddings,
                'style': style_embeddings,
                'output': output_embeddings
            }
            return output, embeddings
        else:
            return output

    def _encode(self, content, style):
        content_embeddings = self.encoder(content)
        style_embeddings = self.encoder(style)
        return content_embeddings, style_embeddings

    def _decode(self, content_embedding, style_embedding):
        style_embedding = self.style_encoder(style_embedding)
        output = self.decoder(content_embedding, style_embedding)
        return output


class AdaConvDecoder(nn.Module):
    def __init__(self, style_channels, pred_kernel_size):
        super().__init__()
        self.style_channels = style_channels
        self.pred_kernel_size = pred_kernel_size

        # Inverted VGG with first conv in each scale replaced with AdaConv
        group_div = [1, 2, 4, 8]
        n_convs = [1, 4, 2, 2]
        self.layers = nn.ModuleList([
            *self._make_layers(512, 256, group_div=group_div[0], n_convs=n_convs[0]),
            *self._make_layers(256, 128, group_div=group_div[1], n_convs=n_convs[1]),
            *self._make_layers(128, 64, group_div=group_div[2], n_convs=n_convs[2]),
            *self._make_layers(64, 3, group_div=group_div[3], n_convs=n_convs[3], final_act=False, upsample=False)])

    def _make_layers(self, in_channels, out_channels, group_div, n_convs, final_act=True, upsample=True):
        n_groups = in_channels // group_div

        layers = [KernelPredictor(in_channels, in_channels,
                                  n_groups=n_groups,
                                  w_channels=self.style_channels,
                                  kernel_size=self.pred_kernel_size),
                  AdaConv2d(in_channels, in_channels, n_groups=n_groups),
                  nn.ReLU(),
                  ]
        for i in range(1, n_convs - 1):
            layers += [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU()
            ]

        # Final layer:
        layers.append(nn.Conv2d(in_channels, out_channels,  kernel_size=3, padding=1, padding_mode='reflect'))
        if final_act:
            layers.append(nn.ReLU())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        return layers

    def forward(self, content, w_style):
        # Checking types is a bit hacky, but it works well.
        for module in self.layers:
            if isinstance(module, KernelPredictor):
                w_spatial, w_pointwise, bias = module(w_style)
            elif isinstance(module, AdaConv2d):
                content = module(content, w_spatial, w_pointwise, bias)
            else:
                content = module(content)

        output = content
        return output


class GlobalStyleEncoder(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        channels = in_shape[0]

        self.downscale = nn.Sequential(
            self._conv(channels, channels),
            nn.LeakyReLU(),
            self._downsample(),
            #
            self._conv(channels, channels),
            nn.LeakyReLU(),
            self._downsample(),
            #
            self._conv(channels, channels),
            nn.LeakyReLU(),
            self._downsample(),
        )

        in_features = self.in_shape[0] * (self.in_shape[1] // 8) * self.in_shape[2] // 8
        out_features = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, xs):
        ys = self.downscale(xs)
        ys = ys.reshape(len(xs), -1)

        w = self.fc(ys)
        w = w.reshape(len(xs), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return w

    def _conv(self, in_channels, out_channels, kernel_size=3, padding_mode='reflect'):
        padding = (kernel_size - 1) // 2
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=3,
                         padding=padding,
                         padding_mode=padding_mode)

    def _downsample(self, scale=2):
        return nn.AvgPool2d(scale, scale)
