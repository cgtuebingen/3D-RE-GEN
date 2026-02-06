import torch.nn as nn
from vst_main.Models.t2t_vit import T2t_vit_t_14
from vst_main.Models.Transformer import Transformer
from vst_main.Models.Transformer import token_Transformer
from vst_main.Models.Decoder import Decoder


class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()

        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=False, args=args)

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)

    def forward(self, image_Input):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)

        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]

        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4)

        return outputs
