import torch
import torch.nn as nn

from .hybrid_encoder import HybridEncoder
from .presnet import PResNet
from .rtdetr import RTDETR
from .rtdetrv2_decoder import RTDETRTransformerv2
from .rtdetr_postprocessor import RTDETRPostProcessor

from pathlib import Path
RESUME_PATH = Path(__file__).parent / 'rtdetrv2_r18vd_120e_coco_rerun_48.1.pth'

class Model(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

        # Initialize model parameters
        backbone = PResNet(
            depth       = 18,
            variant     = 'd',
            freeze_at   = -1,
            return_idx  = [1, 2, 3],
            num_stages  = 4,
            freeze_norm = False,
            pretrained  = True,
            act         = 'relu')
        encoder = HybridEncoder(
            in_channels        = [128, 256, 512],
            feat_strides       = [8, 16, 32],
            hidden_dim         = 256,
            use_encoder_idx    = [2],
            num_encoder_layers = 1,
            nhead              = 8,
            dim_feedforward    = 1024,
            dropout            = 0.0,
            enc_act            = 'gelu',
            expansion          = 0.5,
            depth_mult         = 1,
            act                = 'silu',
            pe_temperature     = 10000,
            eval_spatial_size  = [640, 640],
            version            = 'v2')
        decoder = RTDETRTransformerv2(
            feat_channels       = [256, 256, 256],
            feat_strides        = [8, 16, 32],
            hidden_dim          = 256,
            num_levels          = 3,
            num_layers          = 3,
            num_queries         = 300,
            num_denoising       = 100,
            label_noise_ratio   = 0.5,
            box_noise_scale     = 1.0,
            eval_idx            = -1,
            num_points          = [4, 4, 4],
            cross_attn_method   = 'default',
            query_select_method ='default',
            num_classes         = 80,
            nhead               = 8,
            dim_feedforward     = 1024,
            dropout             = 0.0,
            activation          = 'relu',
            learn_query_content = False,
            eval_spatial_size   = [640, 640],
            eps                 = 0.01,
            aux_loss            = True)
        postprocessor = RTDETRPostProcessor(
            num_top_queries       = 300,
            num_classes           = 80,
            use_focal_loss        = True,
            remap_mscoco_category = True)

        rtdetr = RTDETR(backbone, encoder, decoder)

        # Load trained model
        print('Load RTDETR state_dict')
        checkpoint = torch.load(RESUME_PATH, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        rtdetr.load_state_dict(state)

        self.rtdetr = rtdetr.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.rtdetr(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs