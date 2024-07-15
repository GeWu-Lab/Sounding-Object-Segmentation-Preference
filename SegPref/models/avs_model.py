from models.local.mask2former import Mask2FormerImageProcessorForRefAVS
from models.local.mask2former import Mask2FormerForRefAVS
from models.local.mask2former import logging
from models.local.mask2former import refavs_transformer

from PIL import Image
import requests
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module
import re
import matplotlib.pyplot as plt

logging.set_verbosity_error()


image_processor = Mask2FormerImageProcessorForRefAVS.from_pretrained("facebook/mask2former-swin-base-ade-semantic")
model_m2f = Mask2FormerForRefAVS.from_pretrained(
    "facebook/mask2former-swin-base-ade-semantic"
)

# avs_dataset = AVS()

class AVS_Model_Base(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.model_v = model_m2f.cuda()

        self.dim_v = 1024
        self.num_heads = 8
        
        self.audio_proj = nn.Sequential(
            nn.Linear(128, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.dim_v),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.dim_v),
        )

        self.prompt_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
        )

        self.cfgs = cfgs
        
        self.loss_fn = F.binary_cross_entropy_with_logits  # 'bce'

        self.mha_A_T = nn.MultiheadAttention(self.dim_v, self.num_heads)
        self.mha_V_T = nn.MultiheadAttention(self.dim_v, self.num_heads)
        self.mha_mm = nn.MultiheadAttention(self.dim_v, self.num_heads)

        self.cache_mem_beta = 1
        self.ca = refavs_transformer.REF_AVS_Transformer(embed_dim=1024)

    def forward(self, batch_data):
        uid, mask_recs, img_recs, image_sizes, feat_aud, feat_text = batch_data
        bsz = len(uid)
        frame_n = len(img_recs[0])
        loss_uid = []
        uid_preds = []
        assert len(uid) == len(img_recs)

        mask_recs = [torch.stack(rec) for rec in mask_recs]
        gt_label = torch.stack(mask_recs).view(bsz*frame_n, mask_recs[0].shape[-2], mask_recs[0].shape[-1]).squeeze().cuda()

        og_aud = feat_aud = torch.stack(feat_aud).cuda()
        feat_aud = feat_aud.unsqueeze(2).repeat(1, 1, self.cfgs.num_a, 1)
        feat_text = [torch.stack(feat_t).cuda() for feat_t in feat_text]
        feat_text = torch.stack(feat_text).cuda()

        feat_aud = self.audio_proj(feat_aud).view(bsz*feat_aud.shape[-3], feat_aud.shape[-2], self.dim_v)
        feat_text = self.text_proj(feat_text).view(bsz*feat_text.shape[-3], feat_text.shape[-2], self.dim_v).permute(1, 0, 2)

        batch_pixel_values, batch_pixel_mask = [], []

        for idx, _ in enumerate(uid):
            img_input = img_recs[idx]
            
            for img in img_input:
                batch_pixel_values.append(img['pixel_values'])
                batch_pixel_mask.append(img['pixel_mask'])

        batch_pixel_values = torch.stack(batch_pixel_values).squeeze().cuda()
        batch_pixel_mask = torch.stack(batch_pixel_mask).squeeze().cuda()

        fused_feat = self.ca(target=feat_text, source=feat_aud).permute(1, 0, 2)
        fused_feat = self.prompt_proj(fused_feat)

        batch_input = {
            'pixel_values': batch_pixel_values,
            'pixel_mask': batch_pixel_mask,
            'mask_labels': gt_label,
            'prompt_features_projected': fused_feat, 
        }

        outputs = self.model_v(**batch_input)
 
        pred_instance_map = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[[256, 256]]*(bsz*frame_n),
        )

        pred_instance_map = torch.stack(pred_instance_map, dim=0).view(bsz*frame_n, 256, 256)

        loss_frame = self.loss_fn(input=pred_instance_map.squeeze(), target=gt_label.squeeze().cuda())
        loss_uid.append(loss_frame)
        uid_preds.append(pred_instance_map.squeeze())

        return loss_uid, uid_preds
