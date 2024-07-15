import sys
import os

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pdb


import random

from torchvision import transforms
from collections import defaultdict
import cv2
from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
from PIL import Image

from towhee import pipe, ops
from transformers import pipeline


# logger = log_agent('audio_recs.log')

import pickle as pkl

class AVS(Dataset):
    def __init__(self, split='train', cfg=None):
        # metadata: train/test/val
        self.data_dir = f'{cfg.data_dir}/{cfg.ver}'
        meta_path = f'{cfg.data_dir}/metadata.csv'

        metadata = pd.read_csv(meta_path, header=0)
        self.split = split
        self.metadata = metadata[metadata['split'] == split]  # split= train,test,val.
        self.metadata = self.metadata[self.metadata['label'] == cfg.ver]

        self.media_path = f'{self.data_dir}'
        if cfg.ver == 'v1s' and split == 'train':
            self.frame_num = 1
        elif cfg.ver == 'v2':
            self.frame_num = 10
        else:
            self.frame_num = 5
 
        # modalities processor/pipelines
        self.img_process = AutoImageProcessor.from_pretrained(cfg.m2f_model)

        self.audio_vggish_pipeline = (   # pipeline building
            pipe.input('path')
                .map('path', 'frame', ops.audio_decode.ffmpeg())
                .map('frame', 'vecs', ops.audio_embedding.vggish())
                .output('vecs')
        )

        with open(cfg.text_json_path, 'r') as f:
            self.text_json = json.load(f)  # format: {"vid": {"f0": "guitar,", "f1": "guitar,man,", ...}...}

        self.feat_text_dir = cfg.feat_text_dir  # [n_possible_class, 1024] for each frame

    def get_audio_emb(self, wav_path):
        """ wav string path. """ 
        emb = torch.tensor(self.audio_vggish_pipeline(wav_path).get()[0])
        return emb
    
    def get_text_feat(self, exp, vid, idx):
        """
        # You can try with below code to extract imagebind features.
        exp = f'{exp}None'.split(',')
        exp_list = ['None', 'None', 'None', 'None']
        len_ = min(len(exp_list), len(exp))
        for i in range(len_):
            exp_list[i] = exp[i]

        inputs = {
                ImageBind.imagebind.imagebind_model.ModalityType.TEXT: data.load_and_transform_text(exp_list, self.text_encoder.device),
        }
        # shape: [n, 1024], n=4 for v1m and v2
        # for details: https://github.com/facebookresearch/ImageBind
        """
        _path = f'{self.feat_text_dir}/{vid}/{idx}.np'  # [4, 1024] for each frame, 
        if os.path.exists(_path):
            feat_text = torch.tensor(np.load(_path)).cuda()
        else:
            feat_text = torch.randn([4, 1024]).float().cuda()  # just for example running
        return feat_text

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid = df_one_video['uid']  # uid for vid.

        img_recs = []
        mask_recs = []
        images = []

        rec_audio = f'{self.media_path}/{vid}/audio.wav'
        rec_texts = self.text_json[df_one_video['vid']]

        feat_aud = self.get_audio_emb(rec_audio)
        feat_texts = []  # image bind

        for _idx in range(self.frame_num):  # set frame_num as the batch_size
            # frame 
            path_frame = f'{self.media_path}/{vid}/frames/{_idx}.jpg'  # image
            image = Image.open(path_frame)
            image_sizes = [image.size[::-1]]
            image_inputs = self.img_process(image, return_tensors="pt")  # single frame rec
            
            # mask label
            path_mask = f'{self.media_path}/{vid}/labels_rgb/{_idx}.png' 
            mask_cv2 = cv2.imread(path_mask)
            mask_cv2 = cv2.resize(mask_cv2, (256, 256))
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
            gt_binary_mask = torch.as_tensor(mask_cv2 > 0, dtype=torch.float32)
            
            # feat text at frame x
            # print(rec_texts.keys())  # dict_keys(['f0', 'f1', 'f2', 'f3', 'f4'])
            feat_text = self.get_text_feat(rec_texts[f'f{_idx}'], vid, _idx)

            # video frames collect
            img_recs.append(image_inputs)
            mask_recs.append(gt_binary_mask)
            feat_texts.append(feat_text)
        # print(len(feat_texts))
        return vid, mask_recs, img_recs, image_sizes, feat_aud, feat_texts