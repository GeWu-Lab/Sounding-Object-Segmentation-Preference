from email.policy import default
import os

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2  # type: ignore

import argparse
import json
import os
from typing import Any, Dict, List

parser = argparse.ArgumentParser(
    description=(
        "TeSO, ECCV'2024."
    )
)

parser.add_argument(
    "--train_params",
    type=list,
    default=[
        'audio_proj',
        'text_proj',
        'prompt_proj',
        'avs_adapt',
        'avs_attn',
    ],
    help="Text model to extract textual reference feature.",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="The path to load the model checkpoints."
)
parser.add_argument("--save_ckpt", type=str, default='./ckpt', help='Checkpoints save dir.')
parser.add_argument("--log_path", type=str, default='./logs', help='Log info save path.')
parser.add_argument("--feat_text_dir", type=str, default='./text_feat', help='Extracted imagebind feature.')
parser.add_argument("--text_json_path", type=str, default='./v1m/text_json', help='reaoned potential sounding objects.')

 
parser.add_argument(
    "--data_dir",
    type=str,
    default='/home/data/AVS/' 
)

parser.add_argument("--show_params", action='store_true', help=f"Show params names with Requires_grad==True.")
parser.add_argument("--m2f_model", type=str, default='facebook/mask2former-swin-base-ade-semantic', help="Pretrained mask2former.")

parser.add_argument("--ver", type=str, default='v1m', help='v1m, v1s, v2')
parser.add_argument("--lr", type=float, default=1e-4, help='lr to fine tuning adapters.')
parser.add_argument("--epochs", type=int, default=50, help='epochs to fine tuning adapters.')
parser.add_argument("--loss", type=str, default='bce', help='')

parser.add_argument("--train", default=False, action='store_true', help='start train?')
parser.add_argument("--val", type=str, default=None, help='type: str; val | test')  # NOTE: for test and val.
parser.add_argument("--test", default=False, action='store_true', help='start test?')


parser.add_argument("--gpu_id", type=str, default="0", help="The GPU device to run generation on.")

parser.add_argument("--run", type=str, default='train', help="train, test")

parser.add_argument("--frame_n", type=int, default=10, help="Frame num of each video. Fixed to 10.")
parser.add_argument("--num_a", type=int, default=4, help="Number of audio compoents.")



args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print(f'>>> Sys: set "CUDA_VISIBLE_DEVICES" - GPU: {args.gpu_id}')
