# -*- coding: utf-8 -*-

# ***************************************************
# * File        : fllash_attn_install.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-15
# * Version     : 1.0.071514
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
# from utils.log_util import logger




# 测试代码 main 函数
def main():
    print(f"Device Capability: {torch.cuda.get_device_capability()[0]}")
    assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'
    # os.system("uv pip install ninja packaging")
    # os.system("uv pip install flash-attn --no-build-isolation")

if __name__ == "__main__":
    main()
