from transformers import AutoModelForCausalLM
from peft import LoraModel, LoraConfig
import pdb
import torch
import torch.nn as nn
import sys
sys.path.append("/home/wanghanqing/projects/peft-group_lora/src/peft/tuners/group_lora")

from model import group_LoraModel

model = AutoModelForCausalLM.from_pretrained("/home/wanghanqing/projects/models/Llama-2-7b-chat-hf")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

lora_model_1 = group_LoraModel(model, lora_config, "lora1")
pdb.set_trace()
lora_model_2 = group_LoraModel(model, lora_config, "lora2")