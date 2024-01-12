from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
###peft version=0.7.1 更低的版本没有测试过






# def load_fusion_gate(model,gate_path):
#     gate_weight_dict = torch.load(gate_path)
#     target_dict = model.state_dict()
#     for key in gate_weight_dict.keys():
#         #把
#         target_key = key.replace("model.","")

#### 载入pretrained_model
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "/home/wanghanqing/projects/models/Llama-2-7b-hf"
        )
tokenizer = AutoTokenizer.from_pretrained("/home/wanghanqing/projects/models/Llama-2-7b-hf")
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
lora_model_name_or_path = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/zh/lora"

#### 初始化PeftModel, 并且load第一个adapter
lora_model = PeftModel.from_pretrained(pretrained_model, model_id = lora_model_name_or_path, adapter_name = "zh")
lora_model = lora_model.to("cuda")

# (Pdb) type(lora_model)
# <class 'peft.peft_model.PeftModelForCausalLM'>
# (Pdb) type(lora_model.base_model)
# <class 'peft.tuners.lora.model.LoraModel'>
# (Pdb) type(lora_model.base_model.model)
# <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>
# (Pdb) type(lora_model.base_model.model.model)
# <class 'transformers.models.llama.modeling_llama.LlamaModel'>

#### 读取另外两个adapter
lora_model.load_adapter(model_id = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/math_epoch3/lora",adapter_name = "math")


# import pdb
# pdb.set_trace()
# lora_model.load_adapter(model_id = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/en/lora", adapter_name = "en")

# #### 对list里的adapter做weight_sum
# lora_model.base_model.add_weighted_adapter(adapters = ['en','zh','math'],weights = [0.3,0.3,0.4],adapter_name = "en-zh-math",combination_type='linear')

prompts = ["[INST] Hello, who are you? [/INST]"]
lora_model.base_model.set_adapter(["zh","math"])
# import pdb
# pdb.set_trace()
lora_model.base_model.model.model.to("cuda")
inputs = tokenizer(
                    prompts,
                    max_length=256,
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
outputs = lora_model.generate(
                input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"], 
                max_new_tokens=200,
                do_sample=True,
                temperature=0.1,
                top_p=0.95, 
                )
                
outputs = tokenizer.batch_decode(
    outputs.to("cpu"), skip_special_tokens=True
)

print(f"outputs {outputs}")
#### 激活新建的adapter，使得forward时只加上它的lora结果
#### 激活多个adapter会使得forward时加入多个lora的lora results
lora_model.base_model.set_adapter(["en-zh-math"])

# (Pdb) lora_model.base_model.active_adapters
# ['en-zh-math']
# (Pdb) lora_model.active_adapters
# ['en-zh-math']


#### 可以验证active_adapter在每一个module(LoraLayer类)都生效了
lora_model.base_model.model.model.layers[30].self_attn.q_proj.active_adapters  #:['en-zh-math']

#### 把lora的参数merge到模型本体上
#### 不指定adapter_names的话默认只merge active adapters, 这里指不指定都一样
merged_model = lora_model.base_model.merge_and_unload(adapter_names = ["en-zh-math"])

# (Pdb) type(merged_model)
# <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>

output_dir = "xxxxx"
###如果没有改变词表就不用管

###*****open-instruct的代码是会修改词表的*******

# tokenizer = AutoTokenizer.from_pretrained(args.lora_model_tokenizer)
# embedding_size = merged_model.get_input_embeddings().weight.shape[0]
# if len(tokenizer) > embedding_size:
#     print(f"The vocabulary size of the tokenizer in the lora model folder contains {len(tokenizer)-embedding_size} more tokens than the base model.")
#     print("Resizing the token embeddings of the merged model...")
#     merged_model.resize_token_embeddings(len(tokenizer))
# print(f"Saving to {output_dir}...")


#### 可以调用.save_pretrained()保存模型
merged_model.save_pretrained(output_dir)




