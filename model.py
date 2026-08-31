import torch
from transformers import AutoModelForCausalLM,BitsAndBytesConfig
from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
class NER_SFT(torch.nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg=arg
        lora_config=LoraConfig(task_type=TaskType.CAUSAL_LM,
                                           r=arg.lora_r,
                                           lora_alpha=arg.lora_alpha,
                                           lora_dropout=arg.lora_dropout,
                                           target_modules=arg.target_modules)
        if arg.peft_method=="lora":
            self.model=AutoModelForCausalLM.from_pretrained(arg.model_path)
            self.model=get_peft_model(self.model,lora_config)
        elif arg.peft_method=="qlora":
            bnb_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            self.model=AutoModelForCausalLM.from_pretrained(
                arg.model_path,
                quantization_config=bnb_config
            )
            self.model=prepare_model_for_kbit_training(self.model)
            self.model=get_peft_model(self.model,lora_config)
        else:
            self.model=AutoModelForCausalLM.from_pretrained(arg.model_path)

    def forward(self,input_ids,attention_mask,labels=None):
        if labels is not None:
            outputs=self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        else:
            outputs=self.model.generate(input_ids=input_ids,attention_mask=attention_mask,
                                        max_new_tokens=self.arg.max_new_tokens,do_sample=False)
        return outputs
