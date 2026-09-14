import torch
from transformers import AutoModelForCausalLM,BitsAndBytesConfig
from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training,PeftModel
class NER_SFT(torch.nn.Module):
    def __init__(self,arg,tokenizer):
        super().__init__()
        self.arg=arg
        self.tokenizer=tokenizer
        lora_config=LoraConfig(task_type=TaskType.CAUSAL_LM,
                                           r=arg.lora_r,
                                           lora_alpha=arg.lora_alpha,
                                           lora_dropout=arg.lora_dropout,
                                           target_modules=arg.target_modules)
        self.model=self.load_base_model()
        self.model=get_peft_model(self.model,lora_config)
        self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant":False}
                )
        self.model.enable_input_require_grads()
        self.model.config.use_cache=False
    def load_base_model(self):
        if self.arg.peft_method=="lora":
            base_model=AutoModelForCausalLM.from_pretrained(self.arg.model_path,torch_dtype=torch.bfloat16)
        elif self.arg.peft_method=="qlora":
            bnb_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            base_model=AutoModelForCausalLM.from_pretrained(
                self.arg.model_path,
                quantization_config=bnb_config
            )
            base_model=prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True)
        else:
            raise ValueError(
                f"不支持的peft_method: {self.arg.peft_method}"
            )
        return base_model
    def load_best_model(self,best_model_path):
        base_model=self.load_base_model()
        self.model=PeftModel.from_pretrained(base_model, best_model_path, is_trainable=False)
    
    def forward(self,input_ids,attention_mask,labels=None):
        if labels is not None:
            outputs=self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        else:
            outputs=self.model.generate(input_ids=input_ids,attention_mask=attention_mask,
                                        max_new_tokens=self.arg.max_new_tokens,do_sample=False,
                                        pad_token_id=self.tokenizer.pad_token_id)
        return outputs