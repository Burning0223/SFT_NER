import json
from torch.utils.data import Dataset
from template import TEMPLATES
class SFTDataset(Dataset):
    def __init__(self,arg,dataset_type,tokenizer):
        self.arg=arg
        self.dataset_type=dataset_type
        self.template=TEMPLATES[arg.template]
        self.tokenizer=tokenizer
        self.data=self.load_data(dataset_type)
        
    def load_data(self,dataset_type):
        if dataset_type=="train":
            file_path=f"{self.arg.data_path}/train.json"
        elif dataset_type=="dev":
            file_path=f"{self.arg.data_path}/dev.json"
        elif dataset_type=="test":
            file_path=f"{self.arg.data_path}/test.json"
        else:
            raise ValueError(f"Unknown dataset_path:{dataset_type}")
        with open(file_path,"r",encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    def build_prompt(self,sentence):
        return (
            self.arg.Instruction+f"Text:{sentence}"+"Output:"
        )
    def build_response(self,entities):
        entity_list=[]
        for entity in entities:
            entity_list.append({"name":entity['name'],"type":entity['type']})
        response={"entities":entity_list}
        return json.dumps(response,ensure_ascii=False)
    def collate_fn(self,batch):
        prompts=[]
        full_texts=[]
        for sample in batch:
            sentence=sample["sentence"]
            user_content=self.build_prompt(sentence)
            assistant_content=self.build_response(sample["entities"])
            prompt,full_text=self.template.format(user_content,assistant_content)
            prompts.append(prompt)
            full_texts.append(full_text)
        self.tokenizer.padding_side = "right"
        prompts_ids=self.tokenizer(prompts,max_length=self.arg.max_length,
                                   padding=False,truncation=True)
        full_texts_ids=self.tokenizer(full_texts,return_tensors='pt',max_length=self.arg.max_length,
                                      padding=True,truncation=True)
        labels=full_texts_ids["input_ids"].clone()
        for i,prompt_ids in enumerate(prompts_ids["input_ids"]):
            prompt_len=len(prompt_ids)
            labels[i,:prompt_len]=-100
        labels[full_texts_ids["attention_mask"]==0]=-100
        return {
            "input_ids":full_texts_ids["input_ids"],
            "attention_mask":full_texts_ids["attention_mask"],
            "labels":labels
        }
    def generate_collate_fn(self,batch):
        prompts=[]
        true_entities=[]
        for sample in batch:
            sentence=sample["sentence"]
            user_content=self.build_prompt(sentence)
            prompt=self.template.format_prompt(user_content)
            prompts.append(prompt)
            true_entities.append(sample["entities"])
        self.tokenizer.padding_side = "left"
        prompts_ids=self.tokenizer(prompts,return_tensors='pt',max_length=self.arg.max_length,
                                      padding=True,truncation=True)
        return {
            "input_ids":prompts_ids["input_ids"],
            "attention_mask":prompts_ids["attention_mask"],
            "true_labels":true_entities
        }
            