import torch
import random
import numpy as np
import argparse
import os
import swanlab
from utils import NERSFT_Argument,Metric,get_Labels
from model import NER_SFT
from data_process import SFTDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
parser = argparse.ArgumentParser(description='exp arg path')
parser.add_argument('exp_arg')
args = parser.parse_args()
def random_seed(seed):
    random .seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
class Trainer():
    def __init__(self,arg,model,optimizer,scheduler,metric,tokenizer):
        self.arg=arg
        self.model=model.to(device)
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.metric=metric
        self.tokenizer=tokenizer
        self.experiment_name=f"max_len_{self.arg.max_length}_num_epochs_{self.arg.epochs_num}_bs_{self.arg.batch_size}_lr_{self.arg.lr}_{self.arg.peft_method}_lora_r_{self.arg.lora_r}"
        self.experiment_dir=os.path.join("experiment",self.experiment_name)
        os.makedirs(self.experiment_dir,exist_ok=True)
        swanlab.init(project="SFT_NER",
             experiment_name=self.experiment_name,
             arg=self.arg.arg_dict
             )
    def train(self,dataloader):
        self.model.train()
        total_loss=0.0
        steps_in_dataloader=len(dataloader)
        epoch_iterator=iter(dataloader)
        self.optimizer.zero_grad()
        num_update_steps_per_epoch=(steps_in_dataloader+self.arg.gradient_accumulation_steps-1)//self.arg.gradient_accumulation_steps
        remainder=steps_in_dataloader%self.arg.gradient_accumulation_steps
        if remainder==0:
            remainder=self.arg.gradient_accumulation_steps
        for update_step in range(num_update_steps_per_epoch):
            num_batches=(
                self.arg.gradient_accumulation_steps if update_step!=(num_update_steps_per_epoch-1) else remainder
            )
            current_gradient_accumulation_steps=num_batches
            for i in range(num_batches):
                batch=next(epoch_iterator)
                batch={k:v.to(device) for k,v in batch.items()}
                input_ids=batch["input_ids"]
                attention_mask=batch["attention_mask"]
                labels=batch["labels"]
                outputs=self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                loss=outputs.loss
                total_loss+=loss.item()
                loss=loss/current_gradient_accumulation_steps
                loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        ave_loss=total_loss / steps_in_dataloader
        return ave_loss

    def dev(self,dataloader):
        self.model.eval()
        self.metric.reset()
        with torch.no_grad():
            for batch in dataloader:
                input_ids=batch["input_ids"].to(device)
                attention_mask=batch["attention_mask"].to(device)
                true_entities_batch=batch["true_labels"]
                generated_ids=self.model(input_ids=input_ids,attention_mask=attention_mask)#原来的输入+新生成的内容
                generated_ids=[
                    outputs[len(inputs):]
                    for inputs,outputs in zip(input_ids,generated_ids)
                ]
                responses=self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
                pred_entities_batch=[
                    self.metric.parse_json(response)
                    for response in responses
                ]
                self.metric.calculate(pred_entities_batch,true_entities_batch)
            dev_precision,dev_recall,dev_f1=self.metric.compute(self.metric.tp,self.metric.pred_sum,self.metric.true_sum)
            self.metric.report()
            return dev_precision,dev_recall,dev_f1

    def save_checkpoint(self,train_loader,dev_loader,test_loader):
        dev_f1_best=0.0
        best_model_path=None
        for epoch in range(self.arg.epochs_num):
            print(f"Epoch{epoch+1}")
            train_loss=self.train(train_loader)
            print(f"训练集损失:{train_loss:.4f}")
            dev_precision,dev_recall,dev_f1=self.dev(dev_loader)
            print(f"验证集精准率:{dev_precision:.4f}")
            print(f"验证集召回率:{dev_recall:.4f}")
            print(f"验证集f1分数:{dev_f1:.4f}")
            checkpoint_name=f"check_epoch_{epoch+1}"
            checkpoint_path=os.path.join(self.experiment_dir,checkpoint_name)
            if dev_f1>=dev_f1_best:
                dev_f1_best=dev_f1
                best_model_path=checkpoint_path
                self.model.model.save_pretrained(checkpoint_path)
                torch.save({
                    "optimizer":self.optimizer.state_dict(),
                    "scheduler":self.scheduler.state_dict(),
                    "epoch":epoch + 1,
                    "dev_f1":dev_f1_best,
                    },os.path.join(checkpoint_path,"training_state.pt"))
                print(f"当前模型最优f1分数为:{dev_f1_best}")
                print(f"保存最佳模型：{best_model_path}")
            swanlab.log({
                "epoch":epoch,
                "训练集损失":train_loss,
                "验证集f1分数":dev_f1
            })
        print(f"模型最优f1分数为:{dev_f1_best}")
        if not os.path.exists(best_model_path):
            print("未找到最优模型，无法进行测试！")
            return
        self.model.load_best_model(best_model_path)
        test_precision,test_recall,test_f1=self.dev(test_loader)
        print(f"测试集精准率:{test_precision:.4f}")
        print(f"测试集召回率:{test_recall:.4f}")
        print(f"测试集f1分数:{test_f1:.4f}")
        swanlab.log({
            "测试集f1分数":test_f1
        })
def main(arg_path):
    arg=NERSFT_Argument(arg_path)
    random_seed(arg.random)
    tokenizer = AutoTokenizer.from_pretrained(
        arg.model_path,
        )
    tokenizer.pad_token=tokenizer.eos_token
    train_dataset=SFTDataset(arg=arg,dataset_type="train",tokenizer=tokenizer)
    dev_dataset=SFTDataset(arg=arg,dataset_type="dev",tokenizer=tokenizer)
    test_dataset=SFTDataset(arg=arg,dataset_type="test",tokenizer=tokenizer)

    train_dataloader=DataLoader(train_dataset,batch_size=arg.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(dev_dataset,batch_size=arg.batch_size,shuffle=False,collate_fn=dev_dataset.generate_collate_fn)
    test_dataloader=DataLoader(test_dataset,batch_size=arg.batch_size,shuffle=False,collate_fn=test_dataset.generate_collate_fn)
    model=NER_SFT(arg=arg,tokenizer=tokenizer)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],lr=arg.lr
    )
    steps_per_epoch=(len(train_dataloader)+arg.gradient_accumulation_steps-1)//arg.gradient_accumulation_steps
    total_steps = steps_per_epoch * arg.epochs_num
    num_warmup_steps=int(total_steps*0.1)
    scheduler=get_linear_schedule_with_warmup(optimizer=optimizer,
                                              num_warmup_steps=num_warmup_steps,
                                              num_training_steps=total_steps
                                              )
    labels=get_Labels(arg.labels_path)
    metric=Metric(labels=labels.labels)
    trainer=Trainer(arg=arg,model=model,optimizer=optimizer,scheduler=scheduler,metric=metric,tokenizer=tokenizer)
    trainer.save_checkpoint(train_loader=train_dataloader,dev_loader=dev_dataloader,test_loader=test_dataloader)

if __name__=="__main__":
    main(args.exp_arg)
