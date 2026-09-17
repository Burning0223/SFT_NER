# 大模型的指令微调的实体识别（NER）任务
## 项目简介
本项目基于 PyTorch 和 Hugging Face Transformers 框架，实现基于大语言模型指令微调的中文命名实体识别（NER）任务。

项目以 Qwen2.5-7B-Instruct 为基础模型，通过构造指令（Instruction）、输入文本（Input）和实体标注结果（Output），对模型进行监督微调，使模型能够根据给定的医学文本识别其中的实体及其类型。

本项目主要用于学习和实现大模型指令微调的完整流程，包括：

（1）Prompt 构造与 Tokenizer 处理

（2）训练集、验证集和测试集处理

（3）LoRA、Qlora

（4）梯度检查点（Gradient Checkpointing）

（5）梯度累计

（6）模型训练与验证

（7）最优模型保存与加载

（8）Precision、Recall和F1-Score评估

## 项目结构
``` 
├── Argument         # 参数配置文件
│   ├── arg_1.json
│   └── arg_2.json
├── utils.py          # 功能类
├── model.py           # 模型定义
├── data_process.py    # 数据预处理
├── train.py     # 训练器以及程序入口
├── template.py     # Prompt模板
├── requirements.txt   # 环境依赖
└── README.md          # 项目说明
```
## 数据格式
```
[{"sentence": "Comparison with alkaline phosphatases and 5 - nucleotidase", "entities": [{"name": "alkaline phosphatases", "type": "GENE", "pos": [16, 37]}]}]
```

## 环境依赖
``` 
pip install -r requirements.txt
``` 
## 模型参数设置
``` 
   {
    "model_path":"/home/model/Qwen2.5-7B",
    "max_length":1024,
    "epochs_num":4,
    "batch_size":4,
    "lr":2e-4,
    "data_path":"./bc2gm1",
    "labels_path":"./bc2gm1/labels.json",
    "random":42,
    "Instruction":"You are a biomedical named entity recognition model. Given a biomedical text, identify all biomedical entities mentioned in the text. For each entity, provide its exact name and corresponding entity type. Carefully read the entire text and identify all valid biomedical entities. Do not omit valid entities or extract non-biomedical words. Preserve each entity name exactly as it appears in the text. Return the results strictly in JSON format. Each entity should contain \"name\" and \"type\". If no biomedical entities are found, return an empty list.Output format:{\"entities\":[{\"name\":\"entity name\",\"type\":\"entity type\"}]}Do not include any explanation, comments, or text outside the JSON output.",
    "lora_r":8,
    "lora_alpha":16,
    "lora_dropout":0.05,
    "target_modules":[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
    ],
    "peft_method":"qlora",
    "max_new_tokens":256,
    "gradient_accumulation_steps":16,
    "template":"qwen"

}
```
## 输出示例
```
[{"entities": [{"name": "hFIRE", "type": "GENE"}, {"name": "Sp1", "type": "GENE"}, {"name": "Sp3", "type": "GENE"}, {"name": "CBF", "type": "GENE"}]}]
```
## 实验结果
### QLora
运行命令：
``` 
python train.py Argument/arg_1.json
```

``` 
Entity    Precision      Recall         F1-Score       Support
GENE      0.84           0.83           0.84           5947
micro avg 0.84           0.83           0.84           5947
测试集精准率:0.8425
测试集召回率:0.8278
测试集f1分数:0.8351
```
<img width="2210" height="841" alt="image" src="https://github.com/user-attachments/assets/13aad9d9-8afd-4b15-885f-5b9993af4e16" />
显存峰值为13342MB≈13GB

  #### Lora
运行命令：
``` 
python train.py Argument/arg_2.json
```

``` 
Entity    Precision      Recall         F1-Score       Support
GENE      0.84           0.83           0.83           5947
micro avg 0.84           0.83           0.83           5947
测试集精准率:0.8425
测试集召回率:0.8255
测试集f1分数:0.8339
``` 
<img width="760" height="328" alt="image" src="https://github.com/user-attachments/assets/0ca34927-0691-4382-8b9e-956ea004a185" />

显存峰值为26916MB≈26.29GB

