import os
import json
class NERSFT_Argument:
    def __init__(self,arg_path):
        self.arg_dict=self.load_arg(arg_path)
        for k,v in self.arg_dict.items():
            setattr(self,k,v)

    def load_arg(self,arg_path):
        if not os.path.exists(arg_path):
            raise FileExistsError(f"配置文件{arg_path}不存在")
        else:
            with open(arg_path,"r",encoding="utf-8") as f:
                return json.load(f)
class get_Labels:
    def __init__(self,labels_path):
        self.labels=self.load_labels(labels_path)
    def load_labels(self,labels_path):
        if not os.path.exists(labels_path):
            raise FileExistsError(f"labels文件{labels_path}不存在")
        else:
            with open(labels_path,"r",encoding="utf-8") as f:
                return json.load(f)
        
class Metric:
    def __init__(self,labels):
        self.reset()
        self.types=labels

    def reset(self):
        self.tp=0
        self.pred_sum=0
        self.true_sum=0

        self.entities_tp={}
        self.entities_pred_num={}
        self.entities_true_num={}
    def parse_json(self,response):
        try:
            entities=json.loads(response)
            if not isinstance(entities,list):
                return []
            return entities
        except (json.JSONDecodeError,TypeError):
            return []
    def entity_to_tuple(self,entity):
        try:
            return (
                entity["name"],
                entity["type"],
                entity["pos"][0],
                entity["pos"][1]
            )
        except (KeyError,TypeError,IndexError):
            return None
    def calculate(self,pred_entities_batch,true_entities_batch):
        for pred_entities,true_entities in zip(pred_entities_batch,true_entities_batch):
            pred_entities_set=set()
            true_entities_set=set()
            for pred_entity in pred_entities:
                entity_tuple=self.entity_to_tuple(pred_entity)
                if entity_tuple is not None:
                    pred_entities_set.add(entity_tuple)
            for true_entity in true_entities:
                entity_tuple2=self.entity_to_tuple(true_entity)
                if entity_tuple2 is not None:
                    true_entities_set.add(entity_tuple2)

            self.tp+=len(pred_entities_set&true_entities_set)
            self.pred_sum+=len(pred_entities_set)
            self.true_sum+=len(true_entities_set)

            for type in self.types:
                pred_type_set=set()
                true_type_set=set()
                for entity in pred_entities_set:
                    if entity[1]==type:
                        pred_type_set.add(entity)
                for entity in true_entities_set:
                    if entity[1]==type:
                        true_type_set.add(entity)
                self.entities_tp[type]=self.entities_tp.get(type,0)+len(pred_type_set&true_type_set)
                self.entities_pred_num[type]=self.entities_pred_num.get(type,0)+len(pred_type_set)
                self.entities_true_num[type]=self.entities_true_num.get(type,0)+len(true_type_set)

    def compute(self,tp,pred,true):
        eps=1e-8
        precision=tp/(pred+eps)
        recall=tp/(true+eps)
        f1=2*precision*recall/(precision+recall+eps)

        return precision,recall,f1

    def report(self):
        print(f"{'Entity':<10}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'Support':<10}")
        for type in self.types:
            precision,recall,f1=self.compute(self.entities_tp.get(type,0),self.entities_pred_num.get(type,0),self.entities_true_num.get(type,0))
            print(f"{type:<10}{precision:<15.2f}{recall:<15.2f}{f1:<15.2f}{self.entities_true_num.get(type,0):<10}")
        micro_avg_p,micro_avg_r,micro_avg_f=self.compute(self.tp,self.pred_sum,self.true_sum)
        print(f"{'micro avg':<10}{micro_avg_p:<15.2f}{micro_avg_r:<15.2f}{micro_avg_f:<15.2f}{self.true_sum:<10}")



      