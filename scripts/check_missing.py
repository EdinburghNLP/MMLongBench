import os
import ast
import json
import argparse
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm

dataset_to_metrics = {
    "infoseek": "sub_em",
    "viquae": "sub_em",
    
    "vh_single": "acc",
    "vh_multi": "acc",
    "mm_niah_retrieval-text": "sub_em",
    "mm_niah_counting-text": "soft_acc",
    "mm_niah_reasoning-text": "sub_em",
    "mm_niah_retrieval-image": "mc_acc",
    "mm_niah_counting-image": "soft_acc",
    "mm_niah_reasoning-image": "mc_acc",
    
    "cars196": "cls_acc",
    "food101": "cls_acc",
    "inat2021": "cls_acc",
    "sun397": "cls_acc",
    
    "gov-report": "rougeLsum_f1",# "gpt4-flu-f1", "rougeLsum_f1",
    "multi-lexsum": "rougeLsum_f1",# "gpt4-flu-f1", "rougeLsum_f1",
    
    "longdocurl": "doc_qa",
    "mmlongdoc": "doc_qa",
    "slidevqa": "doc_qa",
}
dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}

@dataclass
class arguments:
    input_max_length: int = None
    generation_max_length: int = None
    generation_min_length: int = 0
    max_test_samples: int = None
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    use_chat_template: bool = None
    seed: int = 42
    test_name: str = None
    dataset: str = None
    output_dir: str = None
    
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_path(self):
        path = os.path.join(self.output_dir, "{args.dataset}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self))

        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        
        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None
    
    def get_averaged_metric(self):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            return None
        with open(path) as f:
            results = json.load(f)
        
        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}
        
        s = {m : v * (100 if m == "gpt4-f1" else 1) for m, v in s.items()}
        print("found scores:", s)
        return s
        
    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []        
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d['ctxs'])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)
        
        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="all")
    parser.add_argument("--use_chat_template", type=ast.literal_eval, choices=[True, False], default=True)
    parser.add_argument("--res_dir", type=str, default="/workspace/zw/data_dir/mmlb_result") # "/workspace/zw/data_dir/mmlb_result") "/home/zhaowei.wang/data_dir/mmlb_result"
    global_args = parser.parse_args()

    missing_model_list = []
    if global_args.model_name == "all":
        model_list = [model_name for model_name in os.listdir(global_args.res_dir) if os.path.isdir(os.path.join(global_args.res_dir, model_name))]
        models_configs = [
            {"model": model_name, "use_chat_template": global_args.use_chat_template} for model_name in model_list
        ]
    else:
        # comment out the models you don't want to include
        models_configs = [
            {"model": global_args.model_name, "use_chat_template": global_args.use_chat_template},
        ]

    # set your configs_test here
    configs = ["configs/vrag_all.yaml", "configs/vh_all.yaml", "configs/mm_niah_text_all.yaml", "configs/mm_niah_image_all.yaml", "configs/icl_all.yaml", "configs/summ_all.yaml", "configs/docqa_all.yaml"]
    datasets_configs = []
    for config in configs:
        c = yaml.safe_load(open(config))
        print(c)
        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
        if isinstance(c["input_max_length"], int):
            c["input_max_length"] = ",".join([str(c["input_max_length"])] * len(c["datasets"].split(",")))
        for d, t, l, g in zip(c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','), c['generation_max_length'].split(',')):
            datasets_configs.append({"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]), "input_max_length": int(l), "generation_max_length": int(g), "use_chat_template": c["use_chat_template"], "max_test_samples": c["max_test_samples"]})
    
    df = []
    for model in tqdm(models_configs):
        args = arguments()
        model_dir = model['model'].split("/")[-1]
        args.output_dir = f"{global_args.res_dir}/{model_dir}"
        missing_config_list = []
        for dataset in datasets_configs:
            args.update(dataset)
            args.update(model)

            metric = args.get_averaged_metric()
            dsimple, mnames = args.get_metric_name()

            if metric is None:
                missing_config_list.append([args.dataset, args.input_max_length])
                continue
                
            for k, m in metric.items():
                df.append({**asdict(args), **model,
                    "metric name": k, "metric": m, 
                    "dataset_simple": dsimple + " " + k, "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                })
        print("-" * 15 + "Task Summary" + "-" * 15)
        if missing_config_list:
            missing_model_list.append(model["model"])
            print("{} has missing task(s):".format(model["model"]))
            missing_config_list.sort(key=lambda x: x[1])
            for missing_config in missing_config_list:
                print(missing_config)
        else:
            print("{} has no missing task.".format(model["model"]))
        print("-" * 30)

    # all_df = pd.DataFrame(df)
    # lf_df = all_df.pivot_table(index=["model", "input_max_length", ], columns="dataset_simple", values="metric", sort=False)
    # lf_df = lf_df.reset_index()
    #
    # print(lf_df.to_csv(index=False))

    print(missing_model_list)
    # import pdb; pdb.set_trace()