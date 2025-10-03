# MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly

---

<p align="center">
    <a href="https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/blob/main/README.md" target="_blank" rel="noopener noreferrer">
        <img alt="paper" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue">
    </a>
    <a href="https://arxiv.org/abs/2505.10610" target="_blank" rel="noopener noreferrer">
        <img alt="paper" src="https://img.shields.io/badge/paper-paper?logo=arxiv&logoColor=%23B31B1B&labelColor=white&color=%23B31B1B">
    </a>
    <a href="https://zhaowei-wang-nlp.github.io/MMLongBench-page/" target="_blank" rel="noopener noreferrer">
        <img alt="homepage" src="https://img.shields.io/badge/🏠-Homepage-green">
    </a>
    <a href="https://github.com/zhaowei-wang-nlp/DivScene/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
        <img alt="license" src="https://img.shields.io/badge/Code%20License-MIT-green">
    </a>

</p>

MMLongBench is a comprehensive benchmark covering a diverse set of long-context vision-language tasks, to evaluate long-context vision-language models (LCVLMs) effectively and thoroughly. 
It is composed of 13,331 examples spanning five different categories of downstream tasks, including Visual RAG, NIAH, Many-Shot ICL, Summarization (based on PDF documents), and Long-Document VQA.
Please check out the paper for more details, and this repo will detail how to run the evaluation.

## Quick Links

- [Setup](#setup)
- [Data](#data)
- [Running evaluation](#running-evaluation)
- [Adding new models](#adding-new-models)
- [Adding new tasks](#adding-new-tasks)
- [Tips](#tips)
- [Contacts](#contacts)
- [Acknowledgment](#acknowledgment)


## Setup

Please install the necessary packages with (using a virtual environment with anaconda is recommended, tested with python 3.11):
```bash
pip install -r requirements.txt
```

if you wish to use API models, you need to install the corresponding packages.
```bash
pip install openai # OpenAI API (GPT), we also used OpenAI SDK to run Anthropic API (Claude) 
pip install google-genai
```
You also need to set the correct environmental variables according to the official documentation of the API SDK.
Please check ```vlm_model/gemini.py``` and ```vlm_model/openai_model.py``` for more details.

## Data

<div align="center">
    <img src="assets/overview_page.jpg" alt="Model"/>
    <br>
    <span>An overview of MMLongBench.</span>
</div>

Our benchmarks have two parts:
1. You can download the image data with the command:
```bash
bash scripts/download_image_data.sh
```
This will iteratively download the .tar.gz files of images for each task and then decompress them to a `mmlb_image` directory.

2. You can download the text data with this command:
```bash
bash scripts/download_text_data.sh
```
This will decompress the 0_mmlb_data.tar.gz into the `mmlb_data` directory. All the test files are stored in jsonl format.

Our dataset is hosted on this [HuggingFace Dataset](https://huggingface.co/datasets/ZhaoweiWang/MMLongBench)

### Using Image URL for API models
We find that when using API models, uploading images are very slow.
Thus, we upload uncompressed images to the Huggingface Dataset [image_collection](https://huggingface.co/datasets/ZhaoweiWang/image_collection).
For GPT-4o and Claude-3.7-Sonnet, our code can use URLs instead of Base64 image encoding.


## Running evaluation

To run the evaluation, just select one of the config files from the configs directory. You can override any settings from the config file or introduce new arguments directly via the command line (refer to arguments.py for details).

```bash
model_name=Qwen/Qwen2.5-VL-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "vrag"; do
    python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir {your output directory}/{dir_name} \
    --test_file_root {your data directory}/mmlb_data \
    --image_file_root {your data directory}/mmlb_image \
    --num_workers 16 --test_length 8
done
```

For shorter tasks, such as 8K and 16K, usually a single GPU is enough. 
Thus, we provide the scripts ```scripts/eval_task_manager.py```, which can run multiple config files at the same time.
We list the commands for using it in ```scripts/run_eval.sh```.

This will output the results file under the output directory in two files: `.json` contains all the data point details while `.json.score` only contain the aggregated metrics.

If you tested new models, please email me the result files and I will add them to the spreadsheet!
See [Contacts](#contacts) for my email.

### LLM-based evaluation (GPT-4o metric)
The code of running LLM-based evaluation for summarization is in ```scripts/eval_gpt4_summ.py```.
The corresponding command to use it is in ```scripts/eval_gpt4_summ.sh```

Additionally, we provide pre-generated atomic claims extracted from gold summaries using GPT-4o. The files ```mmlb_data/summ/gov_claims.jsonl``` and ```mmlb_data/summ/lexsum_claims.jsonl``` contain these atomic claims and can be directly used for GPT-4o-based evaluation (precision/recall/fluency).

## Adding new models
We comprehensively evaluated 46 models using HuggingFace. Meanwhile, all the models count image tokens by the number of 
14x14 patches, followed by a 2x2 pixel unshuffle, or similar methods 
(e.g., first tile the image with the 336x336 size, then use 14x14 patch size and 2x2 pixel unshuffle).
The pixel unshuffle is very important to compress visual tokens.

Check whether your model is easily accessible using HuggingFace Transformers and whether your model is token efficient (with pixel unshuffle).

To add a new model, add a new python scripts called ```vlm_model/{your model name}.py```.
Meanwhile, you need to change ```load_LLM(args)``` function in ```vlm_model/__init__.py``` to correctly load your model.

Furthermore, your model scripts should implement `format_chat` (constructing the chat format data with both image and text)
, `prepare_inputs` (use processor to tokenize text and preprocess image), and `generate` functions. 
Please refer to the existing classes, such as ```vlm_model/qwen2_vl.py```.



## Adding new tasks
To add a new task/dataset:
1. you need to add a new config that specify the fields, such as input_max_length, generation_length,
test_files, etc.
2. you just need to modify the `data.py` file. Following our current data loading functions, such as ```def load_vrag(args, path, max_test_samples=None):```, and also remember to revise ```def load_data(args, dataset, path=None):``` to ensure your function is used.
3. you need to add your metric for the new task in `utils.py`

## Tips

<details>

<summary>Check Missing Evaluation Tasks</summary>
We provide a script to quickly check which evaluation tasks have not been completed for your model:

```bash
python scripts/check_missing.py
```

</details>

<details>

<summary>Figure Drawing</summary>

We provide all the scripts for drawing the figures in our paper in the folder ```figure_scripts```.
We can easily change them to meet your own requirements.
</details>

## Contacts

For any questions, please email `zwanggy@cse.ust.hk`.


## Citation
```bibtext
@inproceedings{wang2025mmlongbenchbenchmarkinglongcontextvisionlanguage,
      title={MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly}, 
      author={Zhaowei Wang and Wenhao Yu and Xiyu Ren and Jipeng Zhang and Yu Zhao and Rohit Saxena and Liang Cheng and Ginny Wong and Simon See and Pasquale Minervini and Yangqiu Song and Mark Steedman},
      year={2025},
      eprint={2505.10610},
      booktitle={The 39th (2025) Annual Conference on Neural Information Processing Systems},
      url={https://arxiv.org/abs/2505.10610}, 
}
```

## Acknowledgment
The code is built based on [HELMET](https://github.com/princeton-nlp/HELMET/tree/main).
We made extensive revisions to evaluate LVLMs, including configs, arguments, eval.py, metrics, data loading.
The main difference is that, since LVLMs have non-unified function APIs, we write a script for each model in ```vlm_model```.

## Misuse for malicious purposes
This dataset is constructed to support the development of Long-Context Vision-Language Models. It is strictly forbidden to use this dataset for other usage with high risk, such as generating or disseminating false, misleading, or harmful content, or for activities involving privacy violation, fraud, discrimination, harassment, violence, pornography, or any illegal purposes. Users must not use this dataset to train, fine-tune, or deploy any models or applications that violate laws, regulations, or ethical standards. By using this dataset, you agree to bear all legal and ethical responsibilities resulting from any misuse.

