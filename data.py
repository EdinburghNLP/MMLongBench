import os
import json
import random
from functools import partial
from datasets import load_dataset
from torch.utils.data import Dataset
from utils import calculate_metrics, parse_output

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pipline
# 1. template 2. load data 3. max test_sample sampling 4. update context 5. post process (later)
# field
# 1. context 2. question 3. answer 4. image_list


def default_post_process(output, example, metrics=None, prefix=None):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    prediction = output["output"]
    answer = example["answer"]
    mets = calculate_metrics(prediction, answer, metrics)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(prediction, prefix=prefix)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer, metrics)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    return mets, {"parsed_output": parsed_pred}


### instruction\n\nDocument (Title: {title}): {text}\n\nDocument (Title: {title}): {text}\n\nQuestion: {question}
def load_vrag(args, path, max_test_samples=None):
    """
    Load the data for vision rag
    """
    user_template = "Use the given documents to write a concise and short answer to the question about the entity shown in the image. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = set(data["id"])
        key_list = random.sample(sorted(key_list), min(max_test_samples, len(key_list)))
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    passage_template = "Document (Title: {title}): {text}"

    def update(sample):
        passage_text = "\n\n".join([passage_template.format(**c) for c in sample["ctxs"]])
        question = "<image>" + sample["question"]
        return {"context": passage_text,
                "question": question,
                "image_list": [os.path.join(args.image_file_root, sample["image"])]}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="sub_em", prefix=system_template)
    }


def load_vrag_occlusion(args, path, max_test_samples=None):
    """
    Load the data for vision rag
    """
    user_template = "Use the given documents to write a concise and short answer to the question about the entity shown in the image. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = set(data["id"])
        key_list = random.sample(sorted(key_list), min(max_test_samples, len(key_list)))
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)
        import pandas as pd
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=["id"], keep="first")
        from datasets import Dataset as HFDataset
        data = HFDataset.from_pandas(df)

    passage_template = "Document (Title: {title}): {text}"

    def update(sample):
        # filter pos
        pos_doc_id = [c["doc_id"] for c in sample["positive_ctxs"]]
        neg_ctxs = [c for c in sample["ctxs"] if c["doc_id"] not in pos_doc_id]
        passage_text = "\n\n".join([passage_template.format(**c) for c in neg_ctxs])
        question = "<image>" + sample["question"]
        return {"context": passage_text,
                "question": question,
                "image_list": [os.path.join(args.image_file_root, sample["image"])]}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="sub_em", prefix=system_template)
    }


def load_text_rag(args, path, max_test_samples=None):
    """
    Load the data for textual rag
    """
    user_template = "Use the given documents to write a concise and short answer to the question about a named entity. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = set(data["id"])
        key_list = random.sample(sorted(key_list), min(max_test_samples, len(key_list)))
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    passage_template = "Document (Title: {title}): {text}"

    def update(sample):
        passage_text = "\n\n".join([passage_template.format(**c) for c in sample["ctxs"]])
        question = sample["original_question"]
        return {"context": passage_text,
                "question": question,
                "image_list": []}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="sub_em", prefix=system_template)
    }

class DefaultAnswerGenerator():
    def __init__(self):
        self.idx = 0
        self.default_answer_list = [0, 1, 1, 0]
    def get_default_answer(self):
        default_answer  = self.default_answer_list[self.idx]
        self.idx = (self.idx + 1) % len(self.default_answer_list)
        return default_answer

### instruction\n\n<image>\n<image>\n<image>\n\nQuestion: {question}
def load_visual_haystack(args, path, max_test_samples=None):
    user_template = "You are given a set of images. Please answer the question in Yes or No based on the given images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = set(data["id"])
        key_list = random.sample(sorted(key_list), min(max_test_samples, len(key_list)))
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    def update(sample):
        image_list = sample["ctxs"]
        passage_text = "\n".join(["<image>"] * len(image_list))
        return {"context": passage_text,
                "image_list": [os.path.join(args.image_file_root, image) for image in sample["ctxs"]]}
    data = data.map(update, num_proc=args.preprocessing_num_workers)

    default_answer_generator = DefaultAnswerGenerator()

    def vh_post_process(output, example):
        """
        Returns: metrics (dict) and additional info to update the original sample with (dict)
        """
        prediction = output["output"]
        answer = example["answer"]
        default_answer = default_answer_generator.get_default_answer()
        mets = calculate_metrics((prediction, default_answer), answer, "binary_acc")
        return mets, {"parsed_output": prediction, "default_answer": default_answer}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": vh_post_process
    }


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}
def load_mm_niah_text(args, path, max_test_samples=None):
    user_template = "You are given interleaved text and images. Please answer the question based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = set(data["id"])
        key_list = random.sample(sorted(key_list), min(max_test_samples, len(key_list)))
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    def update(sample):
        paragraph_list = sample["ctxs"]
        passage_text = "\n\n".join([p["text"] for p in paragraph_list])
        image_list = [os.path.join(args.image_file_root, image) for image in sample["image_list"]]
        return {"context": passage_text, "image_list": image_list}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    if "retrieval" in os.path.basename(path):
        metric = "sub_em"
    elif "counting" in os.path.basename(path):
        metric = "soft_acc"
    elif "reasoning" in os.path.basename(path):
        metric = "sub_em"
    else:
        raise NameError(f"Wrong mm-niah task: {os.path.basename(path)}")
    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics=metric, prefix=system_template)
    }


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}"
def load_mm_niah_image_count(args, path, max_test_samples=None):
    user_template = "You are given interleaved text and images. Please answer the question based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = set(data["id"])
        key_list = random.sample(sorted(key_list), min(max_test_samples, len(key_list)))
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    def update(sample):
        paragraph_list = sample["ctxs"]
        passage_text = "\n\n".join([p["text"] for p in paragraph_list])
        # question of image count has <image> already
        # process image
        image_list = sample["image_list"] + sample["needle_image_list"]
        image_list = [os.path.join(args.image_file_root, image) for image in image_list]

        return {"context": passage_text, "image_list": image_list}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="soft_acc", prefix=system_template)
    }


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}"
# used for multiple choices problems in mm-niah-image
def load_mm_niah_image_mc(args, path, max_test_samples=None):
    user_template = "You are given interleaved text and images. Please answer the question with the option's letter (A, B, etc.) based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = set(data["id"])
        key_list = random.sample(sorted(key_list), min(max_test_samples, len(key_list)))
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    def update(sample):
        paragraph_list = sample["ctxs"]
        passage_text = "\n\n".join([p["text"] for p in paragraph_list])
        question = sample["question"]
        # add choices to question
        question += "".join([f"\n{chr(c_idx + ord('A'))}. <image>" for c_idx in range(len(sample["choices_image"]))])

        # add choices image to image_list
        image_list = sample["image_list"] + sample["choices_image"]
        image_list = [os.path.join(args.image_file_root, image) for image in image_list]

        return {"context": passage_text, "image_list": image_list, "question": question}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="mc_acc", prefix=system_template)
    }


def load_icl_er(args, path, max_test_samples=None):
    user_template = "You need to recognize entities in images. Use the provided mapping from the image to label to assign a label to the test image. Only output \"label: {{label}}\" and nothing else.\n\nTraining examples:\n{context}\n\nNow classify this image: {question}"
    item_template = "<image>\nlabel: {label}"
    system_template = "label:"

    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    with open(path) as fin:
        data = json.load(fin)

    exemplar_list_by_domain = {domain: data_dict["exemplar_list"] for domain, data_dict in data.items()}
    test_examples = [{"domain": domain, "example": example} for domain, data_dict in data.items() for example in data_dict["test_example"]]

    if max_test_samples is not None:
        random.shuffle(test_examples)
        test_examples = test_examples[: min(max_test_samples, len(test_examples))]

    def update(sample):
        domain, example = sample["domain"], sample["example"]
        exemplar_list = exemplar_list_by_domain[domain]

        sampled_exemplar = random.choice(exemplar_list)
        for i in range(len(sampled_exemplar)): # this is a list of demonstration rounds
            random.shuffle(sampled_exemplar[i])
        sampled_exemplar = [item for sublist in sampled_exemplar for item in sublist]

        question = "<image>"
        image_list = [item["image"] for item in sampled_exemplar] + [example["image"]]
        image_list = [os.path.join(args.image_file_root, image) for image in image_list]
        context = "\n\n".join([item_template.format(label=item["id"]) for item in sampled_exemplar])

        return {"context": context, "question": question,
                "image_list": image_list, "answer": example["answer"]}

    from datasets import Dataset as HFDataset
    test_examples = HFDataset.from_list(test_examples)
    test_examples = test_examples.map(update, num_proc=args.preprocessing_num_workers)

    return {
        "data": test_examples,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="cls_acc", prefix=system_template)
    }


def load_gov_report(args, path, max_test_samples=None):
    user_template = "You are given a government report from U.S. Government Accountability Office (GAO), and you are tasked to summarize the report. Write a concise summary (around 550 words) organized in multiple paragraphs. Where applicable, the summary should contain a short description of why GAO did this study, what GAO found, and what GAO recommends.\n\nGovernment Report:\n{context}\n\nNow please summarize the report."
    item_template = "Document {doc_id:.15} (page {page_id}): <image>"
    system_template = "Summary:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        data = data.shuffle(seed=args.seed).select(range(min(len(data), max_test_samples)))

    def update(sample):
        page_prompt_list = [item_template.format(
            doc_id=image_path.split("/")[-2], page_id=image_path.split("page")[1].split(".")[0])
            for image_path in sample["image_list"]]
        image_list = [os.path.join(args.image_file_root, image) for image in sample["image_list"]]
        passage_text = "\n\n".join(page_prompt_list)

        answer = "\n\n".join([aspect['section_title'] + ':\n'
                        + '\n'.join(aspect['paragraphs']) for aspect in sample["summary"]])

        return {"context": passage_text, "image_list": image_list, "answer": answer}

    data = data.map(update, num_proc=args.preprocessing_num_workers)
    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="rouge", prefix=system_template)
    }


def load_multi_lexsum(args, path, max_test_samples=None):
    user_template = "You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description of the background, the parties involved, and the outcomes of the case.\n\nLegal documents:\n{context}\n\nNow please summarize the case."
    item_template = "Document {doc_id:.15} (page {page_id}): <image>"
    system_template = "Summary:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        data = data.shuffle(seed=args.seed).select(range(min(len(data), max_test_samples)))

    def update(sample):
        page_prompt_list = [item_template.format(
            doc_id=image_path.split("/")[-2], page_id=image_path.split("page")[1].split(".")[0])
            for image_path in sample["image_list"]]
        image_list = [os.path.join(args.image_file_root, image) for image in sample["image_list"]]
        passage_text = "\n\n".join(page_prompt_list)
        answer = sample["summary"]

        return {"context": passage_text, "image_list": image_list, "answer": answer}

    data = data.map(update, num_proc=args.preprocessing_num_workers)
    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="rouge", prefix=system_template)
    }


def load_doc_qa(args, path, max_test_samples=None):
    user_template = "You are given a document with text and images, and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write 'Not answerable.' Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    item_template = "Document {doc_id:.15} (page {page_id}): <image>"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        data = data.shuffle(seed=args.seed).select(range(min(len(data), max_test_samples)))

    def update(sample):
        image_list = sample["page_list"]
        page_prompt_list = [item_template.format(
            doc_id=image_path.split("/")[-2], page_id=image_path.split("page")[-1].split(".")[0]) for image_path in image_list]
        image_list = [os.path.join(args.image_file_root, image) for image in image_list]
        passage_text = "\n\n".join(page_prompt_list)
        question = sample["question"]
        doc_id = sample["doc_name"]
        question = "Based on Document {doc_id:.15}, answer the following question. ".format(doc_id=doc_id) + question

        return {"context": passage_text, "image_list": image_list, "question": question}

    data = data.map(update, num_proc=args.preprocessing_num_workers)

    def dcoqa_post_process(output, example):
        """
        Returns: metrics (dict) and additional info to update the original sample with (dict)
        """
        prediction = output["output"]
        answer = [example["answer"], example["answer_format"]]
        parsed_pred = parse_output(prediction, prefix=system_template)
        if parsed_pred is None:
            parsed_pred = prediction
        mets = calculate_metrics(parsed_pred, answer, "doc_qa")
        return mets, {"parsed_output": parsed_pred}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": dcoqa_post_process
    }


def load_text_doc_qa(args, path, max_test_samples=None):
    user_template = "You are given a document, and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write 'Not answerable.' Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    item_template = "Document {doc_id:.15} (page {page_id}): \n{page_text}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        data = data.shuffle(seed=args.seed).select(range(min(len(data), max_test_samples)))

    def update(sample):
        image_list = sample["page_list"]
        image_text_list = sample["page_text_list"]
        assert len(image_list) == len(image_text_list)
        page_prompt_list = [item_template.format(
            doc_id=image_path.split("/")[-2],
            page_id=image_path.split("page")[-1].split(".")[0],
            page_text=image_text) for image_path, image_text in zip(image_list, image_text_list)]

        passage_text = "\n\n".join(page_prompt_list)
        question = sample["question"]
        doc_id = sample["doc_name"]
        question = "Based on Document {doc_id:.15}, answer the following question. ".format(doc_id=doc_id) + question

        return {"context": passage_text, "image_list": [], "question": question} # no image

    data = data.map(update, num_proc=args.preprocessing_num_workers)

    def dcoqa_post_process(output, example):
        """
        Returns: metrics (dict) and additional info to update the original sample with (dict)
        """
        prediction = output["output"]
        answer = [example["answer"], example["answer_format"]]
        parsed_pred = parse_output(prediction, prefix=system_template)
        if parsed_pred is None:
            parsed_pred = prediction
        mets = calculate_metrics(parsed_pred, answer, "doc_qa")
        return mets, {"parsed_output": parsed_pred}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": dcoqa_post_process
    }


def load_data(args, dataset, path=None):
    if "infoseek" in dataset or "viquae" in dataset:
        data = load_vrag(args, path, max_test_samples=args.max_test_samples)
    elif "triviaqa" in dataset:
        data = load_text_rag(args, path, max_test_samples=args.max_test_samples)
    elif "occ_vrag" in dataset:
        data = load_vrag_occlusion(args, path, max_test_samples=args.max_test_samples)
    elif "vh_single" in dataset or "vh_multi" in dataset:
        data = load_visual_haystack(args, path, max_test_samples=args.max_test_samples)
    elif "mm_niah" in dataset:
        if "text" in path:
            data = load_mm_niah_text(args, path, max_test_samples=args.max_test_samples)
        else:
            if "retrieval" in path or "reasoning" in path:
                data = load_mm_niah_image_mc(args, path, max_test_samples=args.max_test_samples)
            else:
                data = load_mm_niah_image_count(args, path, max_test_samples=args.max_test_samples)
    elif dataset == "text-haystack_retrieval-image": # our ablation of removing other text
        data = load_mm_niah_image_mc(args, path, max_test_samples=args.max_test_samples)
    elif any(key in dataset for key in ["cars196", "food101", "inat2021", "sun397"]):
        data = load_icl_er(args, path, max_test_samples=args.max_test_samples)
    elif "gov-report" in dataset:
        data = load_gov_report(args, path, max_test_samples=args.max_test_samples)
    elif "lexsum" in dataset:
        data = load_multi_lexsum(args, path, max_test_samples=args.max_test_samples)
    elif any(key in dataset for key in ["longdocurl", "mmlongdoc", "slidevqa"]):
        data = load_doc_qa(args, path, max_test_samples=args.max_test_samples)
    elif dataset == "text_doc":
        data = load_text_doc_qa(args, path, max_test_samples=args.max_test_samples)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    return data


class TestItemDataset(Dataset):
    def __init__(self, data, llm, processor):
        self.data = data
        self.llm = llm
        self.processor = processor

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        inputs = self.llm.prepare_inputs(self.data["data"][idx], self.data)
        original_text = None
        if hasattr(inputs, "input_ids") or "input_ids" in inputs:
            if hasattr(self.llm, "safe_decode"):
                original_text = self.llm.safe_decode(inputs["input_ids"][0], skip_special_tokens=False)
            else:
                original_text = self.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        return inputs, original_text
