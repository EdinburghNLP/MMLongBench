"""
Adopted from https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/utils/eval_utils.py
"""

import re
import string
import json
import math
import unicodedata
from math import isclose
from collections import Counter
from rouge_score import rouge_scorer

import torch
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def white_space_fix(text):
    return ' '.join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower()


def normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_answer_with_punc(s):
    return white_space_fix(remove_articles(lower(s)))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def drqa_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def drqa_exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction, ground_truth):
    """Check if the ground truth is a (soft) exact match substring of the prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def drqa_metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    # ground truth could be a string or a list of strings or a list of list of strings
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif isinstance(ground_truths[0], list):
        ground_truths = [ground_truth for ground_truths_list in ground_truths for ground_truth in ground_truths_list]

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def get_top_tokens(logits, tokenizer, top_k=10):
    """Get the top tokens and their probabilities from the logits."""
    top_tokens = []
    for logit in logits:
        a, b = torch.topk(torch.softmax(logit, dim=-1), top_k, dim=-1)
        l = [(y, f"{x*100:.02f}") for x, y in zip(a[0], tokenizer.convert_ids_to_tokens(b[0]))]
        top_tokens.append(l)
    return top_tokens


def parse_output(output, prefix="Answer:"):
    def lstrip_string(s, sub):
        return re.sub(f'^{re.escape(sub)}', '', s, flags=re.IGNORECASE)
    patterns = [re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),  # prefix + answer + sentence end
                re.compile(r"(?:^)(.*)(?:\n|$)")] # the beginning + answer + sentence end
    for pat in patterns:
        matches = pat.search(output)
        if matches is not None:
            return lstrip_string(matches[1].strip(), prefix).strip() # 0 index includes the non-capturing group # lstrip again because for chat models sometimes it will repeat the prefix
    # if still not found, return None, but should actually never get this case...
    return None


def extract_binary_label(prediction):
    prediction = normalize_answer_with_punc(prediction)
    pattern = r"\b(?:yes|no)\b(?!.*\b(?:yes|no)\b)"

    match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
    label_map = {"yes": 1, "no": 0}
    if match:
        return label_map[match.group(0)]
    else:
        if prediction.strip().isdigit():
            return int(prediction.strip())
        return -1

def turn_int_list(curr_list):
    new_list = []
    for item in curr_list:
        try:
            item = int(item)
            new_list.append(item)
        except:
            pass
    return new_list


def extract_cnt_list(prediction):
    prediction = normalize_answer_with_punc(prediction)

    pattern1 = r'\[[\d\s,]+\]' # r'\[[\d\s,]+\](?!.*\[[\d\s,]+\])'
    match = re.findall(pattern1, prediction, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            return json.loads(match[-1])
        except json.JSONDecodeError:
            pass

    pattern2 = r'\[.*?\]' # r'\[.*?\](?!.*\[.*\])'
    match = re.findall(pattern2, prediction, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            return turn_int_list(json.loads(match[-1]))
        except json.JSONDecodeError:
            pass

    numbers = re.findall(r'\d+', prediction)
    num_list = [int(x) for x in numbers]
    return num_list


def extract_choice_letter(prediction):
    prediction = white_space_fix(prediction)
    pattern1 = r"answer is \(?([A-J])\)?"
    match = re.search(pattern1, prediction)
    if match:
        choice = match.group(1)
        return ord(choice) - ord("A")

    pattern2 = r'[aA]nswer:\s*([A-J])'
    match = re.search(pattern2, prediction)
    if match:
        choice = match.group(1)
        return ord(choice) - ord("A")

    pattern3 = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern3, prediction, re.DOTALL)
    if match:
        choice = match.group(0)
        return ord(choice) - ord("A")

    return -1


def extract_class_num(prediction):
    prediction = white_space_fix(prediction)
    # pattern 1: "label: X" format
    pattern1 = r"label:\s*(\d+)"
    match = re.search(pattern1, prediction, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # pattern 2: Last number in the text
    pattern2 = r"\b(\d+)\b(?!.*\b\d+\b)"
    match = re.search(pattern2, prediction)
    if match:
        return int(match.group(1))
    # pattern 3: First number as fallback
    numbers = re.findall(r'\d+', prediction)
    if numbers:
        return int(numbers[0])
    return -1


# NOTE: we replace soft_acc with the sum of a list for robust evaluation
# NOTE: we still use the name soft_acc to distinguish it with other metrics in the code
def get_soft_acc(prediction, answer):
    return sum(prediction) == sum(answer)


def get_docqa_clean_string(s):
    s = str(s).lower().strip()
    s = s.replace(",", "")

    suffix_list = ["kg", "meters", "acres", "minutes", "miles", "mile",
                   "feet",
                   "million", "thousand", "billion", "mm", "m"]

    for suffix in suffix_list:
        s = re.sub(re.escape(suffix) + r'$', '', s).strip()

    # remove parenthesis
    # s = re.sub(r'\s*\([^)]*\)', "", s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().strip("$").strip()
    s = s.strip().strip("£").strip()
    s = s.strip().strip("%").strip()
    return s


def is_float_equal(reference, prediction, include_percentage=False) -> bool:
    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        if isclose(item, prediction, rel_tol=0.01):
            return True
    return False


def need_exact_match_check(s):
    patterns = [
        r'https://',
        r'.*\.(py|ipynb)$',
        r'^page',
        r'^\d+(-\d+|\s\d+)?$',
        r'(a\.m\.|p\.m\.)',
        r'^\d{4}[-/\s]\d{1,2}[-/\s]\d{1,2}$',  # YYYY-MM-DD, YYYY/MM/DD
        r'^\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4}$',  # DD-MM-YYYY, MM/DD/YYYY
        r'^\d{4}[-/\s]\d{1,2}$',  # YYYY-MM, YYYY/MM
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    ]

    return any(re.search(pattern, s) for pattern in patterns)


def extract_number_list(pred):
    pred_clean = pred.replace(',', '')
    num_pattern = r'(-?\d+(\.\d*)?|-?\.\d+)' # r'-?(\d+(\.\d*)?|\.\d+)'
    matches = re.findall(num_pattern, pred_clean)
    numbers = []
    for match_tuple in matches:
        match = match_tuple[0]
        try: # TODO filter inf number. Note this is added at the end of the experiment. previous is numers.append(float(match))
            num = float(match)
            if math.isfinite(num):
                numbers.append(num)
        except ValueError:
            continue
    return numbers


def get_str_type(num_str):
    try:
        num = float(get_docqa_clean_string(num_str))
        if num == int(num) and "%" not in num_str:
            return "Integer"
        else:
            return "Float"
    except:
        return "String"


def eval_docqa_score(gt, pred, answer_type):
    if answer_type == "Integer":
        gt = float(get_docqa_clean_string(str(gt)))
        assert int(gt) == gt
        gt = int(gt)
        pred = get_docqa_clean_string(str(pred))
        pred_num_list = [int(num) for num in extract_number_list(pred) if int(num) == num]
        score = any(gt == pred_num for pred_num in pred_num_list)
    elif answer_type == "Float":
        gt = float(get_docqa_clean_string(str(gt)))
        pred = get_docqa_clean_string(str(pred))
        pred_num_list = extract_number_list(pred)
        score = any(is_float_equal(gt, pred_num, include_percentage=True)
                    for pred_num in pred_num_list)
    elif answer_type in ["String", "None"]:
        if need_exact_match_check(gt):
            score = gt in pred
        else:
            score = f1_score(pred, gt)[0]
    elif answer_type == "List":
        gt_list = json.loads(gt)
        # merge f1 score text to prevent low precision
        merge_flag = [True if isinstance(item, str) and get_str_type(item) == "String" and not
                              need_exact_match_check(item)
                      else False for item in gt_list] # we merge all answers that are string and don't need EM for better recall
        merged_str = " ".join([item for item, m_flag in zip(gt_list, merge_flag) if m_flag]).strip()
        if merged_str:
            new_gt_list = [merged_str] + [item for item, m_flag in zip(gt_list, merge_flag) if not m_flag]
            gt_list = new_gt_list

        gt_score_list = [] # This is the greedy score similar to that used in LongDocURL
        for gt in gt_list:
            assert not isinstance(gt, list)
            if isinstance(gt, int):
                gt_type = "Integer"
            elif isinstance(gt, float):
                gt_type = "Float"
            else: # String answers can also represent int and float
                gt_type = get_str_type(gt)
            gt_score = eval_docqa_score(gt, pred, gt_type)
            gt_score_list.append(gt_score)
        score = sum(gt_score_list) / len(gt_list)
    else:
        raise KeyError("Wrong answer type:", answer_type)
    return float(score)


r_scorer = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
def calculate_metrics(prediction, answers, metrics):
    metric_list = [m.strip() for m in metrics.split(",")]
    metric_res = {}
    if "sub_em" in metric_list:
        sub_em = drqa_metric_max_over_ground_truths(substring_exact_match_score, prediction, answers)
        metric_res["sub_em"] = sub_em

    if "binary_acc" in metric_list:
        prediction, default_answer = prediction
        label = extract_binary_label(prediction)
        if label == -1:
            label = default_answer
        gt_label = extract_binary_label(answers)
        metric_res["acc"] = int(label == gt_label)

    # NOTE: We replace soft_acc with the sum of a list for robust evaluation
    # NOTE: We still use the name soft_acc to distinguish it with other metrics in the code
    if "soft_acc" in metric_list:
        cnt_list = extract_cnt_list(prediction)
        metric_res["soft_acc"] = get_soft_acc(cnt_list, answers)

    if "mc_acc" in metric_list:
        choice = extract_choice_letter(prediction)
        metric_res["mc_acc"] = int(choice == answers)

    if "cls_acc" in metric_list:
        class_num = extract_class_num(prediction)
        metric_res["cls_acc"] = int(class_num == answers)

    if "rouge" in metric_list:
        if isinstance(answers, str):
            answers = [answers]
        elif isinstance(answers[0], list):
            answers = [ground_truth for ground_truths_list in answers for ground_truth in ground_truths_list]

        rouges = [r_scorer.score(target=a, prediction=prediction) for a in answers]
        for k in r_scorer.rouge_types:
            metric_res[k + "_f1"] = max([r[k].fmeasure for r in rouges])
            metric_res[k + "_recall"] = max([r[k].recall for r in rouges])

    if "doc_qa" in metric_list:
        answer, answer_type = answers
        metric_res["doc_qa"] = eval_docqa_score(answer, prediction, answer_type)

    return metric_res

