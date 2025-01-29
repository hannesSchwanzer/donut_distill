from nltk import edit_distance
from anls import anls_score
from typing import List

def calculate_metrics_funsd(ground_truth, predictions, strict=False):
    
    def funsd_result_to_dict(funsd_result):
        result = dict()
        for item in funsd_result:
            if not isinstance(item, dict):
                continue
            key = (item.get("text", ""), item.get("label", "")) if strict else item.get("text", "")
            result[key] = result.get(key, 0) + 1

        return result

    ground_truth_dict = funsd_result_to_dict(ground_truth)
    predictions_dict = funsd_result_to_dict(predictions)

    true_positives = 0
    for key, value in ground_truth_dict.items():
        true_positives += min(value, predictions_dict.get(key, 0))

    recall = true_positives / len(ground_truth) if true_positives != 0 else 0
    precision = true_positives / len(predictions) if true_positives != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    return f1_score, recall, precision

def calculate_metrics_docvqa(ground_truths: List[str], prediction: str):
    exact_match = prediction in ground_truths 
    anls = anls_score(prediction=prediction, gold_labels=ground_truths, threshold=0.5)

    return {
        "anls": anls,
        "exact_match": exact_match,
    }
