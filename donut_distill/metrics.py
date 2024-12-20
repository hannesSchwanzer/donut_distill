def calculate_metrics(ground_truth, predictions):
    
    def funsd_result_to_dict(funsd_result):
        result = dict()
        for item in funsd_result:
            key = (item.get("text", ""), item.get("label", ""))
            result[key] = result.get(key, 0) + 1

        return result

    ground_truth_dict = funsd_result_to_dict(ground_truth)
    predictions_dict = funsd_result_to_dict(predictions)

    true_positives = 0
    for key, value in ground_truth_dict.items():
        true_positives += min(value, predictions_dict.get(key, 0))

    recall = true_positives / len(ground_truth)
    precision = true_positives / len(predictions)

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score, recall, precision

