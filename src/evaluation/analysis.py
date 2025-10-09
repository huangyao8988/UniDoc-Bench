import json
from collections import defaultdict

if __name__ == "__main__":
    vrag_path = "YOUR_DATA_PATH/QA/baseline/vrag/eval/hq_Electrical_Computer_and_Telecom_Engineering.json"  # e.g., "/path/to/data/QA/baseline/vrag/eval"
    with open(vrag_path, "r") as f:
        dataset_vrag = json.load(f)

    text_path = "YOUR_DATA_PATH/QA/baseline/text/eval/hq_Electrical_Computer_and_Telecom_Engineering.json"  # e.g., "/path/to/data/QA/baseline/text/eval"
    with open(text_path, "r") as f:
        dataset_text = json.load(f)

    results_question_recall = defaultdict(list)
    for idx in range(len(dataset_vrag["question"])):
        question = dataset_vrag["question"][idx]
        if dataset_text["question"][idx].strip() != question.strip():
            continue
        results_question_recall[question] = [dataset_text["gpt4-correctness-recall"][idx], dataset_vrag["gpt4-correctness-recall"][idx], idx]

    for question in results_question_recall:
        recall = results_question_recall[question]
        if recall[0] == 1.0 and recall[1] == 0.0: