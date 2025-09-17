import json
from llm_client import extract_event_info
from sklearn.metrics import accuracy_score


def load_dataset(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def evaluate(validation_file):
    data = load_dataset(validation_file)
    y_true, y_pred = [], []
    for item in data:
        user_input = item["messages"][1]["content"]
        expected = item["completion"]
        prediction = extract_event_info(user_input)

        # Normalize expected and prediction to JSON strings for comparison
        try:
            # Parse expected string as JSON, if possible
            expected_obj = json.loads(expected)
        except Exception:
            # If expected is not valid JSON, keep as string
            expected_obj = expected

        # Convert prediction obj to JSON string if it's a dict
        if isinstance(prediction, dict):
            prediction_str = json.dumps(prediction, sort_keys=True)
        else:
            prediction_str = str(prediction).strip()

        # Convert expected_obj to string for comparison
        if isinstance(expected_obj, dict):
            expected_str = json.dumps(expected_obj, sort_keys=True)
        else:
            expected_str = str(expected_obj).strip()

        y_true.append(expected_str)
        y_pred.append(prediction_str)

    acc = accuracy_score(y_true, y_pred)
    print(f"Validation Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate("data/validation_dataset.jsonl")
