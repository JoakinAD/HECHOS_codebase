"""Train-only event-attribute majority baseline and gold-span evaluation."""
from collections import Counter

EVENT_ATTRIBUTES = ["aspect", "certainty", "modality", "polarity", "pos", "pred", "special_cases", "tense", "time"]


def train_attribute_majorities(documents):
    values = {attribute: Counter() for attribute in EVENT_ATTRIBUTES}
    for document in documents:
        for mention in document.mentions:
            if mention.kind == "EVENT_MENTION":
                for attribute in EVENT_ATTRIBUTES:
                    values[attribute][mention.attributes.get(attribute, "")] += 1
    return {attribute: counts.most_common(1)[0][0] for attribute, counts in values.items()}


def attribute_metrics(documents, majorities):
    result = {}
    for attribute, prediction in majorities.items():
        gold = [m.attributes.get(attribute, "") for d in documents for m in d.mentions if m.kind == "EVENT_MENTION"]
        classes = {}
        for label in sorted(set(gold) | {prediction}):
            tp = sum(value == label and prediction == label for value in gold)
            fp = sum(value != label and prediction == label for value in gold)
            fn = sum(value == label and prediction != label for value in gold)
            precision, recall = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
            classes[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": gold.count(label)}
        result[attribute] = {"accuracy": sum(value == prediction for value in gold) / len(gold), "macro_f1": sum(item["f1"] for item in classes.values()) / len(classes), "classes": classes}
    return result
