from collections import Counter

import numpy as np
from math_verify import parse, verify


def majority(answers: list) -> str:
    """
    Find the most common answer in a list of answers.

    Need to be careful about hashability of the objects in the list.
    """

    def to_string(item):
        try:
            key = str(item)
        except:
            key = None
        return key, item

    key_to_obj = {}
    hashable_keys = []
    for item in answers:
        key, obj = to_string(item)
        if key is not None:
            key_to_obj[key] = obj  # only keeps last seen item per key
            hashable_keys.append(key)

    counter = Counter(hashable_keys)
    return key_to_obj[counter.most_common(1)[0][0]]


AGGREGATION_METHODS = {
    "majority": majority,
}


class MetricBase:
    """
    Base class for metrics.

    Metrics can be local or global; local means score are calculated per example.
    Global means score is calculated by looking at the entire dataset, e.g. fluency.
    """

    def __init__(self, key_names, **kwargs):
        self.key_names = key_names
        self.kwargs = kwargs
        self.field = self.key_names["cleaned"]
        self.target = self.key_names["label"]

    def measure(self, example: dict) -> dict[str, float]:
        pass


class MathVerify(MetricBase):
    """
    Math equality of latex expressions.

    Based on code from DeepScaler and DeepSeekMath.
    """

    def __init__(self, key_names, aggregation="majority", **kwargs) -> None:
        super().__init__(key_names, **kwargs)
        self.local = True
        self.aggregation = aggregation

    def measure(self, example: dict):
        input = example[self.field]
        target = example[self.target]

        # assert isinstance(input, str), f"Generated text should be a string: {input}"

        gold = parse("$" + target + "$")  # Latex environment required, from experimentation
        if isinstance(input, str):
            input = [input]

        answers = []
        for i in input:
            try:
                answers.append(parse(i))
            except Exception as e:
                # Handle or log the exception as needed
                answers.append(parse(""))

        aggregated_answer = AGGREGATION_METHODS[self.aggregation](answers)

        # Order here is important!
        agg_score = int(verify(gold, aggregated_answer))
        scores_arr = [verify(gold, a) for a in answers]
        ave_score = np.mean(scores_arr).item()
        pass_k = any(scores_arr)
        score = int(verify(gold, answers[0]))

        return {
            f"math_verify_{self.aggregation}@{len(answers)}": agg_score,
            f"math_verify_pass@{len(answers)}": pass_k,
            f"math_verify_averaged@{len(answers)}": ave_score,
            "math_verify": score,
        }


class OutputLength(MetricBase):
    def __init__(self, key_names: dict, **kwargs) -> None:
        """
        Initializes an instance of the class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        from transformers import AutoTokenizer

        super().__init__(key_names, **kwargs)
        self.local = True
        self.tok = AutoTokenizer.from_pretrained(
            kwargs.get("model", "Qwen/Qwen3-4B-Thinking-2507"), fast=True
        )

    def measure(self, example):
        input = example[self.key_names["generated"]]
        if isinstance(input, str):
            input = [input]

        return {"length": np.mean([len(self.tok.tokenize(i)) for i in input]).item()}
