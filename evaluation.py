import logging
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="evaluation")
def main(args):
    logger.info(OmegaConf.to_yaml(args))

    generated_data = load_dataset("json", data_files=args.generated_file, split="train")
    logging.info(f"Loaded {len(generated_data)} examples from {args.generated_file}")

    if args.limit:
        generated_data = generated_data.select(range(args.limit))

    if args.answer_processor:
        answer_processor = hydra.utils.instantiate(args.answer_processor, _convert_="object")
    else:

        def answer_processor(x):
            return x

    def map_load(example):
        generated = example[args.key_names["generated"]]
        example[args.key_names["cleaned"]] = answer_processor(generated)
        return example

    generated_data = generated_data.map(map_load)
    size = len(generated_data)

    results = {"local": {}, "global": {}, "raw": defaultdict(list)}
    for metric in args.metrics:
        obj = hydra.utils.instantiate(metric, key_names=args.key_names, _convert_="object")
        if obj.local:
            for example in tqdm(generated_data):
                calculation = obj.measure(example)
                for key, val in calculation.items():
                    results["raw"][key].append(val)
        else:
            calculation = obj.measure(generated_data)
            for key, val in calculation.items():
                results["global"][key] = val
        del obj
        torch.cuda.empty_cache()

    logging.info(f"Normalizing by size {size}")
    for key in results["raw"].keys():
        results["local"][key] = np.mean(results["raw"][key]).item()
        results["local"][key + "_std"] = np.std(results["raw"][key]).item()
        results["local"][key + "_se"] = (
            np.std(results["raw"][key], ddof=1).item() / np.sqrt(size).item()
        )
    results["raw"] = dict(results["raw"])
    results["local"]["size"] = size

    logging.info(f"Results: {results}")

    if getattr(args, "feature", None):
        left, right = args.generated_file.split("-test-")
        args.generated_file = f"{left}-{args.feature}-test-{right}"

    if args.results_file is None:
        args.results_file = Path(args.generated_file).stem + "-results.yaml"

    with open(args.results_file, "w") as f:
        yaml.dump(results, f, sort_keys=True)
    logging.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
