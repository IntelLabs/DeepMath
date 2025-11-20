import importlib
import logging
import os
import sys
from datetime import timedelta
from operator import itemgetter

import hydra
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import load_dataset
from omegaconf import OmegaConf
from trl import GRPOConfig

from deep_math.rewards import (
    code_format_reward,
    open_r1_accuracy_reward,
)
from deep_math.vllm_client import VLLMClient as myVLLMClient

logger = logging.getLogger(__name__)

# NOTE: Overriding TRL vllm_client with ours
vllm_client_mod = importlib.import_module("trl.extras.vllm_client")
vllm_client_mod.VLLMClient = myVLLMClient
sys.modules["trl.extras.vllm_client"] = vllm_client_mod

# NOTE: we now load our temperature-based GRPOTrainer
from deep_math.training_utils import GRPOTrainerTemperature, TemperatureScheduler


def setup_wandb(args: dict):
    """
    WANDB integration for tracking training runs.
    """
    env = {key: os.getenv(key) for key in os.environ}
    wandb.init(
        project=args["project"],
        job_type="train",
        group=args["experiment"],
        entity=args["wandb_entity"],
        name=args["experiment"],
        config={**args, **env},
        tags=["train"],
    )


@hydra.main(version_base=None, config_path="./configs", config_name="training")
def main(args):
    logger.info(OmegaConf.to_yaml(args))
    OmegaConf.set_struct(args, False)

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))])

    if accelerator.is_main_process:
        setup_wandb(OmegaConf.to_container(args))

    ds = load_dataset(args.data_file)
    ds = ds["train"]
    ds = ds.rename_column(args.input_key, "prompt")
    logger.info("Dataset was loaded.")

    temperature_scheduler = (
        TemperatureScheduler(
            OmegaConf.select(args, "temp_scheduling.temp_a_step", default=0),
            OmegaConf.select(args, "temp_scheduling.temp_a", default=1.2),
            OmegaConf.select(args, "temp_scheduling.temp_b_step", default=200),
            OmegaConf.select(args, "temp_scheduling.temp_b", default=0.6),
        )
        if OmegaConf.select(args, "temp_scheduling", default=None)
        else None
    )

    # NOTE: system prompt and fewshot examples are introduced here, as well as in the vllm server generating the
    # candidates because my math agent needs the system instruction at initialization time. Here we need to have the
    # right "prefix" for the GRPO completion training.
    module = importlib.import_module("deep_math.prompts")
    template = getattr(module, args.template)
    instruction = getattr(module, args.system_instruction) if args.system_instruction else None
    examples = open(args.fewshot_examples).read() if args.fewshot_examples else None
    system_prompt = instruction.format(examples=examples) if instruction else ""

    model_class = hydra.utils.instantiate(args.model, _convert_="object")
    logger.info("Model was loaded.")

    def format_answer(example):
        """
        We apply the template and make it into a conversation format.
        There's no response needed. The reward function would use the
        key `answer` to calculate the reward.
        We also remove the `prompt` key because TRL is looking for it
        in order to know whether chat templating is needed.
        """
        query = example["prompt"]

        messages = [
            {"role": "user", "content": template.format(question=query)},
        ]

        # TRL expects the prompt key and can apply chat template itself
        example["prompt"] = messages
        return example

    def format_answer_agent(example):
        """
        We apply the template and make it into a conversation format.
        There's no response needed. The reward function would use the
        key `answer` to calculate the reward.
        We also remove the `prompt` key because TRL is looking for it
        in order to know whether chat templating is needed.
        """
        query = example["prompt"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.format(question=query)},
        ]

        # TRL expects the prompt key and can apply chat template itself
        example["prompt"] = messages
        return example

    ds = ds.map(format_answer_agent if getattr(args, "use_agent", False) else format_answer)
    logger.info(f"Dataset was formatted. Use agent: {getattr(args, 'use_agent', False)}")

    train, dev = itemgetter("train", "test")(ds.train_test_split(args.dev_split, seed=args.seed))
    logger.info("Dataset was split")

    logger.info("Reward functions were loaded: %s", args.reward_funcs)
    logger.info("Reward weights: %s", args.reward_weights)

    reward_func_map = {
        "code_format": code_format_reward,
        "open_r1_accuracy": open_r1_accuracy_reward,
    }

    training_args = GRPOConfig(seed=args.seed, reward_weights=args.reward_weights, **args.train)

    trainer = GRPOTrainerTemperature(
        temperature_scheduler=temperature_scheduler,
        model=model_class.model,
        reward_funcs=[reward_func_map[f] for f in args.reward_funcs],
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
    )

    # NOTE: synchronization with vllm_server service. Some assertions and logging.
    if accelerator.is_main_process:
        vllm_server_config = trainer.vllm_client.get_config()["message"]
        wandb.config.update({"vllm_server_config": vllm_server_config}, allow_val_change=True)
        logger.info("VLLM server config: %s", vllm_server_config)

        assert args.max_agent_steps == vllm_server_config["max_steps"], (
            "Agent runs with a different MAX_STEPS parameter than training script."
        )
        if getattr(args, "max_agent_output", None) is not None:
            assert args.max_agent_output == vllm_server_config["max_agent_output"], (
                "Agent runs with a different MAX_AGENT_OUTPUT parameter than training script."
            )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_state()
    trainer.save_model(args.train.output_dir)


if __name__ == "__main__":
    main()
