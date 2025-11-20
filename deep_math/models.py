import importlib
import logging
import types
from pathlib import Path

from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class HFInference:
    """
    Class for running HF model inference locally with a prompt.
    """

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype=None,
        device_map="auto",
        use_vllm=False,
        num_gpus=1,
        vllm_params={},
        instruction: str | None = None,
        instruct_in_prompt: bool = False,
        template="basic_template",
        template_map: dict = None,
        extra_keys: dict = None,
        examples: Path = None,
        lora_path=None,
        lora_path_2nd=None,
        math_agent=False,
        max_steps: int = None,  # For the math agent, default is 10
        max_agent_output: int = None,
        sampling: int = 1,
        generation=None,
        task="text-generation",
        **kwargs,
    ):
        self.model_name = model_name_or_path
        self.device_map = device_map
        self.torch_type = torch_dtype
        self.generation_kwargs = generation
        self.use_vllm = use_vllm
        self.template_map = template_map
        self.examples = open(examples).read() if examples else None
        self.extra_keys = extra_keys
        self.math_agent = math_agent
        self.sampling = sampling

        self.instruct_in_prompt = instruct_in_prompt

        module = importlib.import_module("deep_math.prompts")
        self.template = getattr(module, template)
        logger.info(f"Using the following template: {self.template}")
        self.instruction = getattr(module, instruction) if instruction else None
        logger.info(f"Using the following instruction: {self.instruction}")

        self.lora_path = lora_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.config = AutoConfig.from_pretrained(self.model_name, **kwargs)
        self.config.torch_dtype = torch_dtype or "auto"

        if self.use_vllm:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest

            self.lora_request = (
                LoRARequest("adapter", 1, lora_path=self.lora_path) if self.lora_path else None
            )

            self.sampling_params = SamplingParams(
                **{
                    "temperature": generation.get("temperature") or 1,
                    "top_p": generation.get("top_p") or 0.9,
                    "top_k": generation.get("top_k") or -1,
                    "repetition_penalty": generation.get("repetition_penalty") or 1.0,
                    "max_tokens": generation.get("max_new_tokens") or 100,
                }
            )

            if math_agent:
                from .agent import MathAgent, VLLMCustom

                self.vllm = VLLMCustom(
                    model_id=self.model_name,
                    lora_request=self.lora_request,
                    model_kwargs=dict(
                        tensor_parallel_size=num_gpus,
                        max_model_len=50_000,  # NOTE: 50k context length. More than enough, maybe too much but neesd to be specified.
                        enable_lora=True if lora_path else False,
                        **vllm_params,
                    ),
                )

                self.agent = MathAgent(
                    tools=[],
                    max_steps=max_steps or 10,
                    max_agent_output=max_agent_output or 20000,
                    model=self.vllm,
                )

                logger.info(f"Loaded math agent with prompt: {self.agent.system_prompt}")

            else:
                self.vllm = LLM(
                    model=self.model_name,
                    tensor_parallel_size=num_gpus,
                    max_model_len=generation.get("max_new_tokens", 8096) + 256,
                    enable_lora=True if lora_path else False,
                    **vllm_params,
                )

        else:
            if math_agent:
                raise NotImplementedError(
                    "Math agent is not supported in non-vLLM mode. Please set `use_vllm=True`."
                )
                from smolagents import TransformersModel

                from .agent import MathAgent, substring_stopping_criteria

                self.model = TransformersModel(
                    model_id=self.model_name,
                    device_map=self.device_map,
                    torch_dtype=self.config.torch_dtype,
                    max_new_tokens=self.generation_kwargs["max_new_tokens"],
                )

                self.model.make_stopping_criteria = types.MethodType(
                    substring_stopping_criteria, self.model
                )

                self.agent = MathAgent(
                    tools=[],
                    system_prompt=self.instruction.format(examples=self.examples)
                    if self.instruction
                    else "",
                    model=self.model,
                    stream_outputs=False,
                )

                logger.info(f"Loaded math agent with prompt: {self.agent.system_prompt}")

            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, config=self.config, device_map=device_map, **kwargs
                )

                if lora_path:
                    logger.info(f"Loading LORA: {lora_path}")
                    self.model.load_adapter(lora_path)

                self.pipe = pipeline(
                    task=task,
                    model=self.model,
                    tokenizer=self.tokenizer,
                )

    def generate(self, example: dict) -> str | list[str]:
        """
        Given an input, generate a response.
        """

        # We move task instruction to system instruction which requires template filling.

        prompt = self.template.format(
            **{k: example[v] for k, v in self.template_map.items()},
            **{k: v for k, v in self.extra_keys.items()} if self.extra_keys else {},
        )

        messages = []

        if self.instruction:
            system_prompt = (
                self.instruction.format(examples=self.examples)
                if self.examples
                else self.instruction
            )
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=True,
            max_length=(
                self.config.max_position_embeddings - self.generation_kwargs["max_new_tokens"]
            ),
        )
        if self.math_agent:
            ans = []
            for _ in range(self.sampling):
                try:
                    output = self.agent.run(chat_prompt, sampling_params=self.sampling_params)
                    ans.append(str(output))
                except Exception as e:
                    logger.error(f"Error in agent run: {e}")
                    ans.append("")

            if self.sampling == 1:
                ans = ans[0]

            return ans

        else:
            if self.use_vllm:
                # We use vLLM batch sampling only in non-agent mode
                self.sampling_params.n = self.sampling

                output = self.vllm.generate(
                    prompts=chat_prompt,
                    sampling_params=self.sampling_params,
                    lora_request=(self.lora_request if self.lora_request else None,),
                )
                ans = [out.text for out in output[0].outputs]

                if self.sampling == 1:
                    ans = ans[0]

                return ans

            else:
                output = self.pipe(chat_prompt, **self.generation_kwargs)
                return output[0]["generated_text"]


class HFTrain:
    """
    Class for training HF models locally.
    """

    def __init__(
        self,
        model_name_or_path,
        torch_dtype,
        device_map,
        lora: LoraConfig = None,
        generation=None,
        completion_start: str = "",
        instruction_in_prompt=None,
        max_sequence_len=None,
        **kwargs,
    ):
        """
        Args:
            model_name_or_path: str - HF model name or path.
            torch_dtype: str - torch dtype for the model.
            device_map: dict - device map for the model.
            lora: dict - LoRA adapter config.
            trained_adapter: str - path to a trained adapter. It will be loaded and merged.
            generation: dict - generation kwargs.
            completion_start: str - used to find the start of the completion in the prompt.
            instruction_in_prompt: bool - whether to include the instruction in the prompt for models without system role.
        """
        self.model_name = model_name_or_path
        self.complete_start = completion_start
        self.generation_kwargs = generation
        self.max_sequence_len = max_sequence_len

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.config = AutoConfig.from_pretrained(self.model_name, **kwargs)
        self.config.torch_dtype = torch_dtype or "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            device_map={"": PartialState().process_index},
            **kwargs,
        )

        self.model.config.use_cache = False
        logger.info(f"Loaded model: {self.model}")

        if lora:
            self.model = get_peft_model(self.model, LoraConfig(**lora))
            logger.info(f"Initialized PEFT based on {lora}")
