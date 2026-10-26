import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

from rich.console import Group
from rich.text import Text
from smolagents import (
    ActionStep,
    AgentError,
    AgentGenerationError,
    ChatMessage,
    CodeAgent,
    FinalAnswerStep,
    MessageRole,
    Model,
    PlanningStep,
    Tool,
    VLLMModel,
    handle_agent_output_types,
    truncate_content,
)
from smolagents.memory import MemoryStep
from smolagents.monitoring import Timing
from transformers.generation import StoppingCriteriaList, StopStringCriteria
from vllm import SamplingParams

from deep_math.math_executor import (
    RestrictedMathExecutor,
    extract_math_code,
    validate_model_output,
)


class LogLevel(IntEnum):
    OFF = -1  # No output
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


YELLOW_HEX = "#d4b702"


def substring_stopping_criteria(
    self, stop_sequences: list[str], tokenizer
) -> "StoppingCriteriaList":
    return StoppingCriteriaList([StopStringCriteria(tokenizer, stop_sequences)])


class VLLMCustom(VLLMModel):
    "My vLLMModel class with custom Sampling parameters."

    def __init__(
        self,
        model_id,
        model_kwargs: dict[str, Any] | None = None,
        lora_request=None,
        **kwargs,
    ):
        self.lora_request = lora_request
        self.last_output_token_count = 0

        super().__init__(model_id=model_id, model_kwargs=model_kwargs, **kwargs)

    def generate(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list[Tool] | None = None,
        sampling_params: SamplingParams | None = None,
        **kwargs,
    ):
        """Custom generate function for smolagents vLLM wrapper."""

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages[1:],
            flatten_messages_as_text=(not self._is_vlm),
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )

        messages = [messages[0]] + completion_kwargs.pop("messages")
        prepared_stop_sequences = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        sampling_params.stop = prepared_stop_sequences
        sampling_params.include_stop_str_in_output = True
        assert sampling_params.n == 1, "Agent working on a single trace at a time"

        assert tools_to_call_from is None
        assert isinstance(messages[0], str), (
            "We assume the task is already chat-formatted, possibly including system instruction"
        )

        # NOTE: we shouldn't add generation prompt here because these are sub-steps in the reasoning, same role
        prompt = messages[0]
        if len(messages) > 1:
            rest = self.tokenizer.apply_chat_template(
                messages[1:],
                tokenize=False,
            )
            prompt = prompt + "\n" + rest

        prompt_tokens = self.tokenizer.tokenize(prompt)
        prompt_token_count = len(prompt_tokens)

        if prompt_token_count > 130_000:
            print(f"############ PROMPT WAS LONGER THAN 130K: {prompt_token_count} tokens")
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I don't know.",
                raw={"out": "I don't know.", "completion_kwargs": completion_kwargs},
            )

        print("#### Agent step sampling params:", sampling_params)

        out = self.model.generate(
            prompt,
            sampling_params=sampling_params,
            lora_request=(self.lora_request if self.lora_request else None,),
        )

        output_text = out[0].outputs[0].text
        self.last_input_token_count = len(out[0].prompt_token_ids)
        self.last_output_token_count = len(out[0].outputs[0].token_ids)

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={"out": output_text, "completion_kwargs": completion_kwargs},
        )


@dataclass
class QueryWithInstructionsStep(MemoryStep):
    """
    Includes the prompt, chat formatted, with a possible system instruction
    """

    prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[str]:
        return [self.prompt]


class MathAgent(CodeAgent):
    """
    A math agent that uses Python code as an integral part of its reasoning.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        # prompt_templates: PromptTemplates | None = None,
        max_steps: int = 10,
        max_agent_output: Optional[int] = None,
        # system_prompt: str = "",
        grammar: dict[str, str] | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        code_execution_timeout: int = 20,  # Timeout in seconds for Python code execution
        additional_authorized_imports: list[str] | None = None,
        **kwargs,
    ):
        # NOTE!: we now assume the "task" is a complete processed chat template, maybe including system instruction

        if tools or additional_authorized_imports or kwargs.get("managed_agents"):
            raise ValueError("MathAgent uses a fixed math policy; extra tools/imports are disabled")
        if any(key in kwargs for key in ("executor", "executor_type", "executor_kwargs")):
            raise ValueError("MathAgent's restricted executor cannot be overridden")
        if kwargs.get("add_base_tools"):
            raise ValueError("Additional base tools are disabled for MathAgent")
        self.code_execution_timeout = code_execution_timeout
        self.grammar = grammar

        super().__init__(
            tools=tools,
            model=model,
            max_steps=max_steps,
            planning_interval=planning_interval,
            stream_outputs=stream_outputs,
            additional_authorized_imports=[],
            **kwargs,
        )
        self.authorized_imports = ["sympy", "numpy"]

        self._max_agent_output = max_agent_output
        self._output_tokens = 0  # we'll manage total output tokens per run as a global constraint instead of #steps x vllm_step_max_length

    def create_python_executor(self):
        return RestrictedMathExecutor(self.code_execution_timeout)

    def initialize_system_prompt(self) -> str:
        return ""

    def run_batch(
        self, prompts: list[str], rank: int = -1, sampling_params: SamplingParams = None
    ) -> list[str]:
        """
        Runs the agent on a batch of prompts.

        Args:
            prompts (list[str]): List of prompt strings.

        Returns:
            list[Any]: List of results from running each prompt.
        """
        answers = []
        for i, prompt in enumerate(prompts):
            print(f"*********************** RANK {rank}, PROMPT {i}")
            answers.append(str(self.run(prompt, sampling_params=sampling_params)))

        return answers

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
        sampling_params: SamplingParams = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform, assuming chat formatted, with possible system instruction.
            stream (`bool`): Whether to run in streaming mode.
                If `True`, returns a generator that yields each step as it is executed. You must iterate over this generator to process the individual steps (e.g., using a for loop or `next()`).
                If `False`, executes all steps internally and returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Nonempty injected arguments are prohibited by the math execution policy.
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.
        """
        # Conversation memory may be retained, but execution state never crosses tasks.
        self.python_executor.reset()
        if additional_args:
            raise ValueError("Host objects cannot be injected into restricted math execution")
        max_steps = max_steps or self.max_steps
        self.task = task
        self.sampling_params = (
            sampling_params.clone() if sampling_params else None
        )  # we assume we mainly work with vLLM-based model
        self._output_tokens = 0  # reset output tokens for this run
        self._typical_step = self.sampling_params.max_tokens if self.sampling_params else 512

        self.interrupt_switch = False

        self.memory.system_prompt = QueryWithInstructionsStep(prompt=self.task)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=truncate_content(self.task.strip(), max_length=500),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )

        steps = self._run_with_math_cleanup(task=self.task, max_steps=max_steps, images=images)
        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return steps
        # Outputs are returned only at the end. We only look at the last step.
        return list(steps)[-1].output

    def _run_with_math_cleanup(self, **kwargs):
        try:
            yield from self._run_stream(**kwargs)
        finally:
            self.python_executor.close()

    def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep]:
        final_answer = None
        self.step_number = 1
        while final_answer is None and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)
            step_start_time = time.time()
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                for element in self._generate_planning_step(
                    task, is_first_step=(self.step_number == 1), step=self.step_number
                ):
                    yield element
                self.memory.steps.append(element)
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=step_start_time),
                observations_images=images,
            )
            try:
                for el in self._execute_step(action_step):
                    final_answer = el
                    yield el
            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
                action_step.is_final_answer = False
                final_answer = None
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                self.step_number += 1
            yield action_step

        yield FinalAnswerStep(handle_agent_output_types(self.collect_all_outputs()))

    def collect_all_outputs(self) -> str:
        """
        Collect all output from the agent's memory and return it as a string.
        """
        all_output = ""
        for step in self.memory.steps:
            if isinstance(step, ActionStep):
                if step.model_output is not None:
                    all_output += str(step.model_output) + "\n"
                if step.observations is not None:
                    all_output += str(step.observations) + "\n"
        return all_output

    def _execute_step(self, memory_step: ActionStep) -> Generator[Any]:
        self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
        final_answer = None
        for el in self._step_stream(memory_step):
            final_answer = el
            yield el
        if final_answer is not None and self.final_answer_checks:
            self._validate_final_answer(final_answer)
        yield final_answer

    def _step_stream(self, memory_step: ActionStep) -> Generator[Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        ### Generate model output ###
        memory_step.model_input_messages = input_messages

        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            assert not self.stream_outputs, (
                "Task prompt special handling is implemented only in `model.generate`"
                " In other words, streaming is not supported."
            )

            if self.stream_outputs:
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=[
                        "```\n",
                        "</tool_call>",
                        "<end_code>",
                        "Observation:",
                        "Calling tools:",
                    ],
                    **additional_args,
                )
                output_text = ""
                # with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                for event in output_stream:
                    if event.content is not None:
                        output_text += event.content
                        # live.update(Markdown(output_text))
                    # yield event

                self.logger.log_markdown(
                    content=output_text,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )
                validate_model_output(output_text)
                model_output = output_text

                chat_message = ChatMessage(role="assistant", content=model_output)
                memory_step.model_output_message = chat_message
                model_output = chat_message.content
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=[
                        "```\n",
                        "</tool_call>",
                        "<end_code>",
                        "Observation:",
                        "Calling tools:",
                    ],
                    sampling_params=self.sampling_params,
                    **additional_args,
                )
                validate_model_output(chat_message.content)
                memory_step.model_output_message = chat_message
                model_output = chat_message.content
                self.logger.log_markdown(
                    content=model_output,
                    title="Output message of the LLM:",
                    level=10,
                )

            memory_step.model_output = model_output
            self._output_tokens += self.model.last_output_token_count

        except Exception as e:
            print(f"########################## Error in generating model output: {e}")
            chat_message = ChatMessage(role="assistant", content="I don't know.")
            memory_step.model_output_message = chat_message
            model_output = chat_message.content
            memory_step.model_output = model_output

        ### Parse output ###

        ## NOTE: If \boxed{} in answer, this is the final answer/step, we return it and yield all the way up
        if model_output and "\\boxed{" in model_output:
            self.logger.log_markdown(
                content=model_output[-500:],
                title="BOXED was found in output, this is the final answer.",
                level=LogLevel.INFO,
            )
            yield model_output
            return

        # NOTE: We examine our token budget and adjust the max_tokens parameter accordingly.
        # We also use `we_done` to make it the last step if we are out of tokens
        # NOTE: notice we EITHER use the `max_agent_output` or the `steps` x `vllm_step_max_length` as the constraint.
        out_of_tokens = None
        if self._max_agent_output is not None:
            self.sampling_params.max_tokens = min(
                max(self._max_agent_output - self._output_tokens, 1), self._typical_step
            )

            self.logger.log(
                f"{self.sampling_params.max_tokens} tokens are left", level=LogLevel.INFO
            )
            if self.sampling_params.max_tokens < 50:
                out_of_tokens = True

        try:
            code_action = extract_math_code(model_output)
        except Exception as e:
            self.python_executor.reset()
            memory_step.observations = f"Exception: {e} (task state discarded)"
            yield out_of_tokens
            return
        if code_action is None:
            self.logger.log("NO CODE REQUESTS THIS STEP", level=LogLevel.INFO)
            yield out_of_tokens
            return
        memory_step.code_action = code_action

        ### Execute action ###
        self.logger.log_code(
            title="Executing parsed code:", content=code_action, level=LogLevel.INFO
        )
        is_final_answer = False
        try:
            execution_outputs_console = []
            result = self.python_executor(code_action)
            output, execution_logs, is_final_answer = (
                result.output,
                result.logs,
                result.is_final_answer,
            )
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            # NOTE: I don't want to introduce EXECUTION LOGS, the model doesn't see it in the fewshot examples

        except Exception as e:
            output = ""
            execution_logs = f"Exception: {e}"

        truncated_output = truncate_content(str(output))
        execution_logs = truncate_content(execution_logs, max_length=500)

        observation = "```output\n" + execution_logs + "\n\n" + truncated_output + "\n```\n"
        memory_step.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        memory_step.is_final_answer = is_final_answer

        yield output if is_final_answer else out_of_tokens
