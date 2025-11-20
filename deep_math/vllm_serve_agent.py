# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection, wait
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors
from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_ascend_available,
    is_vllm_available,
)

if is_fastapi_available():
    from fastapi import FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logger's level


# NOTE: We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["NCCL_CUMEM_ENABLE"] = "0"


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    # The following attributes are initialized when `init_communicator` method is called.
    # pynccl_comm = None  # Communicator for weight updates
    # client_rank = None  # Source rank for broadcasting updated weights

    def update_model_params(self, path: str):
        state_dict = torch.load(path)
        weights = [(name, tensor) for name, tensor in state_dict.items()]
        self.model_runner.model.load_weights(weights=weights)
        return True

    def update_named_param(self, name: str, path: str):
        weights = load_safetensors(path)["tensor"]
        self.model_runner.model.load_weights(weights=[(name, weights)])
        return True


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Revision to use for the model. If not specified, the default branch will be used."
        },
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )
    max_steps: int = field(
        default=20,
        metadata={"help": "Maximum number of steps for the MathAgent reasoning loop."},
    )
    max_agent_output: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens the MathAgent can output for a run"},
    )


def llm_math_worker(
    script_args: ScriptArguments,
    data_parallel_rank: int,
    master_ip: str,
    master_port: int,
    connection: Connection,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data_parallel_rank)
    os.environ["VLLM_DP_MASTER_IP"] = str(master_ip)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"

    from vllm.config import CompilationConfig

    from .agent import MathAgent, VLLMCustom

    llm = VLLMCustom(
        model_id=script_args.model,
        lora_request=None,
        model_kwargs=dict(
            tensor_parallel_size=script_args.tensor_parallel_size,
            revision=script_args.revision,
            gpu_memory_utilization=script_args.gpu_memory_utilization,
            dtype=script_args.dtype,
            # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
            # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
            # This is particularly useful here because we generate completions from the same prompts.
            enable_prefix_caching=script_args.enable_prefix_caching,
            max_model_len=script_args.max_model_len,
            compilation_config=CompilationConfig(
                cache_dir=tempfile.mkdtemp(dir="/tmp/", prefix="vllm_cache_"),
            ),
            # enforce_eager=True,  # DEBUG
            # disable_custom_all_reduce=True,  # DEBUG
            worker_extension_cls="deep_math.vllm_serve_agent.WeightSyncWorkerExtension",  # NOTE: we disable NCCL-based communications
        ),
    )

    agent = MathAgent(
        tools=[],
        model=llm,
        max_steps=script_args.max_steps,
        max_agent_output=script_args.max_agent_output,
        code_execution_timeout=20,
        stream_outputs=False,
        verbosity_level=2,  # DEBUG
    )

    connection.send({"status": "ready"})

    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            break

        match command.get("type"):
            case "call" | "fire_and_forget":
                method_name = command["method"]
                args, kwargs = command.get("args", ()), command.get("kwargs", {})

                match method_name:
                    case "generate":
                        try:
                            outputs = agent.run_batch(
                                *args,
                                rank=data_parallel_rank,
                                **kwargs,
                            )
                            result = llm.tokenizer(outputs).input_ids
                        except Exception as e:
                            print(f"Error running agent {data_parallel_rank}: {e}")
                            result = [40, 1513, 944, 1414, 13]

                    case "update_named_param":
                        logger.info(f"Updating weights tensor, rank {data_parallel_rank}")
                        result = llm.model.collective_rpc(
                            method="update_named_param", kwargs=kwargs
                        )

                    case "update_model_params":
                        logger.info(f"Updating full model params, rank {data_parallel_rank}")
                        result = llm.model.collective_rpc(
                            method="update_model_params", kwargs=kwargs
                        )

                    case "collective_rpc":
                        logger.info("RPC calls are disabled.")

                    case _:
                        method = getattr(llm.model, method_name)
                        result = method(*args, **kwargs)

                if command["type"] == "call":
                    try:
                        connection.send(result)
                    except Exception as e:
                        print(f"Error sending result in conn {data_parallel_rank}: {e}")

            case "shutdown":
                break

            case _:
                print(f"Unknown command type: {command.get('type')}")


def chunk_list(lst: list, n: int) -> list[list]:
    """
    Split list `lst` into `n` evenly distributed sublists.

    Example:
        >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
        [[1, 2, 3], [4, 5, 6]]
        >>> chunk_list([1, 2, 3, 4, 5, 6], 4)
        [[1, 2], [3, 4], [5], [6]]
        >>> chunk_list([1, 2, 3, 4, 5, 6], 8)
        [[1], [2], [3], [4], [5], [6], [], []]
    """
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError(
            "vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`."
        )

    if is_vllm_available():
        from vllm import SamplingParams
        from vllm.utils import get_open_port

        if is_vllm_ascend_available():
            pass

    # Spawn dp workers, and setup pipes for communication
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        master_port = get_open_port()
        parent_connection, child_connection = Pipe()
        process = Process(
            target=llm_math_worker,
            args=(script_args, data_parallel_rank, script_args.host, master_port, child_connection),
        )
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()
        while len(ready_connections) < script_args.data_parallel_size:
            for connection in connections:
                if connection.poll(0.5):  # Check with timeout to avoid blocking
                    # msg = receive_chunked(connection)
                    msg = connection.recv()
                    if isinstance(msg, dict) and msg.get("status") == "ready":
                        ready_connections.add(connection)

        yield

        # Wait for processes to terminate
        for process in processes:
            # Wait for 10 seconds for the process to terminate
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(
                    f"Process {process} is still alive after 10 seconds, attempting to terminate..."
                )
                process.terminate()
                process.join()  # ensure process termination after calling terminate()

    app = FastAPI(lifespan=lifespan)

    app.state.run_hash = None

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """
        Retrieves the world size of the LLM engine, which is `tensor_parallel_size * data_parallel_size`.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

        Returns:
            `GenerateResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """

        assert len(connections) == script_args.data_parallel_size, (
            f"##############  Expected {script_args.data_parallel_size} connections, but got {len(connections)}."
        )

        sampling_params = SamplingParams(
            n=1,  # each agent works on a single trace
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
        )

        # Duplicating prompts; we use SamplingParams.n to represent the GRPO `num_generations`.
        r_prompts = [prompt for prompt in request.prompts for _ in range(request.n)]

        # Evenly distribute prompts across DP ranks
        chunked_prompts = chunk_list(r_prompts, script_args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Receive results
        outputs = [None] * len(connections)
        remaining = set(connections)
        while remaining:
            conns = wait(list(remaining))  # TODO: use relevant connections
            for conn in conns:
                idx = connections.index(conn)
                try:
                    outputs[idx] = conn.recv()
                except EOFError:
                    outputs[idx] = [[40, 1513, 944, 1414, 13] for _ in range(request.n)]
                except TimeoutError:
                    outputs[idx] = [[40, 1513, 944, 1414, 13] for _ in range(request.n)]
                except Exception as e:
                    outputs[idx] = [[40, 1513, 944, 1414, 13] for _ in range(request.n)]
                finally:
                    remaining.remove(conn)
        all_outputs = outputs

        assert len(connections) == script_args.data_parallel_size, (
            f"##############  Expected {script_args.data_parallel_size} connections, but got {len(connections)}."
        )

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]

        # Flatten and combine all results
        # from list of list to single list
        all_outputs = list(chain.from_iterable(all_outputs))

        return {"completion_ids": all_outputs}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int
        run_hash: str

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1

        app.state.run_hash = (
            request.run_hash
        )  # hash used for simple cache management for weights exchange

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {"method": "init_communicator", "args": (request.host, request.port, world_size)}
        for connection in connections:
            connection.send(
                {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
            )

        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        path: str

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function update_named_param is called this way: update_named_param("name", torch.float32, (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))

        kwargs = {"name": request.name, "path": request.path}

        for connection in connections:
            connection.send({"type": "call", "method": "update_named_param", "kwargs": kwargs})

        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {
            "message": f"Request received, updated named param: {request.name}. Success: {str(success)}"
        }

    class UpdateModelRequest(BaseModel):
        model_path: str

    # create an endpoint for update_model_params
    @app.post("/update_model_params/")
    async def update_model_params(request: UpdateModelRequest):
        kwargs = {"model_path": request.model_path}
        for connection in connections:
            connection.send({"type": "call", "method": "update_model_params", "kwargs": kwargs})
        # Wait for and collect all results
        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {"message": "Request received, update full model: " + str(success)}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})

        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)

        for pt_file in glob(f"/tmp/{app.state.run_hash}/weights*"):
            try:
                os.remove(pt_file)
            except Exception as e:
                print(f"### Failed to remove {pt_file}: {e}")

        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send(
                {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
            )
        return {"message": "Request received, closing communicator"}

    @app.get("/get_config/")
    async def get_config():
        """
        Returns the configuration of the vLLM server, including the script arguments used to start it.
        """
        return {"message": script_args.__dict__}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser(
            "vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments
        )
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    print(f"Running vLLM serve with arguments: {script_args}")
    main(script_args)
