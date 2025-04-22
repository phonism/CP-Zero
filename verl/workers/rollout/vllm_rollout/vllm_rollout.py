# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import json
import re
import subprocess

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
import hashlib
import time
import os
import shutil


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def delete_directory(path):
    try:
        shutil.rmtree(path)
        print(f"Directory '{path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {e}")



class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        debug_str = []

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            #setattr(self.sampling_params, "stop_token_ids", [151645, 151643, 151658])
            setattr(self.sampling_params, "stop_token_ids", [151645, 151643])
            setattr(self.sampling_params, "stop", "</execute_command>")
            setattr(self.sampling_params, "include_stop_str_in_output", True)
            setattr(self.sampling_params, "detokenize", True)
            debug_str.append("[LQ_DEBUG]: " + str(self.sampling_params))
            debug_str.append("[LQ_DEBUG]: do_sample=" + str(do_sample))
            debug_str.append("[LQ_DEBUG]: batch_size="+ str(batch_size))
            print(self.sampling_params)
            all_outputs = []
            response_bash_result_attention_mask_list = []
            for i in range(batch_size):
                nums = self.config.n
                if not do_sample:
                    nums = 1
                for j in range(nums):
                    prompt_token_ids = idx_list[i]
                    output_single = [[], [], []]
                    max_length = getattr(self.sampling_params, "max_tokens")
                    all_response_str = ""
                    file_path_list = []
                    while True:
                        setattr(self.sampling_params, "n", 1)
                        generated_ids = self.inference_engine.generate(
                                prompts=None,
                                sampling_params=self.sampling_params,
                                prompt_token_ids=[prompt_token_ids],
                                use_tqdm=False
                        )
                        output_single[0] += generated_ids[0][0].tolist()
                        output_single[1] += generated_ids[1][0].tolist()
                        output_single[2] += [1] * len(generated_ids[0][0].tolist())
                        stop_reason = generated_ids[2][0]
                        #debug_str.append("[LQ_DEBUG]: generated_ids=" + str(generated_ids))
                        if generated_ids[0][0][-1] == 151645 or generated_ids[0][0][-1] == 151643:
                            all_outputs.append([
                                output_single[0][:min(max_length, len(output_single[0]))],
                                output_single[1][:min(max_length, len(output_single[1]))],
                                output_single[2][:min(max_length, len(output_single[2]))],
                            ])
                            debug_str.append("[LQ_DEBUG]: " + str(len(output_single[0])))
                            break
                        if stop_reason == "</execute_command>":
                            BASH_RESULT_START_TAG = "<|im_end|>\n<|im_start|>user\n<command_result>\n"
                            BASH_RESULT_END_TAG = "\n</command_result>\n<|im_end|>\n<|im_start|>assistant\n"
                            def extract_tool_call_content(text):
                                match = re.search(r'<execute_command>(.*?)</execute_command>', text, re.DOTALL)
                                return match.group(1) if match else None
                            def extract_command(text):
                                matches = re.findall(r'<command>(.*?)</command>', text, re.DOTALL)
                                return matches if matches else None
                            response_str = self.tokenizer.decode(generated_ids[0][0])
                            all_response_str += response_str
                            tc_content = extract_tool_call_content(response_str)
                            if tc_content is None:
                                prompt_token_ids = prompt_token_ids + generated_ids[0][0].tolist()
                                tc_result = f"{BASH_RESULT_START_TAG}The extraction command encountered an error. Please verify the command syntax.{BASH_RESULT_END_TAG}"
                            else:
                                debug_str.append("[LQ_DEBUG]: tool_call_content=" + tc_content)
                                tc_result = ""
                                try:
                                    command_list = extract_command(tc_content)
                                except:
                                    tc_result = BASH_RESULT_START_TAG + "extract command fail!" + BASH_RESULT_END_TAG
                                if tc_result == "" and command_list is not None:
                                    for command in command_list:
                                        debug_str.append("[LQ_DEBUG]: command=" + str(command))
                                        stdout, stderr = "no stdout", "no stderr"
                                        def extract_cpp_code_blocks(content):
                                            #pattern = re.compile(r'```cpp\n(.*?)```\n', re.DOTALL)
                                            pattern = re.compile(r'```(?:cpp|c\+\+)\n(.*?)```\n', re.DOTALL)
                                            cpp_code_blocks = pattern.findall(content) 
                                            return cpp_code_blocks
                                        code_string_list = extract_cpp_code_blocks(all_response_str)
                                        if len(code_string_list) == 0:
                                            tc_result = BASH_RESULT_START_TAG + "did not find cpp code" + BASH_RESULT_END_TAG
                                        else:
                                            code_string = code_string_list[-1]
                                            debug_str.append("[LQ_DEBUG]: code_string=" + code_string)
                                            try:
                                                md5_hash = hashlib.md5()
                                                #md5_hash.update(code_string.encode("utf-8") + str(time.time()).encode("utf-8"))
                                                md5_hash.update(code_string.encode("utf-8")) 
                                                file_path = "/root/workspace/CP-Zero/tmp/" + md5_hash.hexdigest()
                                                file_path_list.append(file_path)
                                                os.makedirs(file_path, exist_ok=True)
                                                file_name = file_path + "/main.cpp"
                                                f = open(file_name, "w+")
                                                f.write(code_string)
                                                f.close()
                                                try:
                                                    result = subprocess.run(
                                                            "cd " + file_path + " && " + command, 
                                                            text=True, shell=True, capture_output=True, timeout=5)
                                                    stdout = result.stdout
                                                    if result.stderr:
                                                        stderr = result.stderr
                                                except subprocess.TimeoutExpired as e:
                                                    stdout = ""
                                                    stderr = "Command timed out after 5 seconds!"
                                            except Exception as e:
                                                stderr = "execute shell command fail"
                                                print(f"An error occurred: {e}")
                                            debug_str.append("[LQ_DEBUG]: stdout=" + str(stdout) + "stderr=" + str(stderr))
                                            tc_result = f"{BASH_RESULT_START_TAG}after run the bash, the bash stdout is: {stdout}, stderr is: {stderr}{BASH_RESULT_END_TAG}"
                            debug_str.append("[LQ_DEBUG]: tool_call_result=" + str(tc_result))
                            tc_result_ids = self.tokenizer.encode(tc_result, add_special_tokens=False)
                            output_single[0] += tc_result_ids
                            output_single[1] += [0.0] * len(tc_result_ids)
                            output_single[2] += [0] * len(tc_result_ids)
                            prompt_token_ids = prompt_token_ids + generated_ids[0][0].tolist() + tc_result_ids
                            if len(prompt_token_ids) > max_length:
                                all_outputs.append([
                                    output_single[0][:min(max_length, len(output_single[0]))],
                                    output_single[1][:min(max_length, len(output_single[1]))],
                                    output_single[2][:min(max_length, len(output_single[2]))],
                                ])
                                break
                            continue
                        all_outputs.append([
                            output_single[0][:min(max_length, len(output_single[0]))],
                            output_single[1][:min(max_length, len(output_single[1]))],
                            output_single[2][:min(max_length, len(output_single[2]))],
                        ])
                        break
                    #for file_path in file_path_list:
                        #delete_directory(file_path)
                    #setattr(self.sampling_params, "max_tokens", max_length)


            #print("\n".join(debug_str))
    
            # Calculate the max length for padding
            max_length_responses = max(len(o[0]) for o in all_outputs)
            max_length_log_probs = max(len(o[1]) for o in all_outputs)
    
            # Pad response_ids and log_probs and convert them back to tensors
            padded_response_ids = torch.tensor([
                    o[0] + [151643] * (max_length_responses - len(o[0]))
                    for o in all_outputs
            ])
            padded_log_probs = torch.tensor([
                    o[1] + [0.0] * (max_length_log_probs - len(o[1]))
                    for o in all_outputs
            ])
            padded_response_attention_mask = torch.tensor([
                    o[2] + [0] * (max_length_log_probs - len(o[2]))
                    for o in all_outputs
            ])
    
            # Combine the padded response_ids and log_probs into a list of tuples of tensors
            output = [padded_response_ids, padded_log_probs, padded_response_attention_mask]

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        res_mask = output[2].to(idx.device)
        #log_probs = output[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            res_mask = pad_sequence_to_length(res_mask, self.config.response_length, 0)
            #log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(
                #response_id=response, eos_token=eos_token_id, 
                response_id=response, eos_token=[151643], 
                dtype=attention_mask.dtype)
        # TO BE FIX,这里会有些问题,包括可能需要多个token,以及有可能前后不一样的话,切出来的token也不一样
        response_bash_result_attention_mask = res_mask.to(attention_mask.dtype)
        print("DEBUG:", response_bash_result_attention_mask.shape, response_attention_mask.shape)
        bash_result_attention_mask = torch.cat((attention_mask, response_bash_result_attention_mask), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                #'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                "bash_result_attention_mask": bash_result_attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
