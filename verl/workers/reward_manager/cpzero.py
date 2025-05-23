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

from verl import DataProto
import torch
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from verl.utils.reward_score import _default_compute_score
import torch


class CpzeroRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        lock = threading.Lock()
        print_lock = threading.Lock()

        def process_item(i, data_item):
            nonlocal already_print_data_sources, reward_tensor

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
    
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
    
            score, compute_score_debug_str = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            with print_lock:
                print(compute_score_debug_str)
    
            reward_tensor[i, valid_response_length - 1] = score

            with lock:
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                current_count = already_print_data_sources[data_source]
    
                if current_count < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    need_print = True
                else:
                    need_print = False
    
            if need_print:
                with print_lock:
                    print(sequences_str)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_item, i, data[i]) for i in range(len(data))]
            for future in futures:
                future.result()
        return reward_tensor
