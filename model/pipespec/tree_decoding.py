import sys 
import os
import time

from pipeline_spec.src.utils.config import TRIM_LEVEL, ASYNC_STEP
from pipeline_spec.src.utils.streaming_utils import trim_ids
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline_spec.src.rpc.utils import from_dict_to_request_response, from_request_to_dict 
import torch 
from transformers import GenerationConfig, StoppingCriteriaList, LogitsProcessorList
from transformers.generation.candidate_generator import CandidateGenerator
from typing import Optional, Union
import asyncio
# from loguru import logger
from pipeline_spec.src.utils.debug_logger import my_logger
from pipeline_spec.src.utils.timer import Timer
from pipeline_spec.src.model.generation_mixin import longest_common_prefix_index
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GreedySearchOutput, ModelOutput, GenerationConfig, GenerateNonBeamOutput
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decider generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None

async def async_target_generate(self, candidate_input_ids, candidate_logits, candidate_logits_index, candidate_length, cur_len, input_ids, is_done_candidate, candidate_generator=None, streamer=None, synced_gpus=None, order=0, **kwargs):
        # 确保在当前事件循环中设置异步 channel
        with Timer("prepare_request_time"):
            trim_level = TRIM_LEVEL
            input_ids, candidate_input_ids = trim_ids(input_ids, candidate_input_ids, candidate_length, trim_level=trim_level, cur_len=cur_len)
            # logger.debug(f"{candidate_input_ids.shape=} {input_ids.shape=}")
            branch_parent_idxs = kwargs.pop('branch_parent_idxs', None)
            branch_position_idxs = kwargs.pop('branch_position_idxs', None) 
            
            # input_ids, candidate_input_ids = prepare_tree_request(input_ids, candidate_input_ids, all_parent_idxs, all_position_idxs, candidate_length, trim_level)
            my_logger.info(f"debug vice target_generate after trim {order=} {input_ids.shape=} {candidate_input_ids.shape=}")
            # logger.debug(f"{input_ids=}")
            # logger.debug(f"{candidate_input_ids=}")
            self._ensure_async_setup()
            request = from_dict_to_request_response(
                name="target_generate", 
                candidate_input_ids=candidate_input_ids, 
                candidate_logits=candidate_logits, 
                candidate_logits_index=candidate_logits_index,
                candidate_length=candidate_length, 
                cur_len=cur_len, 
                input_ids=input_ids, 
                is_done_candidate=is_done_candidate,
                order=order,
                branch_parent_idxs=branch_parent_idxs,
                branch_position_idxs=branch_position_idxs
                )     
        return_response = {"this_peer_finished": False, "input_ids": torch.tensor([]), "accepted_length": 0}  
        try:
            # 使用异步调用
            my_logger.info(f"debug sending to target {order=}")
            response = await self.async_stub.target_generate(request)
            with Timer("parse_response_time"):
                response = from_request_to_dict(response)
            # response = from_request_to_dict(response)
            return_response.update(response)
            return return_response['input_ids'], return_response['this_peer_finished'], return_response['accepted_length']
        except Exception as e:
            print(f"async grpc error: {e}")
            raise
        
def target_generate(self, candidate_input_ids, candidate_logits, candidate_logits_index, candidate_length, cur_len, input_ids, is_done_candidate, candidate_generator=None, streamer=None, synced_gpus=None, order=0, trim_level=None):
        if trim_level is None: 
            trim_level = TRIM_LEVEL
        stub = self.stub
        # lists = ['input_ids', 'attention_mask']
        input_ids, candidate_input_ids = trim_ids(input_ids, candidate_input_ids, candidate_length, trim_level=trim_level)
        request = from_dict_to_request_response(
            name="target_generate", 
            candidate_input_ids=candidate_input_ids, 
            candidate_logits=candidate_logits, 
            candidate_logits_index=candidate_logits_index,
            candidate_length=candidate_length, 
            cur_len=cur_len, 
            input_ids=input_ids, 
            is_done_candidate=is_done_candidate, order=order)     
        return_response = {"this_peer_finished": False, "input_ids": torch.tensor([]), "accepted_length": 0}  
        response = stub.target_generate(request)
        response = from_request_to_dict(response)
        return_response.update(response)
        return return_response['input_ids'], return_response['this_peer_finished'], return_response['accepted_length']
       
async def async_tree_spec_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: CandidateGenerator,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        do_sample = generation_config.do_sample
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # 存储所有异步任务的Future对象
        pending_generations = []
        order = 0
        async_reqs = []
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.target_model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        

        this_peer_finished = False
        is_first_iteration = True  # to preserve the same API in the output as other generation methods
        has_started = False
        async_step = ASYNC_STEP
        last_verified_idx = 0
        updated_by_target = False   
        verified_by_target = False
        skip_future_generate = False
        while self.draft_model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # 检查已完成的异步任务
            s = time.perf_counter()
            if len(pending_generations) > 0:
                done_generations = []
                still_pending = []
                for task in pending_generations:
                    if task.done():
                        done_generations.append(task)
                    else:
                        still_pending.append(task)
                pending_generations = still_pending
                
                
                
                # 按顺序处理已完成的任务
                done_generations = sorted(done_generations, key=lambda x: x.order)
                # 处理已完成的生成结果
                if len(done_generations) == 0: 
                    # my_logger.info(f"debug no done generations")
                    await asyncio.sleep(0)
                for done_future in done_generations:
                    # try:
                    # 获取异步任务的结果
                    
                    result_input_ids, result_finished, accepted_length = await done_future
                    this_peer_finished = result_finished
                    my_logger.info(f"done verification order={done_future.order}")
                    if len(result_input_ids) == 0: 
                        # logger.debug(f"throw this result order={done_future.order}")
                        continue # 不返回任何东西说明这次的结果需要丢弃  
                    else: 
                        # logger.debug(f"result_input_ids {result_input_ids.shape=} {result_input_ids[0]=} {accepted_length=}")
                        # extract tree from tree state 
                        tree_state = self.draft_model.tree_state
                        sequences, paths = _extract_sequences_from_tree_branch(input_ids[0], tree_state.parent_idxs[0], range(tree_state.last_tree_layer_node_idx, input_ids.shape[1]), tree_state.branch_start_idx)
                        id_len = result_input_ids.shape[1]
                        result_path = result_input_ids[0, id_len // 2 + 1:]
                        result_input_ids = result_input_ids[:, :id_len // 2 + 1].to(input_ids.device) 
                        # # logger.debug(f"{result_path=} {result_input_ids[0]=}")
                        if last_verified_idx >= len(result_input_ids[0]): 
                            continue
                        largest_matching_sequence = None
                        largest_common_index = -1
                        for sequence in sequences:
                            common_index = longest_common_prefix_index(result_input_ids[0], sequence) #
                            if common_index >= largest_common_index: 
                                largest_common_index = common_index
                                largest_matching_sequence = sequence
                            
                        this_peer_finished = result_finished
                        last_verified_idx = len(result_input_ids[0])
                        verified_by_target = True
                        my_logger.info(f"verified_by_target {done_future.order=} {this_peer_finished=}")
                        if (not this_peer_finished) and largest_common_index == len(result_input_ids[0]): 
                            continue # 说明 draft model 正确地生成了已验证的 token + 额外的 token，并且已经传上服务器了，就不需要管
                            # TODO:对于 tree 来说, 就需要管, 因为有很多无用的 token 需要被裁剪掉, 否则下次上传就会传很多垃圾 token
                        else: 
                            my_logger.info(f"largest_common_index {largest_common_index=} {len(result_input_ids[0])=} {len(input_ids[0])=}")
                        my_logger.info(f"updated by target {done_future.order=}")
                        updated_by_target = True
                        if not this_peer_finished:
                            input_ids, candidate_generator = self.draft_model.update_tree_state_from_target(result_input_ids, result_path, candidate_generator, device=input_ids.device)
                        else: 
                            input_ids = result_input_ids
                if self.draft_model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device) == False: 
                    break 
            # barrier to avoid generate overwhelming drafts
            if skip_future_generate and not updated_by_target:
                continue
            if (
                (last_verified_idx != 0 and len(input_ids[0]) + (async_step // 2) > last_verified_idx + async_step) and not verified_by_target): 
                # my_logger.info(f"debug {last_verified_idx=} {len(input_ids[0])=} {async_step=}")
                continue
            skip_future_generate = False
            is_async_req = ((not verified_by_target) or (len(input_ids[0]) != last_verified_idx)) and (order != 0 and order != 1)
            my_logger.info(f"current {order=} {verified_by_target=} {updated_by_target=} {len(input_ids[0])=} {last_verified_idx=} {is_async_req=}")
            if is_async_req:
                async_reqs.append(order)
            # logger.debug(f"before tree draft input ids {input_ids.shape=} {input_ids[0]=}")
            with Timer(f"draft_generate order={order}" if not is_async_req else f"draft_generate_async order={order}"):
                cur_len, candidate_input_ids, candidate_logits, candidate_logits_index, candidate_length, is_done_candidate, branch_parent_idxs, branch_position_idxs = self.draft_model.draft_tree_generate(input_ids, candidate_generator, stopping_criteria)
            # barrier to avoid sending overwhelming requests
            # if order > 0 and (
            #     (not verified_by_target) or 
            #     (not updated_by_target)
            # ) and (is_done_candidate == True): 
            #     my_logger.info(f"skip future generation for input {input_ids.shape=}")
            #     skip_future_generate = True
                # continue
            
            updated_by_target = False 
            verified_by_target = False
            if do_sample == False: 
                candidate_logits = None 
                candidate_logits_index = None
            # 创建异步任务并添加到pending列表
            coro = self.target_model.async_target_generate(
                candidate_input_ids=candidate_input_ids.detach().clone(), 
                candidate_logits=candidate_logits, 
                candidate_logits_index=candidate_logits_index,
                candidate_length=candidate_length, 
                cur_len=cur_len, 
                input_ids=input_ids.detach().clone(), 
                is_done_candidate=is_done_candidate,
                synced_gpus=synced_gpus, 
                streamer=streamer,
                order=order,
                branch_parent_idxs=branch_parent_idxs,
                branch_position_idxs=branch_position_idxs
            )
            # 使用create_task创建Future对象
            task = asyncio.create_task(coro)
            task.order = order
            # logger.debug(f"sending task {order=}")
            order += 1
            if has_started == False: 
                has_started = True
                # logger.debug(f"starting task {order=} awaiting")
                result_input_ids, this_peer_finished, accepted_length = await task
                updated_by_target = True
                if self.use_tree:
                    id_len = result_input_ids.shape[1]
                    result_path = result_input_ids[0, id_len // 2 + 1:]
                    result_input_ids = result_input_ids[:, :id_len // 2 + 1].to(input_ids.device)
                    # # logger.debug(f"result from target {result_path=} {result_input_ids[0]=}")
                    input_ids, candidate_generator = self.draft_model.update_tree_state_from_target(result_input_ids, result_path, candidate_generator, device=input_ids.device)
                else: 
                    input_ids = result_input_ids
                # logger.debug(f"{input_ids=}")
                last_verified_idx = len(input_ids[0])
                # 第一次必须要等待，然后才能开始异步。
            else: 
                pending_generations.append(task)
                # 使用临时结果继续循环
                input_ids = candidate_input_ids  # 使用候选序列作为临时结果
                this_peer_finished = False  # 假设未完成
                await asyncio.sleep(0)
        pending_generations = [] # clear the pending generations
        if streamer is not None:
            streamer.end()
        my_logger.info(f"debug {len(async_reqs)=} {async_reqs=}")
        if (
            hasattr(candidate_generator, "assistant_model")
            and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
        ):
            candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
                candidate_generator.num_assistant_tokens
            )
        # currently return None values for these states 
        scores = None 
        raw_logits = None
        decoder_attentions = None
        cross_attentions = None
        decoder_hidden_states = None
        if return_dict_in_generate:
            if self.target_model.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
