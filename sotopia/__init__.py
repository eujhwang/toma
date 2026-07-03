# -*- coding: utf-8 -*-
import unsloth
from unsloth import FastLanguageModel
from peft import PeftModel

import os
import time
import logging
import collections
# from typing import Optional
from google import genai
from google.genai import types
import numpy as np
# import tqdm
import openai
import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, MistralForCausalLM, Gemma3ForConditionalGeneration
from peft import PeftModelForCausalLM, PeftModel

from utils.data_utils import DatasetLoader
from utils.models import OPEN_MODEL_HF, PROPRIETARY_LLM

from utils.sotopia_prompts import *
from utils.sotopia_eval_prompts import *
from utils.parse_actions import *

open_router_client = openai.OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ.get("OPEN_ROUTER_KEY", ""),
)
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_KEY", ""))


class LLMAgent:
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        scenario: str,
        agent_goal: str,
        agent_background: str,
        model,
        tokenizer,
        model_name: str,
        partner_model_name: str,
        is_partner: bool,
        train_type: str = "none",
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.scenario = scenario
        self.agent_goal = agent_goal
        self.agent_background = agent_background
        self.model_name = model_name
        self.model = model
        self.partner_model_name = partner_model_name
        self.tokenizer = tokenizer
        self.train_type = train_type
        self.template = prompt_utterance_without_ms
        self.is_partner = is_partner

        print(f"[model]: {self.model_name}, [partner_model]: {self.partner_model_name}")

        self.open_model_gen_cnt = 0
        self.gpt_call_cnt = 0

    @torch.inference_mode()
    def act(
        self,
        history: list[str],
        do_sample: bool = False,
        max_new: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.9,
        ms_text: str = "",
        strategy_text: str = "",
        cot_text: str = "",
        max_retries: int = 2,
    ) -> dict:
        # print("[ms_text]:", ms_text)
        # print("[strategy_text]:", strategy_text)

        if self.is_partner and self.model_name != self.partner_model_name:
            prompt = prompt_utterance_without_ms(
                history=history,
                speaker=self.agent_name,
                scenario=self.scenario,
                social_goal=self.agent_goal,
                turn_number=len(history)+1,
            )
        else:
            if ms_text:
                prompt = prompt_utterance_with_ms(
                    history=history,
                    speaker=self.agent_name,
                    scenario=self.scenario,
                    social_goal=self.agent_goal,
                    turn_number=len(history)+1,
                    ms_text=ms_text,
                )
            elif strategy_text:
                prompt = prompt_utterance_with_strategy(
                    history=history,
                    speaker=self.agent_name,
                    scenario=self.scenario,
                    social_goal=self.agent_goal,
                    turn_number=len(history)+1,
                    strategy_text=strategy_text,
                )
                # print("[prompt]:", prompt)
            elif cot_text:
                prompt = prompt_utterance_with_cot(
                    history=history,
                    speaker=self.agent_name,
                    scenario=self.scenario,
                    social_goal=self.agent_goal,
                    turn_number=len(history)+1,
                    cot_text=cot_text,
                )
            else:
                prompt = prompt_utterance_without_ms(
                    history=history,
                    speaker=self.agent_name,
                    scenario=self.scenario,
                    social_goal=self.agent_goal,
                    turn_number=len(history)+1,
                )
        chat = [{
            "role": "user",
            "content": prompt
        }]
        # A strict JSON formatting nudge we’ll add if we need to retry
        JSON_ONLY_NUDGE = (
            "Return ONLY a single valid JSON object with keys:\n"
            "  - \"action_type\" (one of: \"speak\", \"non-verbal communication\", \"action\", \"none\")\n"
            "  - \"argument\" (string)\n"
            "  - \"mental_state\" (optional string)\n"
            "No prose, no code fences, no comments, no additional text."
            "\nExample:\n"
            "{\"action_type\":\"speak\",\"argument\":\"...\",\"mental_state\":\"...\"}"
        )

        last_parsed = None
        last_decoded = None

        # Decode → parse → validate; retry if needed
        for attempt in range(max_retries + 1):
            # On retries, we can tighten sampling and add a JSON-only reminder
            if attempt > 0:
                # push a corrective “format only” instruction with the prior bad output
                chat.append({
                    "role": "user",
                    "content": (
                        f"{JSON_ONLY_NUDGE}\n\nYour previous output was invalid or required repair:\n"
                        f"{(last_decoded or '')[:2000]}"
                    )
                })

            # if isinstance(self.model, str):
            #     if self.model == "qwen2.5-72b" or self.model == "gpt-120b":
            #         if "qwen2.5-72b":
            #             model_path = "qwen/qwen3-235b-a22b-2507"
            #         elif "gpt-120b":
            #             model_path = "openai/gpt-oss-120b:free"
            #         completion = open_router_client.chat.completions.create(
            #         # extra_headers={
            #         #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            #         #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            #         # },
            #         extra_body={},
            #         model=model_path,
            #         messages=chat
            #         )
            #         decoded = completion.choices[0].message.content
            #         # print("[decoded]:", decoded)
            #         self.open_model_gen_cnt += 1
            #         time.sleep(2)
            #     else:
            #         assert self.model in PROPRIETARY_LLM, self.model
            #         decoded = SotopiaEvaluator.gpt_call(PROPRIETARY_LLM[self.model], chat, do_parse=False)
            #         self.gpt_call_cnt += 1
            # else:
            # assert SotopiaEvaluator.is_causal_lm(self.model)
            decoded = SotopiaEvaluator.open_model_gen(
                self.model, self.tokenizer, chat,
                do_sample=do_sample,
                temperature=temperature,
                max_new=max_new,
                top_p=top_p,
            )
            self.open_model_gen_cnt += 1
            last_decoded = decoded

            # Parse
            action = parse_action(decoded)
            last_parsed = action

            # Validate (strict means: no errors and not "repaired")
            if is_valid_action(action, strict=True):
                # logging.info(f"[action]: {action}")
                return action  # success

        # If we exhausted retries, return the best we could parse (perhaps a safe fallback)
        return last_parsed


class SotopiaEvaluator:

    def __init__(
            self,
            logger,
            cuda_dict: dict,
            seed: int = 42,
            verbose: bool = False,
            debug: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            output_dir: Optional[str] = None,
            eval_mode: str = "all",
            use_ms: bool = False,
            use_strategy: bool = False,
            use_cot: bool = False,
            gen_only: bool = False,
            eval_only: bool = False,
            eval_filepath: Optional[str] = None,
    ):
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.verbose = verbose
        self.debug = debug
        self.eval_mode = eval_mode
        self.use_ms = use_ms
        self.use_strategy = use_strategy
        self.use_cot = use_cot
        self.gen_only = gen_only  # Only generate and save conversations (no evaluation)
        self.eval_only = eval_only  # Only evaluate conversations
        self.eval_filepath = eval_filepath  # The filepath to the conversations to be evaluated

        assert os.path.isdir(project_root_dir), f"`>>> project_root_dir` is not a directory: {project_root_dir}"
        self.project_root_dir = project_root_dir
        if output_dir is None:
            output_dir = os.path.join(project_root_dir, "results/sotopia")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.cache_dir = cache_dir
        self.home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.home_dir, ".cache/huggingface")
            # self.cache_dir = os.path.join(self.project_root_dir, ".cache/huggingface/")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        if self.verbose:
            self.logger.info(f">>> cache_dir: {self.cache_dir}")

        # os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_HOME"] = self.cache_dir
        self.open_models = OPEN_MODEL_HF
        self.open_models_local = {
            k: (
                os.path.join(self.cache_dir, "models--" + "--".join(v.split("/")), "snapshots/model")
                if not os.path.exists("/model-weights/" + v.split("/")[-1])
                else "/model-weights/" + v.split("/")[-1]  # use unsloth model first # vector server model weights
            )
            for k, v in self.open_models.items()
        }  # Local model path after running `utils/download_hf_model.py`
        self.proprietary_models = PROPRIETARY_LLM
        # self.device_mps = torch.device("mps")  # MacOS M chip

        self.generator_model_name = None
        self.gen_partner_model_name = None
        self.evaluator_model_name = None

        # To avoid loading the models (and tokenizers) twice in `run_gen_eval`
        self.generator_model, self.generator_tokenizer = None, None
        self.gen_partner_model, self.gen_partner_tokenizer = None, None

        self.open_model_gen_cnt = 0
        self.gpt_call_cnt = 0

    def initialize_model(
            self,
            model_name: str,
            ckpt_path: Optional[str] = None,
            do_4bit: bool = False,
    ):
        print(f"[model_name]: {model_name}")
        if isinstance(ckpt_path, str) and os.path.isdir(ckpt_path):
            assert model_name in ckpt_path.lower()
            model_path = ckpt_path
        else:
            assert model_name in self.open_models_local, f">>> Unsupported `model_name`: {model_name}"
            model_path_local = self.open_models_local[model_name]
            if os.path.isdir(model_path_local):
                model_path = model_path_local
            else:
                assert model_name in self.open_models, f">>> Unsupported `model_name`: {model_name}"
                model_path = self.open_models[model_name]
        self.logger.info(f">>> Loading open LLMs: model_path = {model_path}")

        if do_4bit:
            model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
                model_path, device_map="auto",
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                load_in_4bit=True,
            )
        else:
            base_model_path = self.open_models_local[model_name]
            print("[base_model_path]:", base_model_path, "[ckpt_path]:", ckpt_path)
            if model_name == "gpt-oss-20b":
                base_model_path = "openai/gpt-oss-20b"
                
            model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
                base_model_path, device_map="auto", 
                trust_remote_code=True,
                cache_dir=self.cache_dir, 
                load_in_4bit=False,
                dtype="bfloat16",
                max_seq_length=4096,
            )
            if ckpt_path and base_model_path != ckpt_path:
                # 2) Attach LoRA adapter weights you saved in `last_ckpt_dir`
                model = PeftModel.from_pretrained(model, ckpt_path)
            unsloth.FastLanguageModel.for_inference(model)

        FastLanguageModel.for_inference(model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.verbose:
            self.logger.info(f">>> Model loaded from `{model_path}`")
            self.logger.info(f">>> Number of total parameters: {total_params}")
            self.logger.info(f">>> Number of trainable parameters: {train_params}")

        return tokenizer, model

    @staticmethod
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    # === Helper Function ===
    def load_json(self, path: str, mode: str = "r", encoding: str = "utf-8", verbose: bool = False):
        if os.path.exists(path):
            if verbose:
                self.logger.info(f">>> [load_json] {path}")
            with open(path, mode, encoding=encoding) as f:
                return json.load(f)
        return []

    def save_json(self, path, data, mode: str = "w", encoding: str = "utf-8", verbose: bool = False):
        if verbose:
            self.logger.info(f">>> [save_json] {path}")
        with open(path, mode, encoding=encoding) as f:
            json.dump(data, f, indent=2, ensure_ascii=True, default=self._handle_non_serializable)

    @staticmethod
    def is_causal_lm(model) -> bool:
        return (isinstance(model, AutoModelForCausalLM) or isinstance(model, LlamaForCausalLM) or
                isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen3ForCausalLM) or
                isinstance(model, PeftModelForCausalLM) or isinstance(model,  MistralForCausalLM) or isinstance(model, Gemma3ForConditionalGeneration) or
                isinstance(model, PeftModel))
     

    @staticmethod
    @torch.no_grad()
    def open_model_gen(model, tokenizer, chat, do_sample=True, temperature=1.0, max_new=200, top_p=None):
        inputs = tokenizer.apply_chat_template(
            chat, return_tensors="pt", tokenize=True, return_dict=True, add_generation_prompt=True).to(model.device)
        output = model.generate(**inputs, do_sample=do_sample, temperature=temperature, top_p=top_p,
                                max_new_tokens=max_new)
        decoded = tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return decoded

    @staticmethod
    def gpt_call(model_name, messages, do_parse: bool = False, sleep: float = 1.0):
        # Set up the input prompt (dialog-style) for GPT
        if isinstance(messages, list) and len(messages) > 0:
            input_messages = messages
        elif isinstance(messages, str) and len(messages) > 0:
            input_messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages},
            ]
        else:
            raise ValueError(f">>> Unsupported `messages`: {messages}")

        # API call
        response = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")).with_options(timeout=900.0).chat.completions.create(
            model=model_name,  # or another suitable model
            messages=input_messages,
            # service_tier="flex",
        )
        if sleep > 0.0:
            time.sleep(sleep)

        # Obtain the outputs
        if do_parse:
            # self.logger.info(response.choices[0].message.content)
            norm, ok, report = parse_normalize_validate(response.choices[0].message.content)
            return norm, ok, report
        else:
            res_message = response.choices[0].message
            # refusal = res_message.refusal
            output_text = str(res_message.content).strip()
            return output_text
    
    @staticmethod
    def gemini_call(model_name, user_message, max_tokens=512, do_parse=False):
        response = gemini_client.models.generate_content(
            model=model_name, #"gemini-1.5-flash-8b",  # "gemini-2.0-flash-lite-001",  # "gemini-1.5-flash-8b",
            contents=[user_message]
        )

        # Obtain the outputs
        if do_parse:
            # self.logger.info(response.choices[0].message.content)
            norm, ok, report = parse_normalize_validate(response.text)
            return norm, ok, report
        else:
            output_text = str(response.text).strip()
            return output_text

    @staticmethod
    def qwen_call(model_name, user_message, max_tokens=512, do_parse=False):
        chat = [{
            "role": "user",
            "content": user_message
        }]
        
        completion = open_router_client.chat.completions.create(
            # extra_headers={
            #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            # },
            extra_body={},
            model=model_name,
            messages=chat
        )
        response = completion.choices[0].message.content
        # Obtain the outputs
        if do_parse:
            # self.logger.info(response.choices[0].message.content)
            norm, ok, report = parse_normalize_validate(response)
            return norm, ok, report
        else:
            output_text = str(response).strip()
            return output_text

    def simulate_conversation(self, scenario, agent_a, agent_b, max_turn: int = 20, max_stale_turn: int = 2):
        conversation_history: list[str] = []
        transcript: list[dict] = []
        messages: list[tuple[str, object]] = []  # (agent_name, action_dict_or_obj)

        def _rendered_terminal(text: str) -> bool:
            """Detect rendered terminal strings from your construct_utterance(...)."""
            t = (text or "").strip().lower()
            return t.endswith("left the conversation") # t.endswith("did nothing") or 

        def _atype(_action) -> str:
            """Get normalized action_type from dict or object."""
            return (_action.get("action_type") if isinstance(_action, dict) else getattr(_action, "action_type", "")).lower()

        def _rule_based_terminal(
            turn_number: int,
            messages: list[tuple[str, object]],
            a_name: str,
            b_name: str,
            max_turn_number: int,
            max_stale_turns: int,
        ) -> tuple[bool, str]:
            # Rule 1: conversation too long
            conversation_too_long = turn_number >= max_turn_number

            # Rule 2: leaving (check last two actions like original)
            p1_leaving = (
                len(messages) > 1
                and messages[-2][0] == a_name
                and _atype(messages[-2][1]) == "leave"
            )
            p2_leaving = (
                len(messages) > 0
                and messages[-1][0] == b_name
                and _atype(messages[-1][1]) == "leave"
            )

            # Rule 3: stale too long (consecutive trailing "none")
            stale_count = 0
            for name, act in reversed(messages):
                if _atype(act) == "none":
                    stale_count += 1
                    if stale_count > max_stale_turns:
                        break
                else:
                    break
            stale_too_long = stale_count > max_stale_turns

            _terminated = conversation_too_long or p1_leaving or p2_leaving or stale_too_long
            _reason = (
                f"{'The conversation is too long; ' if conversation_too_long else ''}"
                f"{'Agent 1 is leaving; ' if p1_leaving else ''}"
                f"{'Agent 2 is leaving; ' if p2_leaving else ''}"
                f"{'The conversation stales for too long; ' if stale_too_long else ''}"
            )
            return _terminated, _reason

        def _is_terminal_rendered(text: str, turn_number: int) -> tuple[bool, str]:
            if _rendered_terminal(text):
                return True, "Terminal rendering detected; "
            return _rule_based_terminal(
                turn_number=turn_number,
                messages=messages,
                a_name=agent_a.agent_name,
                b_name=agent_b.agent_name,
                max_turn_number=max_turn,
                max_stale_turns=max_stale_turn,
            )

        terminated = False
        reason = ""

        for turn in range(max_turn):
            agent = agent_a if (turn % 2 == 0) else agent_b
            other_agent = agent_b if (turn % 2 == 0) else agent_a
            ms_text = ""
            if self.use_ms:
                # if generator and partner are different model, then we need to generate mental states only for speaker. so skip using ms for the partner
                if self.generator_model_name != self.gen_partner_model_name and turn%2==1:
                    continue
                chat = [{
                    "role": "user",
                    "content": ms_prompt(
                        history=conversation_history,
                        depth=1,
                        person=agent.agent_name,
                        another_person=other_agent.agent_name,
                        scenario=scenario,
                        social_goal=agent.agent_goal
                    )
                }]
                agent_model = agent.model
                if isinstance(agent_model, str):
                    assert agent_model in PROPRIETARY_LLM, agent_model
                    ms_text = self.gpt_call(PROPRIETARY_LLM[agent_model], chat, do_parse=False)
                    self.gpt_call_cnt += 1
                # elif self.is_causal_lm(agent_model):
                else:
                    ms_text = self.open_model_gen(
                        agent_model, agent.tokenizer, chat, do_sample=True, temperature=0.7, max_new=400, top_p=0.9)
                    self.open_model_gen_cnt += 1
                # else:
                #     raise ValueError(f">>> !!! >>> ValueError: invalid agent.model: {agent_model}")

            strategy_text = ""
            if self.use_strategy:
                # if generator and partner are different model, then we need to generate mental states only for speaker. so skip using ms for the partner
                if self.generator_model_name != self.gen_partner_model_name and turn%2==1:
                    continue
                chat = [{
                    "role": "user",
                    "content": strategy_prompt(
                        history=conversation_history,
                        person=agent.agent_name,
                        another_person=other_agent.agent_name,
                        scenario=scenario,
                        social_goal=agent.agent_goal
                    )
                }]
                agent_model = agent.model
                if isinstance(agent_model, str):
                    raise Exception("Not implemented!")
                else:
                    # import time
                    # start = time.time()
                    strategy_text = self.open_model_gen(
                        agent_model, agent.tokenizer, chat, do_sample=True, temperature=0.7, max_new=400, top_p=0.9)
                    # end = time.time()
                    # print(f"time: {end - start:.3f} seconds")
                    self.open_model_gen_cnt += 1
                    # print("[strategy_text]:", strategy_text)
            
            cot_text = ""
            if self.use_cot:
                # if generator and partner are different model, then we need to generate mental states only for speaker. so skip using ms for the partner
                if self.generator_model_name != self.gen_partner_model_name and turn%2==1:
                    continue
                chat = [{
                    "role": "user",
                    "content": cot_prompt(
                        history=conversation_history,
                        person=agent.agent_name,
                        another_person=other_agent.agent_name,
                        scenario=scenario,
                        social_goal=agent.agent_goal
                    )
                }]
                agent_model = agent.model
                if isinstance(agent_model, str):
                    raise Exception("Not implemented!")
                else:
                    cot_text = self.open_model_gen(
                        agent_model, agent.tokenizer, chat, do_sample=True, temperature=0.7, max_new=400, top_p=0.9)
                    self.open_model_gen_cnt += 1
                    
            action = agent.act(
                history=conversation_history,
                do_sample=True,
                max_new=200,
                temperature=0.7,
                top_p=0.9,
                ms_text=ms_text,
                strategy_text=strategy_text,
                cot_text=cot_text,
            )
            utterance = construct_utterance(action, agent.agent_name)  # your renderer -> str

            # record
            messages.append((agent.agent_name, action))
            if utterance:
                conversation_history.append(utterance)
                transcript.append({
                    "name": agent.agent_name, "action_type": action.action_type, "argument": action.argument,
                    "ms_text": action.mental_state if action.mental_state else ms_text,
                    "strategy_text": strategy_text if strategy_text else "",
                    "cot_text": cot_text if cot_text else "",
                })

            # merged termination check
            terminated, r = _is_terminal_rendered(utterance, turn_number=turn + 1)
            reason += r
            if terminated:
                break

        meta = {"terminated": terminated, "reason": reason, "turns": len(transcript)}
        return conversation_history, transcript, meta

    def compute_final_scores(
            self,
            results: list,
    ) -> dict:
        all_scores = {
            "believability": [],
            "relationship": [],
            "knowledge": [],
            "secret": [],
            "social_rules": [],
            "financial_and_material_benefits": [],
            "goal": [],
        }
        done_cnt = 0
        status_false_cnt = 0
        wrong_format_cnt = 0
        for result_dict in results:
            if not (isinstance(result_dict, dict) and len(result_dict) > 0):
                wrong_format_cnt += 1
                continue
            for agent_id, v in result_dict.items():
                if isinstance(v, dict) and "scores" in v:
                    if not ("status" in v and isinstance(v["status"], bool)):
                        wrong_format_cnt += 1
                        continue
                    if v["status"]:
                        cur_score_dict = v["scores"]
                        if not (isinstance(cur_score_dict, dict) and len(cur_score_dict) > 0):
                            wrong_format_cnt += 1
                            continue

                        try:
                            all_scores["believability"].append(float(cur_score_dict["believability"]["score"]))
                            all_scores["relationship"].append(float(cur_score_dict["relationship"]["score"]))
                            all_scores["knowledge"].append(float(cur_score_dict["knowledge"]["score"]))
                            all_scores["secret"].append(float(cur_score_dict["secret"]["score"]))
                            all_scores["social_rules"].append(float(cur_score_dict["social_rules"]["score"]))
                            all_scores["financial_and_material_benefits"].append(
                                float(cur_score_dict["financial_and_material_benefits"]["score"]))
                            all_scores["goal"].append(float(cur_score_dict["goal"]["score"]))
                            done_cnt += 1
                        except Exception as e:
                            self.logger.info(e)
                            wrong_format_cnt += 1
                            continue
                    else:
                        status_false_cnt += 1
                        continue

        avg_scores = {
            "believability": np.mean(all_scores["believability"]).item(),
            "relationship": np.mean(all_scores["relationship"]).item(),
            "knowledge": np.mean(all_scores["knowledge"]).item(),
            "secret": np.mean(all_scores["secret"]).item(),
            "social_rules": np.mean(all_scores["social_rules"]).item(),
            "financial_and_material_benefits": np.mean(all_scores["financial_and_material_benefits"]).item(),
            "goal": np.mean(all_scores["goal"]).item(),
        }
        self.logger.info(f">>> `compute_final_scores` >>> done_cnt = {done_cnt}; "
                         f"status_false_cnt = {status_false_cnt}; wrong_format_cnt = {wrong_format_cnt}"
                         f"\n{avg_scores}")
        return avg_scores

    def run_eval_open(
        self,
        generator_model_name: str = "qwen2.5-3b",
        gen_partner_model_name: Optional[str] = None,
        evaluator_model_name: str = "gpt-5-mini",
        generator_ckpt_path: Optional[str] = None,
        gen_partner_ckpt_path: Optional[str] = None,
        evaluator_ckpt_path: Optional[str] = None,
        do_4bit: bool = False,
        eval_turns: int = -1,  # For "Exp3"; -1 means evaluating all turns of the generated conversation.
    ):
        self.logger.info(f"[gen_only]: {self.gen_only}, [eval_only]: {self.eval_only}")
        if self.gen_only:
            self.logger.info(f">>> `run_eval_open` gen_only: generator = {generator_model_name}")
            self.run_gen_only(
                generator_model_name, gen_partner_model_name, generator_ckpt_path, gen_partner_ckpt_path,
                do_4bit=do_4bit, do_save_results=True)
        elif self.eval_only:
            self.logger.info(f">>> `run_eval_open` eval_only: evaluator = {evaluator_model_name}")
            assert self.eval_filepath, f"eval_filepath is None!"
            self.run_eval_only(evaluator_model_name, results_fp=self.eval_filepath, eval_turns=eval_turns)
        else:
            self.logger.info(f">>> `run_eval_open` gen_eval: "
                             f"generator = {generator_model_name}; evaluator = {evaluator_model_name}")
            self.run_gen_eval(
                generator_model_name, gen_partner_model_name, evaluator_model_name,
                generator_ckpt_path, gen_partner_ckpt_path, evaluator_ckpt_path,
                do_4bit=do_4bit, eval_turns=eval_turns,
            )

    def run_gen_only(
            self,
            generator_model_name: str = "qwen2.5-3b",
            gen_partner_model_name: Optional[str] = None,
            generator_ckpt_path: Optional[str] = None,
            gen_partner_ckpt_path: Optional[str] = None,
            do_4bit: bool = False,
            do_save_results: bool = True,
    ):
        # Generation-only: construct the conversations and save them into a JSON file
        if self.generator_model is None:
            if generator_model_name in self.proprietary_models or generator_model_name == "qwen2.5-72b" or generator_model_name == "gemini" or generator_model_name == "gpt-120b":
                generator_tokenizer = None
                generator_model = generator_model_name
            else:
                generator_tokenizer, generator_model = self.initialize_model(
                    model_name=generator_model_name, ckpt_path=generator_ckpt_path, do_4bit=do_4bit)
                self.generator_tokenizer = generator_tokenizer
                self.generator_model = generator_model
        else:
            generator_tokenizer = self.generator_tokenizer
            generator_model = self.generator_model
        self.generator_model_name = generator_model_name

        if gen_partner_model_name in self.proprietary_models  or gen_partner_model_name == "qwen2.5-72b" or gen_partner_model_name == "gemini" or gen_partner_model_name == "gpt-120b":
            gen_partner_tokenizer = None
            generator_partner_model = gen_partner_model_name
        else:
            gen_partner_tokenizer, generator_partner_model = self.initialize_model(
                model_name=gen_partner_model_name, ckpt_path=gen_partner_ckpt_path, do_4bit=do_4bit)
            self.gen_partner_tokenizer = gen_partner_tokenizer
            self.generator_partner_model = generator_partner_model

        self.logger.info(f">>> generator_model = {generator_model_name}; gen_partner_model = {gen_partner_model_name}")
        self.gen_partner_model_name = gen_partner_model_name

        dataset_loader = DatasetLoader(
            dataset_name="sotopia-episodes", eval_mode=self.eval_mode,
            cache_dir=self.cache_dir, data_dir=self.project_root_dir)
        data_items = dataset_loader.load_data()
        num_items = len(data_items)
        self.logger.info(f">>> num_items = {num_items}")

        if isinstance(generator_ckpt_path, str) and "checkpoint" in generator_ckpt_path:
            generator_model_name = "/".join(generator_ckpt_path.split("/")[-4:])

        if isinstance(generator_ckpt_path, str) and len(generator_ckpt_path) > 0 and "checkpoint" in generator_ckpt_path:
            if "--ms-uttr-sep" in generator_ckpt_path:
                self.logger.info(">>> Override use_ms to True..")
                self.use_ms = True
                train_type = "ms-uttr-sep"
            elif "--ms" in generator_ckpt_path:
                self.logger.info(">>> Override use_ms to True..")
                self.use_ms = True
                train_type = "ms"
            elif "--uttr" in generator_ckpt_path:
                self.logger.info(">>> Override use_ms to False..")
                self.use_ms = False
                train_type = "uttr"
            else:
                train_type = "none"
        else:
            if self.use_ms:
                train_type = "ms"
            elif self.use_strategy:
                train_type = "strategy"
            elif self.use_cot:
                train_type = "cot"
            else:
                train_type = "none"
        logging.info(f">>> [train_type]: {train_type}")

        if len(generator_model_name.split("/")) > 1:
            generator_model_name_one_word = [x for x in generator_model_name.split("/") if x][-1]
        else:
            generator_model_name_one_word = generator_model_name

        if len(gen_partner_model_name.split("/")) > 1:
            generator_partner_model_name_one_word = [x for x in gen_partner_model_name.split("/") if x][-1]
        else:
            generator_partner_model_name_one_word = gen_partner_model_name
        save_dir = os.path.join(self.output_dir, f"{generator_model_name}__{generator_partner_model_name_one_word}__use_ms{self.use_ms}__use_strategy{self.use_strategy}__use_cot{self.use_cot}")
        os.makedirs(save_dir, exist_ok=True)
        
        results_fp = os.path.join(save_dir, f"{self.eval_mode}_{generator_model_name_one_word}__{train_type}__results.json")

        # resume running
        if os.path.isfile(results_fp):
            results = self.load_json(results_fp, "r+", verbose=True)
            if isinstance(results, list) and len(results) > 0:
                if "processed_id_key_gen" in results[0]:
                    processed_ids = set([_res["processed_id_key_gen"] for _res in results])
                else:
                    processed_ids = set(["||".join(
                        [res["environment_id"], res["agents"][0]["id"], res["agents"][1]["id"]]) for res in results])
                if "open_model_gen_cnt" in results[0]:
                    self.open_model_gen_cnt = max([_res["open_model_gen_cnt"] for _res in results])
                else:
                    self.open_model_gen_cnt = 0
                if "gpt_call_cnt" in results[0]:
                    self.gpt_call_cnt = max([_res["gpt_call_cnt"] for _res in results])
                else:
                    self.gpt_call_cnt = 0
            else:
                results = []
                processed_ids = set()
            self.logger.info(f"Results are loaded from here: {results_fp}; # Processed: {len(results)}")
        else:
            results = []
            processed_ids = set()  # Determine which items are already processed
            self.logger.info(f"Results will be saved at: {results_fp}")

        meta_stat = {
            "terminated": dict(),
            "reason": dict(),
            "turns": dict(),
        }
        print(f"[generator_ckpt_path]: {generator_ckpt_path}, [gen_partner_ckpt_path]: {gen_partner_ckpt_path}, [is_equal]: {generator_ckpt_path == gen_partner_ckpt_path}")

        for idx, item in enumerate(data_items):
            # For each item (conversation), construct an episode
            environment_id = item["environment_id"]
            scenario = item["scenario"]
            # relationship = item["relationship"]
            agent_pair = item["agents"]
            processed_id_key_gen = "||".join([item["environment_id"], agent_pair[0]["id"], agent_pair[1]["id"]])
            if processed_id_key_gen in processed_ids:
                self.logger.info(f"[Skip] environment and agents are processed already: {processed_id_key_gen}")
                continue

            if self.verbose:
                self.logger.info(f">>> [{idx + 1} / {num_items}] >>> [environment_id]: {environment_id}")
                self.logger.info(f">>> [{idx + 1} / {num_items}] >>> [scenario]: {scenario}")

            # Set up agent objects
            agents = []
            assert len(agent_pair) == 2
            for agent_idx, agent in enumerate(agent_pair):
                if agent_idx == 0:
                    agent = LLMAgent(
                        agent_id=agent["id"],
                        agent_name=agent["name"],
                        scenario=scenario,
                        agent_background=agent["bio"],
                        agent_goal=agent["goal"],
                        model=generator_model,
                        tokenizer=generator_tokenizer,
                        model_name=self.generator_model_name,
                        partner_model_name=gen_partner_model_name,
                        train_type=train_type,
                        is_partner=False,
                    )
                else:
                    agent = LLMAgent(
                        agent_id=agent["id"],
                        agent_name=agent["name"],
                        scenario=scenario,
                        agent_background=agent["bio"],
                        agent_goal=agent["goal"],
                        model=generator_partner_model,
                        tokenizer=gen_partner_tokenizer,
                        model_name=self.gen_partner_model_name,
                        partner_model_name=self.generator_model_name,
                        train_type=train_type,
                        is_partner=True,
                    )
                agents.append(agent)

            # Generate conversations
            conversation, transcript, meta = self.simulate_conversation(
                scenario=scenario,
                agent_a=agents[0],
                agent_b=agents[1],
                max_turn=20,
            )

            # Statistics on the meta information
            cur_terminated, cur_reason, cur_turns = meta["terminated"], meta["reason"], meta["turns"]
            if cur_terminated not in meta_stat["terminated"]:
                meta_stat["terminated"][cur_terminated] = 1
            else:
                meta_stat["terminated"][cur_terminated] += 1
            if cur_reason not in meta_stat["reason"]:
                meta_stat["reason"][cur_reason] = 1
            else:
                meta_stat["reason"][cur_reason] += 1
            if cur_turns not in meta_stat["turns"]:
                meta_stat["turns"][cur_turns] = 1
            else:
                meta_stat["turns"][cur_turns] += 1

            for agent in agents:
                self.open_model_gen_cnt += agent.open_model_gen_cnt
                self.gpt_call_cnt += agent.gpt_call_cnt

            cur_result = {
                **item,
                "processed_id_key_gen": processed_id_key_gen,
                "open_model_gen_cnt": self.open_model_gen_cnt,
                "gpt_call_cnt": self.gpt_call_cnt,
                "generator_model_name": generator_model_name,
                # "evaluator_model_name": evaluator_model_name,
                "conversation": conversation,
                "transcript": transcript,
                "meta": meta,
                # **evaluation_scores,
            }
            results.append(cur_result)
            processed_ids.add(processed_id_key_gen)

            if generator_ckpt_path != gen_partner_ckpt_path:
                processed_id_key_gen2 = "||".join([item["environment_id"], agent_pair[1]["id"], agent_pair[0]["id"]])
                # Generate conversations
                conversation2, transcript2, meta2 = self.simulate_conversation(scenario=scenario, agent_a=agents[1], agent_b=agents[0], max_turn=20)
                cur_result2 = {
                    **item,
                    "processed_id_key_gen": processed_id_key_gen2,
                    "open_model_gen_cnt": self.open_model_gen_cnt,
                    "gpt_call_cnt": self.gpt_call_cnt,
                    "generator_model_name": generator_model_name,
                    # "evaluator_model_name": evaluator_model_name,
                    "conversation": conversation2,
                    "transcript": transcript2,
                    "meta": meta2,
                    # **evaluation_scores,
                }
                results.append(cur_result2)
                processed_ids.add(processed_id_key_gen2)

            if do_save_results:
                self.save_json(results_fp, results, verbose=False)

        if do_save_results:
            self.save_json(results_fp, results, verbose=True)
        self.logger.info(f">>> meta_stat:\n{meta_stat}")

        # Show the total generation counters
        self.logger.info(
            f">>> DONE GEN. open_model_gen_cnt = {self.open_model_gen_cnt}; gpt_call_cnt = {self.gpt_call_cnt}")

        return results, results_fp

    def run_eval_only(
            self,
            evaluator_model_name: str = "gpt-5-mini",
            do_save_results: bool = True,
            results_fp: str = "",
            eval_turns: int = -1,
    ):
        # Evaluation-only: load the conversations from the JSON file and evaluate them

        self.logger.info(f">>> `run_eval_only`: evaluator_model = {evaluator_model_name}; eval_turns = {eval_turns}")
        self.evaluator_model_name = evaluator_model_name
        assert os.path.isfile(results_fp) and os.path.exists(results_fp)


        if evaluator_model_name.startswith("qwen") or evaluator_model_name.startswith("deepseek"):
            if evaluator_model_name == "qwen":
                evaluator_model_name = "qwen/qwen3-235b-a22b-2507"
            elif evaluator_model_name == "gpt-120b":
                evaluator_model_name = "openai/gpt-oss-120b:free"
            elif evaluator_model_name.startswith("deepseek"):
                evaluator_model_name = "deepseek/deepseek-chat-v3.1"
            

        if eval_turns > 0:
            results_fp_dir = os.path.join("/".join(results_fp.split("/")[:-1]), f"eval_turn{eval_turns}")
            os.makedirs(results_fp_dir, exist_ok=True)
            out_results_fp = os.path.join(results_fp_dir, results_fp.split("/")[-1])
        else:
            results_fp_dir = os.path.join("/".join(results_fp.split("/")[:-1]), evaluator_model_name)
            os.makedirs(results_fp_dir, exist_ok=True)
            out_results_fp = os.path.join(results_fp_dir, results_fp.split("/")[-1])

        results = []
        processed_ids = set()
        # resume running only if the output filepath is same as input filepath (i.e. eval_turns=-1)
        if os.path.isfile(results_fp):
            results = self.load_json(results_fp, "r+", verbose=True)

        if os.path.isfile(out_results_fp):
            out_results = self.load_json(out_results_fp, "r+", verbose=True)
            for res in out_results:
                agent_ids = [x["id"] for x in res["agents"]]
                is_valid = True
                for agent_id in agent_ids:
                    if agent_id not in res or not res[agent_id]["scores"]:
                        is_valid = False
                        break
                if is_valid:
                    if "processed_id_key_eval" in res:
                        processed_ids.add(res["processed_id_key_eval"])
                    # else:
                    #     processed_ids.add("||".join([res["environment_id"], res["agents"][0]["id"], res["agents"][1]["id"]]))
        self.logger.info(f"Results are loaded from here: {results_fp}; # Processed: {len(processed_ids)}")
        self.logger.info(f"Results will be saved at: {out_results_fp}")
    
        num_items = len(results)
        for idx, item in enumerate(results): # results are the generated items
            # For each item (conversation), construct an episode
            environment_id = item["environment_id"]
            scenario = item["scenario"]
            # relationship = item["relationship"]
            agent_pair = item["agents"]
            agents = [(agent_pair[0]["id"], agent_pair[0]["name"], agent_pair[0]["goal"]), (agent_pair[1]["id"], agent_pair[1]["name"], agent_pair[1]["goal"])]
            # print("processed_id_key_eval:", item["processed_id_key_eval"])
            # print(item["processed_id_key_eval"] in processed_ids)
            processed_id_key_eval = "||".join([item["environment_id"], agent_pair[0]["id"], agent_pair[1]["id"]])
            if processed_id_key_eval in processed_ids:
                print(f"Skip {processed_id_key_eval}..")
                continue
            # else:
            #     processed_id_key_eval = "||".join([item["environment_id"], agent_pair[0]["id"], agent_pair[1]["id"]])
            # processed_id_key_eval2 = "||".join([item["environment_id"], agent_pair[1]["id"], agent_pair[0]["id"]])
            # if processed_id_key_eval in processed_ids:
            #     if (agent_pair[0]["id"] in item and "scores" in item[agent_pair[0]["id"]] and item[agent_pair[0]["id"]]["scores"]):
            #         self.logger.info(f"[Skip] environment and agents are processed already: {processed_id_key_eval}")
            #         continue
            # if processed_id_key_eval2 in processed_ids:
            #     if (agent_pair[0]["id"] in item and "scores" in item[agent_pair[0]["id"]] and item[agent_pair[0]["id"]]["scores"]):
            #         self.logger.info(f"[Skip] environment and agents are processed already: {processed_id_key_eval}")
            #         continue

            if self.verbose:
                self.logger.info(f">>> [{idx + 1} / {num_items}] >>> [environment_id]: {environment_id}")
                self.logger.info(f">>> [{idx + 1} / {num_items}] >>> [scenario]: {scenario}")

            assert "conversation" in item and "transcript" in item and "meta" in item, item
            conversation, _, _ = item["conversation"], item["transcript"], item["meta"]
            assert isinstance(conversation, list) and len(conversation) > 0, conversation

            if eval_turns > 0:
                eval_conv = conversation[:eval_turns]
            else:
                eval_conv = conversation

            # dimensions = ["believability", "relationship", "knowledge", "secret", "social_rules",
            #               "financial_and_material_benefits", "goal"]
            dimensions = ["relationship", "knowledge", "goal"]
            evaluation_scores = collections.defaultdict(dict)
            for (agent_id, agent_name, agent_goal) in agents:
                prompt = build_all_dimensions_prompt(
                    scenario=scenario,
                    agent=agent_name,
                    goal=agent_goal,
                    history="\n".join(eval_conv),
                    dimensions=dimensions,
                )
                if evaluator_model_name.startswith("gpt"):
                    norm, ok, report = self.gpt_call(evaluator_model_name, prompt, do_parse=True)
                elif evaluator_model_name.startswith("gemini"):
                    norm, ok, report = self.gemini_call(evaluator_model_name, prompt, do_parse=True)
                elif evaluator_model_name.startswith("qwen") or evaluator_model_name.startswith("deepseek"):
                    norm, ok, report = self.qwen_call(evaluator_model_name, prompt, do_parse=True)
                    
                self.gpt_call_cnt += 1
                evaluation_scores[agent_id] = {
                    "scores": norm,
                    "status": ok
                }

            item.update({
                "evaluator_model_name": evaluator_model_name,
                "processed_id_key_eval": processed_id_key_eval,
                **evaluation_scores,
            })
            processed_ids.add(processed_id_key_eval)
            if do_save_results:
                self.save_json(out_results_fp, results, verbose=False)

        if do_save_results:
            self.save_json(out_results_fp, results, verbose=True)
        # self.compute_final_scores(results)

        # Show the total generation counters
        self.logger.info(
            f">>> DONE EVAL. open_model_gen_cnt = {self.open_model_gen_cnt}; gpt_call_cnt = {self.gpt_call_cnt}")

        return results, results_fp

    def run_gen_eval(
            self,
            generator_model_name: str = "qwen2.5-3b",
            gen_partner_model_name: Optional[str] = None,
            evaluator_model_name: str = "gpt-5-mini",
            generator_ckpt_path: Optional[str] = None,
            gen_partner_ckpt_path: Optional[str] = None,
            evaluator_ckpt_path: Optional[str] = None,
            do_4bit: bool = False,
            eval_turns: int = -1,
    ):
        # Generation + Evaluation: construct the conversations, evaluate them, and save the scores into a JSON file
        logging.info(f"generator_model_name: {generator_model_name}")
        logging.info(f"gen_partner_model_name: {gen_partner_model_name}")

        _, gen_results_fp = self.run_gen_only(
            generator_model_name=generator_model_name, gen_partner_model_name=gen_partner_model_name,
            generator_ckpt_path=generator_ckpt_path, gen_partner_ckpt_path=gen_partner_ckpt_path,
            do_4bit=do_4bit, do_save_results=True,
        )

        if self.eval_filepath:
            gen_results_fp = self.eval_filepath

        # save evaluation results to the same file that was generated from the previous step by passing 'gen_results_fp'
        eval_results, eval_results_fp = self.run_eval_only(
            results_fp=gen_results_fp,
            evaluator_model_name=evaluator_model_name,
            eval_turns=eval_turns,
        )

        # Show the total generation counters
        self.logger.info(
            f">>> DONE GEN_EVAL. open_model_gen_cnt = {self.open_model_gen_cnt}; gpt_call_cnt = {self.gpt_call_cnt}"
            f"\ngen_results_fp: {gen_results_fp}\neval_results_fp: {eval_results_fp}")

        return eval_results, eval_results_fp
