# -*- coding: utf-8 -*-

import unsloth

import os
import sys
# import json
import time
import random
# import pprint
import logging
import collections
from typing import Optional

# import re
import math
import numpy as np
import tqdm

# import torch
from datasets import load_dataset

from google import genai
from google.genai import types

from utils.parse_actions import *
from utils.sotopia_prompts import *
from sotopia import SotopiaEvaluator


class DataloaderSotopiaRaw:
    """
    Load the raw datasets of Sotopia, including "sotopia", "sotopia-episodes", "sotopia-pi", and "sotopia-pi-episodes"
    """

    def __init__(
            self,
            dataset_name: str,
            project_root_dir: str,
            cache_dir: Optional[str] = None,
            logger=None,
    ):
        if logger is None:
            self.logger = logging.getLogger("DataloaderSotopiaRaw")
        else:
            self.logger = logger

        self.dataset_name = dataset_name
        self.project_root_dir = project_root_dir
        self.home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.home_dir, ".cache/huggingface")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f">>> dataset_name: {self.dataset_name}; cache_dir: {self.cache_dir}")

        self.ds = None

    def load_sotopia_episodes(
            self,
            data_path: str = "data/sotopia/sotopia_episodes_v1.jsonl",
    ):
        data_fp = os.path.join(self.project_root_dir, data_path)
        assert os.path.isfile(data_fp), f">>> {data_fp} does not exist!"

        data_raw = []
        with open(data_fp, "r", encoding="utf-8") as fp_in:
            for line in fp_in:
                data_raw.append(json.loads(line.strip()))

        num_data = len(data_raw)
        self.logger.info(f">>> num_data = {num_data}")  # 7200

        data_tom = []
        for item in data_raw:
            # self.logger.info(item)
            episode_id = item["episode_id"]  # str
            environment_id = item["environment_id"]  # str
            agent_ids = item["agent_ids"]  # List[str]
            # experiment_tag = item["experiment_tag"]  # str
            # experiment_model_name_pairs = item["experiment_model_name_pairs"]  # List[str]
            raw_messages = item["raw_messages"]  # list
            # raw_rewards = item["raw_rewards"]  # List[Union[float, dict]]
            # raw_rewards_prompt = item["raw_rewards_prompt"]  # str
            scenario = item["scenario"]  # <main> str
            # codename = item["codename"]  # str
            agents_background = item["agents_background"]  # dict{agent_name: background}
            social_goals = item["social_goals"]  # dict{agent_name: goal}
            social_interactions = item["social_interactions"]  # str
            reasoning = item["reasoning"]  # str
            rewards = item["rewards"]  # List[dict]

            cur_item = {
                "episode_id": episode_id,
                "environment_id": environment_id,
                "agent_ids": agent_ids,
                "raw_messages": raw_messages,
                "scenario": scenario,
                "agents_background": agents_background,
                "social_goals": social_goals,
                "social_interactions": social_interactions,
                "reasoning": reasoning,
                "rewards": rewards,
            }
            data_tom.append(cur_item)
            # self.logger.info(cur_item)

        return data_tom

    def load_sotopia_pi_episodes(
            self,
            data_path: str = "data/sotopia-pi/sotopia_pi_episodes.jsonl",
    ):
        data_fp = os.path.join(self.project_root_dir, data_path)
        assert os.path.isfile(data_fp), f">>> {data_fp} does not exist!"

        data_raw = []
        with open(data_fp, "r", encoding="utf-8") as fp_in:
            for line in fp_in:
                data_raw.append(json.loads(line.strip()))

        num_data = len(data_raw)
        self.logger.info(f">>> num_data = {num_data}")  # 2281

        data_tom = []
        for item in data_raw:
            # self.logger.info(item)
            episode_id = item["episode_id"]  # str
            environment_id = item["environment_id"]  # str
            agent_ids = item["agent_ids"]  # List[str]
            # experiment_tag = item["experiment_tag"]  # str
            # experiment_model_name_pairs = item["experiment_model_name_pairs"]  # List[str]
            raw_messages = item["raw_messages"]  # list
            # raw_rewards = item["raw_rewards"]  # List[Union[float, dict]]
            # raw_rewards_prompt = item["raw_rewards_prompt"]  # str
            scenario = item["scenario"]  # <main> str
            # codename = item["codename"]  # str
            agents_background = item["agents_background"]  # dict{agent_name: background}
            social_goals = item["social_goals"]  # dict{agent_name: goal}
            social_interactions = item["social_interactions"]  # str
            reasoning = item["reasoning"]  # str
            rewards = item["rewards"]  # List[Union[float, dict]]

            rewards = [_rw[1] for _rw in rewards]  # Only keep the dict

            cur_item = {
                "episode_id": episode_id,
                "environment_id": environment_id,
                "agent_ids": agent_ids,
                "raw_messages": raw_messages,
                "scenario": scenario,
                "agents_background": agents_background,
                "social_goals": social_goals,
                "social_interactions": social_interactions,
                "reasoning": reasoning,
                "rewards": rewards,
            }
            data_tom.append(cur_item)
            # self.logger.info(cur_item)

        return data_tom

    def load_sotopia_pi(
            self,
            data_path: str = "cmu-lti/sotopia-pi",
    ):
        self.ds = load_dataset(data_path, cache_dir=self.cache_dir)["train"]
        self.logger.info(self.ds)

        data_tom = []
        for item in self.ds:
            # self.logger.info(item)
            # context = item["context"]
            # question = item["question"]
            # answerA = item["answerA"]
            # answerB = item["answerB"]
            # answerC = item["answerC"]

            data_tom.append(item)

        return data_tom

    def load_sotopia(
            self,
            data_path: str = "cmu-lti/sotopia",
    ):
        # self.ds = load_dataset("cmu-lti/sotopia")
        self.ds = load_dataset(
            data_path, data_files="sotopia_episodes_v1_hf.jsonl", split="train", cache_dir=self.cache_dir)
        # Training set info: 7200 items with 14 features
        # self.logger.info(dataset[0])

        data_tom = []
        for item in self.ds:
            # self.logger.info(item.keys())
            episode_id = item["episode_id"]  # str
            environment_id = item["environment_id"]  # str
            agent_ids = item["agent_ids"]  # List[str]
            # experiment_model_name_pairs = item["experiment_model_name_pairs"]  # str
            raw_messages = item["raw_messages"]  # List[str]
            # raw_rewards_prompt = item["raw_rewards_prompt"]  # str
            scenario = item["scenario"]  # str
            # codename = item["codename"]  # str
            agents_background = item["agents_background"]  # str --> dict{agent_name: background}
            social_goals = item["social_goals"]  # str --> dict{agent_name: goal}
            social_interactions = item["social_interactions"]  # str
            reasoning = item["reasoning"]  # str
            rewards = item["rewards"]  # List[dict]

            cur_item = {
                "episode_id": episode_id,
                "environment_id": environment_id,
                "agent_ids": agent_ids,
                "raw_messages": raw_messages,
                "scenario": scenario,
                "agents_background": agents_background,
                "social_goals": social_goals,
                "social_interactions": social_interactions,
                "reasoning": reasoning,
                "rewards": rewards,
            }
            data_tom.append(cur_item)
            # self.logger.info(cur_item)

        return data_tom

    def load_data(self):
        if self.dataset_name == "sotopia":
            data_tom = self.load_sotopia()
        elif self.dataset_name == "sotopia-pi":
            data_tom = self.load_sotopia_pi()
        elif self.dataset_name == "sotopia-episodes":
            data_tom = self.load_sotopia_episodes()
        elif self.dataset_name == "sotopia-pi-episodes":
            data_tom = self.load_sotopia_pi_episodes()
        else:
            raise ValueError(f">>> `dataset_name` {self.dataset_name} not supported")

        # Note: `raw_messages` format
        # raw_messages[i] is the i-th turn.
        # raw_messages[i] is a list of four items, and each item is a triplet.
        #     raw_messages[i][0]: ["Environment", "Agent 1", The dialogue of Turn #{i-1}]
        #     raw_messages[i][1]: ["Environment", "Agent 2", The dialogue of Turn #{i-1}]
        #     raw_messages[i][2]: ["Agent 1", "Environment", XXX]
        #     raw_messages[i][3]: ["Agent 2", "Environment", YYY]
        # Only one agent speaks in each turn:
        #     If XXX is the dialogue ('said: "..."') of Turn #i, YYY is "did nothing", and vice versa.

        self.ds = data_tom
        return data_tom


class SampleDataloader:

    def __init__(
            self,
            tokenizer_train,
            tokenizer_eval,
            model,
            simulator,
            simulator_tokenizer,
            model_name: str,
            project_root_dir: str,
            cache_dir: Optional[str] = None,
            do_eval: bool = False,
            debug: bool = False,
            train_type: str = "ms", # [ms, uttr, ms-uttr-sep]
            outer_epoch_num: int = 0, 
            total_finetune_epochs: int = 1,
            logger = None,
            do_sample_ranking: bool = False,
            epoch_new_data: int = 10,
            lora_rank: int = 64,
            lora_alpha: int = 64,
            ms_num: int=2,
            uttr_num: int=2,
            rollout_turns: int=2,
    ):
        if logger is None:
            self.logger = logging.getLogger("SampleDataloader")
        else:
            self.logger = logger

        self.tokenizer_train = tokenizer_train
        self.tokenizer_eval = tokenizer_eval
        self.model = model
        unsloth.FastLanguageModel.for_inference(self.model)
        self.simulator = simulator
        unsloth.FastLanguageModel.for_inference(self.simulator)
        self.simulator_tokenizer = simulator_tokenizer
        self.train_type = train_type
        self.outer_epoch_num = outer_epoch_num
        self.total_finetune_epochs = total_finetune_epochs
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.ms_num = ms_num
        self.uttr_num = uttr_num
        self.rollout_turns = rollout_turns
        self.logger.info(f">>> ms_num: {ms_num}, uttr_num: {uttr_num}, rollout_turns: {rollout_turns}")

        self.judge = genai.Client(api_key=os.environ['GEMINI_KEY'])

        self.model_name = model_name
        self.project_root_dir = project_root_dir
        self.home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.home_dir, ".cache/huggingface")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f">>> project_root_dir: {self.project_root_dir} cache_dir: {self.cache_dir}")

        self.do_eval = do_eval
        self.debug = debug
        self.do_sample_ranking = do_sample_ranking
        assert isinstance(epoch_new_data, int) and 0 <= epoch_new_data <= 100
        self.epoch_new_data = epoch_new_data
        self.logger.info(f">>> do_eval: {do_eval}; "
                         f"do_sample_ranking: {self.do_sample_ranking}; epoch_new_data = {self.epoch_new_data}%")

        self.train_data = None
        self.eval_data = None
        self.ds = None

        # self._load_sample_data()

    def call_gemini(
            self,
            user_prompt: str,
            max_gen_len: int = -1,
            temperature: Optional[float] = 0.0,
    ):
        try_cnt = 0
        api_retry_limit = 3
        api_call_sleep = 10
        while True:
            try_cnt += 1
            try:
                response = self.judge.models.generate_content(
                    model="gemini-2.0-flash-lite-001", #"gemini-2.0-flash-lite-001",  # "gemini-1.5-flash-8b",
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_gen_len,
                        temperature=float(temperature)
                    )
                )
                if api_call_sleep > 0:  # Regular sleep time before running next data item
                    time.sleep(api_call_sleep)
                return response.text
            except Exception as e:
                self.logger.info(f">>> !!! >>> Gemini Exception: {e}")
                if try_cnt >= api_retry_limit:
                    sys.exit(1)
                time.sleep(10)  # Sleep time before retrying API calls

    @staticmethod
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    def _load_sample_data(
            self,
            data_path: str = "data/gemini-tom-sotopia-pi-episodes.json",
    ):
        sample_filepath = os.path.join(self.project_root_dir, data_path)
        assert os.path.isfile(sample_filepath)
        with open(sample_filepath, "r", encoding="utf-8") as fp_in:
            sample_data = json.load(fp_in)
        assert isinstance(sample_data, list) and len(sample_data) >= 2

        if self.do_eval:
            self.train_data = sample_data[:-1]
            self.eval_data = [sample_data[-1]]
        else:
            self.train_data = sample_data
            self.eval_data = []

        if self.debug:
            self.train_data = self.train_data[:10]

    @staticmethod
    def load_json(path, default):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                logging.info(f"[Load] {path}")
                return json.load(f)
        return default

    def save_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True, default=self._handle_non_serializable)
            self.logger.info(f"[Save] {path}")

    def generate_training_instances(
        self,
        all_generated_instances,
        target_episode_ids,
        train_type,
        score_threshold: float = 9.0,          # keep rollouts with avg score >= this
        max_per_scenario: Optional[int] = 3,   # hard cap per scenario (used for 'uttr' and 'ms')
        max_ms_per_scenario: Optional[int] = 3,          # [ms-uttr-sep] max unique mental states per scenario
        max_uttrs_per_ms: Optional[int] = 3,      # [ms-uttr-sep] max utterances per (scenario, MS); None = unlimited
    ):
        """
        Build training instances from pre-generated rollouts.

        - For 'uttr' and 'ms': use `max_per_scenario` as the hard per-scenario cap.
        - For 'ms-uttr-sep': cap the number of unique MS per scenario (`max_ms_per_scenario`),
        but allow multiple utterances for MS that are already included (up to `max_uttrs_per_ms`).
        - Chooses higher-scoring rollouts first (descending avg score) so any cap is filled by the best items.
        - Deduplicates mental states (normalized) and has a per-conversation fallback.
        """
        output_instances = []
        per_conv_added_hist = []     # how many were added per conversation (for logging)
        # added_uttr, added_ms_uttr = 0, 0

        # Track per-scenario counts (for 'uttr'/'ms' caps)
        added_by_scenario: dict[str, int] = collections.defaultdict(int)

        # [ms-uttr-sep] Track per-scenario unique MS and per-(scenario, MS) utterance counts
        ms_set_by_scenario: dict[str, set[str]] = collections.defaultdict(set)            # scenario -> {ms_key, ...}
        uttr_count_by_ms: dict[tuple[str, str], int] = collections.defaultdict(int)       # (scenario, ms_key) -> count

        self.logger.info(f"[all_generated_instances]: {len(all_generated_instances)}")
        target_ids = set(target_episode_ids) if target_episode_ids else None

        def _avg_score(speaker_eval_scores: dict) -> float:
            vals = [v.get("score", 0) for v in (speaker_eval_scores or {}).values()]
            return (sum(vals) / len(vals)) if vals else 0.0

        for item in all_generated_instances:
            episode_id = item.get("episode_id")
            if target_ids and episode_id not in target_ids:
                continue

            scenario = item.get("scenario")
            social_goals = item.get("social_goals", {})
            conversations = item.get("tom_exist_conversation") or []
            if not conversations:
                continue

            for conv_item in conversations:
                added_this_conv = 0
                history = conv_item.get("history", [])
                best_by_depth = conv_item.get("best_rollouts_by_depth")
                all_rollouts = conv_item.get("all_rollouts") or []

                # Sort rollouts by avg score (desc) so caps get the best items
                if self.rollout_turns == 0:
                    # random sampling test; 
                    for rollout in all_rollouts:
                        ms_text = str(rollout.get("a_mental_states", "")).strip()
                        ms_key = " ".join(ms_text.split())
                        sa, sb = rollout.get("speaker_a"), rollout.get("speaker_b")
                        pair = construct_instance_ms_uttr_separately(history, sa, sb, scenario, social_goals, rollout)
                        # print("pair:", pair)
                        # assert False
                        output_instances.extend(pair)
                    continue # do not proceed to below steps

                else:
                    scored = [(_avg_score(r.get("speaker_eval_scores")), r) for r in all_rollouts]
                    scored.sort(key=lambda x: x[0], reverse=True)

                # === 1) Add threshold-passing rollouts, respecting the proper cap logic ===
                for avg, rollout in scored:
                    if avg < score_threshold:
                        continue

                    speaker_a = rollout.get("speaker_a")
                    speaker_b = rollout.get("speaker_b")
                    ms_text = str(rollout.get("a_mental_states", "")).strip()
                    ms_key = " ".join(ms_text.split())

                    if train_type == "uttr":
                        # Use simple per-scenario cap
                        if (max_per_scenario is not None) and (added_by_scenario[scenario] >= max_per_scenario):
                            break
                        utterance_a = rollout['utterance_a']
                        if "{" in utterance_a or "}" in utterance_a:
                            # self.logger.info(f"[Skip] utterance: {utterance_a}")
                            continue
                        inst = construct_instance_uttr_only(history, speaker_a, scenario, social_goals, rollout)
                        output_instances.extend(inst)
                        added_by_scenario[scenario] += 1
                        added_this_conv += 1

                    elif train_type == "ms":
                        if (max_per_scenario is not None) and (added_by_scenario[scenario] >= max_per_scenario):
                            break
                        # Add MS-only once per identical mental state string
                        if ms_key and (ms_key not in ms_set_by_scenario[scenario]):
                            inst = construct_instance_ms_only(history, speaker_a, speaker_b, scenario, social_goals, rollout)
                            output_instances.extend(inst)
                            ms_set_by_scenario[scenario].add(ms_key)
                            added_by_scenario[scenario] += 1
                            added_this_conv += 1

                    elif train_type == "ms-uttr-sep":
                        # First, check if this MS is already present for this scenario
                        ms_already = ms_key in ms_set_by_scenario[scenario] if ms_key else False
                        utterance_a = rollout['utterance_a']
                        if "{" in utterance_a or "}" in utterance_a:
                            # self.logger.info(f"[Skip] utterance: {utterance_a}")
                            continue
                            
                        # print("[utterance_a]:", utterance_a, "[is_valid]:", is_valid_action(parse_action(utterance_a), strict=True))
                        pair = construct_instance_ms_uttr_separately(
                            history, speaker_a, speaker_b, scenario, social_goals, rollout
                        )

                        if ms_already:
                            # We can add utterances beyond the MS cap (diversify utterances).
                            # Respect per-MS utterance limit if provided.
                            key = (scenario, ms_key)
                            if (max_uttrs_per_ms is None) or (uttr_count_by_ms[key] < max_uttrs_per_ms):
                                output_instances.append(pair[-1])  # utterance only
                                uttr_count_by_ms[key] += 1
                                added_this_conv += 1
                            # else: reached per-MS utterance limit, skip
                        else:
                            # New MS candidate: only add if we haven't hit the MS-per-scenario cap.
                            if len(ms_set_by_scenario[scenario]) < max_ms_per_scenario:
                                output_instances.extend(pair)  # add MS + utterance
                                ms_set_by_scenario[scenario].add(ms_key)
                                uttr_count_by_ms[(scenario, ms_key)] += 1
                                added_this_conv += 1
                            # else: MS cap reached; do NOT add utterance-only for unseen MS
                            # (we only diversify utterances for already-added MS)

                # === 2) Per-conversation fallback when nothing passed the threshold ===
                if added_this_conv == 0:
                    # self.logger.info("No instances >= threshold in this conversation; using a fallback rollout.")
                    if train_type == "uttr":
                        best_rollout = conv_item.get("global_best_rollout")
                        if best_rollout:
                            if (max_per_scenario is None) or (added_by_scenario[scenario] < max_per_scenario):
                                inst = construct_instance_uttr_only(history, best_rollout.get("speaker_a"), scenario, social_goals, best_rollout)
                                output_instances.extend(inst)
                                added_by_scenario[scenario] += 1
                                added_this_conv += 1

                    elif train_type == "ms":
                        if best_by_depth:
                            for rollout in best_by_depth.values():
                                if (max_per_scenario is not None) and (added_by_scenario[scenario] >= max_per_scenario):
                                    break
                                sa, sb = rollout.get("speaker_a"), rollout.get("speaker_b")
                                inst = construct_instance_ms_only(history, sa, sb, scenario, social_goals, rollout)
                                output_instances.extend(inst)
                                ms_text = str(rollout.get("a_mental_states", "")).strip()
                                ms_key = " ".join(ms_text.split())
                                if ms_key:
                                    ms_set_by_scenario[scenario].add(ms_key)
                                added_by_scenario[scenario] += 1
                                added_this_conv += 1

                    elif train_type == "ms-uttr-sep":
                        # Try to use best_by_depth as a sensible fallback, but respect MS cap:
                        if best_by_depth:
                            for rollout in best_by_depth.values():
                                ms_text = str(rollout.get("a_mental_states", "")).strip()
                                ms_key = " ".join(ms_text.split())
                                sa, sb = rollout.get("speaker_a"), rollout.get("speaker_b")
                                pair = construct_instance_ms_uttr_separately(history, sa, sb, scenario, social_goals, rollout)
                                if ms_key in ms_set_by_scenario[scenario]:
                                    # we already have this MS -> utterance-only (respect per-MS utterance cap)
                                    key = (scenario, ms_key)
                                    if (max_uttrs_per_ms is None) or (uttr_count_by_ms[key] < max_uttrs_per_ms):
                                        output_instances.append(pair[-1])
                                        uttr_count_by_ms[key] += 1
                                        added_this_conv += 1
                                        break
                                else:
                                    # new MS fallback only if MS-per-scenario cap allows
                                    if len(ms_set_by_scenario[scenario]) < max_ms_per_scenario:
                                        output_instances.extend(pair)
                                        ms_set_by_scenario[scenario].add(ms_key)
                                        uttr_count_by_ms[(scenario, ms_key)] += 1
                                        added_this_conv += 1
                                        break
                            # If best_by_depth yields anything, we could optionally consider the top scored rollout.

                    else:
                        raise ValueError(f">>> train_type not supported: {train_type}")

                per_conv_added_hist.append(added_this_conv)

        self.logger.info(f"[added_by_scenario]: {len(added_by_scenario)}")
        self.logger.info(f"[ms_set_by_scenario]: {len(ms_set_by_scenario)}")
        self.logger.info(f"[uttr_count_by_ms]: {len(uttr_count_by_ms)}")
        self.logger.info(f"[instances_{train_type}]: {len(output_instances)}")
        random.shuffle(output_instances)
        return output_instances

    def generate_rollouts(self, item, scenario, social_goals):
        original_history = item["history"].copy()
        speaker_b = item["speaker_agent"]
        speaker_a = item["response_agent"]
        # tom_required = item["tom_required"]
        # depths = [0, 1]
        depths = [1]

        global_best_rollout = None
        global_best_score = -float("inf")
        best_rollouts_by_depth = {}
        all_rollouts = []

        ms_sample_num = self.ms_num
        utterance_sample_num = self.uttr_num

        def _sample_mental_states(_chat):
            return [
                SotopiaEvaluator.open_model_gen(
                    model=self.model, tokenizer=self.tokenizer_eval, chat=_chat,
                    do_sample=True, temperature=0.7, max_new=512, top_p=0.9)
                for _ in range(ms_sample_num)
            ]

        def _sample_utterance(ms_text, history):
            _chat = [{
                "role": "user",
                "content": prompt_utterance_with_ms(
                    history=history,
                    speaker=speaker_a,
                    ms_text=ms_text,
                    scenario=scenario,
                    social_goal=social_goals[speaker_a],
                    turn_number=len(history)+1,
                )
            }]
            sampled = []
            for _ in range(utterance_sample_num+1):
                _uttr = SotopiaEvaluator.open_model_gen(
                    model=self.model, tokenizer=self.tokenizer_eval, chat=_chat,
                    do_sample=True, temperature=0.7, max_new=200, top_p=0.9)
                parsed = parse_action(_uttr)
                # _uttr = construct_utterance(parsed, speaker_a)
                if not construct_utterance(parsed, speaker_a):
                    continue
                sampled.append(parsed)
                if len(sampled) == utterance_sample_num:
                    break

            return sampled

        def _is_terminal_rendered(text: str) -> bool:
            """
            Fast check for the already-rendered utterance string produced by construct_utterance(...).
            Matches the two terminal renderings:
            - "... did nothing"
            - "... left the conversation"
            """
            t = text.strip().lower()
            return t.endswith("did nothing") or t.endswith("left the conversation")

        def _run_simulation(uttr_a, history):
            history_updated = history.copy()
            history_updated.append(uttr_a)

            # If seeded A utterance is already terminal, stop immediately.
            if _is_terminal_rendered(uttr_a):
                return history_updated

            for _ in range(self.rollout_turns):  # Two full turns
                for model, tokenizer, speaker in [(self.simulator, self.simulator_tokenizer, speaker_b),
                                                  (self.model, self.tokenizer_eval, speaker_a)]:
                    _chat = [{
                        "role": "user",
                        "content": prompt_utterance_without_ms(
                            history_updated, speaker, scenario, social_goals[speaker],
                            turn_number=len(history_updated) + 1,
                        )
                    }]
                    utterance = SotopiaEvaluator.open_model_gen(
                        model=model, tokenizer=tokenizer, chat=_chat,
                        do_sample=True, temperature=0.7, max_new=200, top_p=0.9)
                    parsed = parse_action(utterance)
                    utterance = construct_utterance(parsed, speaker)
                    history_updated.append(utterance)

                    # Stop if this agent did nothing or left.
                    if parsed.action_type in {"none", "leave"} or _is_terminal_rendered(utterance):
                        return history_updated
            return history_updated

        def _parse_score(response: str) -> int:
            response = response.replace("```json", "").replace("`", "").strip()
            try:
                data = json.loads(response)
                return int(data["score"]) if isinstance(data, dict) and "score" in data else -1
            except Exception as e:
                logging.info(e)
            match = re.search(r'"score"\s*:\s*(\d+)', response)
            return int(match.group(1)) if match else -1

        def _evaluate_goal_achievement(history):
            scores = {}
            for speaker in [speaker_a, speaker_b]:
                prompt = GOAL_EVAL_PROMPT.format(
                    scenario=scenario, agent=speaker,
                    social_goal=social_goals[speaker],
                    history="\n".join(history)
                )
                response = self.call_gemini(prompt, max_gen_len=500, temperature=1.0)
                score = _parse_score(response)
                scores[speaker] = {
                    "score": score if score != -1 else 5,
                    "reasoning": response
                }
            _avg_score = (scores[speaker_a]["score"] + scores[speaker_b]["score"]) / 2
            return scores, _avg_score
        
        tmp_rollouts = []
        for d in depths:
            local_best_rollout = None
            local_best_score = -float("inf")

            chat = [{
                "role": "user",
                "content": ms_prompt(
                    history=original_history,
                    depth=d,
                    person=speaker_a,
                    another_person=speaker_b,
                    scenario=scenario,
                    social_goal=social_goals[speaker_a]
                )
            }]
            sampled_ms = _sample_mental_states(chat)
            for ms in sampled_ms:
                # ms = _ms if d == 0 else " ".join(random.sample([
                #     best_rollouts_by_depth[0]["a_mental_states"], _ms
                # ], k=2))

                for parsed_utter in _sample_utterance(ms, original_history):
                    uttr = construct_utterance(parsed_utter, speaker_a)
                    if self.rollout_turns > 0:
                        updated_history = _run_simulation(uttr, original_history)
                        speaker_eval_scores, avg_score = _evaluate_goal_achievement(updated_history)

                        rollout = {
                            "depth": d,
                            "speaker_a": speaker_a,
                            "speaker_b": speaker_b,
                            "a_mental_states": ms,
                            "action_type": parsed_utter.action_type,
                            "argument": parsed_utter.argument,
                            "utterance_a": uttr,
                            "updated_history": updated_history,
                            "speaker_eval_scores": speaker_eval_scores,
                        }

                        all_rollouts.append(rollout)

                        if avg_score > local_best_score:
                            local_best_score = avg_score
                            local_best_rollout = rollout

                    else:
                        rollout = {
                            "depth": d,
                            "speaker_a": speaker_a,
                            "speaker_b": speaker_b,
                            "a_mental_states": ms,
                            "action_type": parsed_utter.action_type,
                            "argument": parsed_utter.argument,
                            "utterance_a": uttr,
                            "updated_history": None,
                            "speaker_eval_scores": None,
                        }
                        tmp_rollouts.append(rollout)

            if self.rollout_turns == 0 and tmp_rollouts:
                random_rollouts = random.sample(tmp_rollouts, 1)
                print("[random_rollouts]:", len(random_rollouts))
                all_rollouts.extend(random_rollouts)

            best_rollouts_by_depth[d] = local_best_rollout
            if local_best_score > global_best_score:
                global_best_score = local_best_score
                global_best_rollout = local_best_rollout

        return global_best_rollout, best_rollouts_by_depth, all_rollouts

    def generate_rollouts_without_ms(self, item, scenario, social_goals):
        original_history = item["history"].copy()
        speaker_b = item["speaker_agent"]
        speaker_a = item["response_agent"]
        # tom_required = item["tom_required"]

        best_rollout = None
        best_score = -float("inf")
        all_rollouts = []

        utterance_sample_num = 2
        def _sample_utterance(history):
            _chat = [{
                "role": "user",
                "content": prompt_utterance_without_ms(
                    history=history,
                    speaker=speaker_a,
                    scenario=scenario,
                    social_goal=social_goals[speaker_a],
                    turn_number=len(history)+1,
                )
            }]
            sampled = []
            for _ in range(utterance_sample_num+1):
                _uttr = SotopiaEvaluator.open_model_gen(
                    model=self.model, tokenizer=self.tokenizer_eval, chat=_chat,
                    do_sample=True, temperature=0.7, max_new=200, top_p=0.9)
                parsed = parse_action(_uttr)
                # _uttr = construct_utterance(parsed, speaker_a)
                if not construct_utterance(parsed, speaker_a):
                    continue
                sampled.append(parsed)
                if len(sampled) == utterance_sample_num:
                    break

            return sampled

        def _is_terminal_rendered(text: str) -> bool:
            """
            Fast check for the already-rendered utterance string produced by construct_utterance(...).
            Matches the two terminal renderings:
            - "... did nothing"
            - "... left the conversation"
            """
            t = text.strip().lower()
            return t.endswith("did nothing") or t.endswith("left the conversation")

        def _run_simulation(uttr_a, history):
            history_updated = history.copy()
            history_updated.append(uttr_a)

            # If seeded A utterance is already terminal, stop immediately.
            if _is_terminal_rendered(uttr_a):
                return history_updated

            for _ in range(self.rollout_turns):  # Two full turns
                for model, speaker in [(self.simulator, speaker_b), (self.model, speaker_a)]:
                    _chat = [{
                        "role": "user",
                        "content": prompt_utterance_without_ms(
                            history_updated, speaker, scenario, social_goals[speaker],
                            turn_number=len(history_updated) + 1,
                        )
                    }]
                    utterance = SotopiaEvaluator.open_model_gen(
                        model=model, tokenizer=self.tokenizer_eval, chat=_chat,
                        do_sample=True, temperature=0.7, max_new=200, top_p=0.9)
                    parsed = parse_action(utterance)
                    utterance = construct_utterance(parsed, speaker)
                    history_updated.append(utterance)

                    # Stop if this agent did nothing or left.
                    if parsed.action_type in {"none", "leave"} or _is_terminal_rendered(utterance):
                        return history_updated

            return history_updated

        def _parse_score(response: str) -> int:
            response = response.replace("```json", "").replace("`", "").strip()
            try:
                data = json.loads(response)
                return int(data["score"]) if isinstance(data, dict) and "score" in data else -1
            except Exception as e:
                logging.info(e)
            match = re.search(r'"score"\s*:\s*(\d+)', response)
            return int(match.group(1)) if match else -1

        def _evaluate_goal_achievement(history):
            scores = {}
            for speaker in [speaker_a, speaker_b]:
                prompt = GOAL_EVAL_PROMPT.format(
                    scenario=scenario, agent=speaker,
                    social_goal=social_goals[speaker],
                    history="\n".join(history)
                )
                response = self.call_gemini(prompt, max_gen_len=500, temperature=1.0)
                score = _parse_score(response)
                scores[speaker] = {
                    "score": score if score != -1 else 5,
                    "reasoning": response
                }
            _avg_score = (scores[speaker_a]["score"] + scores[speaker_b]["score"]) / 2
            return scores, _avg_score

        for parsed_utter in _sample_utterance(original_history):
            uttr = construct_utterance(parsed_utter, speaker_a)
            updated_history = _run_simulation(uttr, original_history)
            speaker_eval_scores, avg_score = _evaluate_goal_achievement(updated_history)

            rollout = {
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "action_type": parsed_utter.action_type,
                "argument": parsed_utter.argument,
                "utterance_a": uttr,
                "updated_history": updated_history,
                "speaker_eval_scores": speaker_eval_scores,
            }

            all_rollouts.append(rollout)

            if avg_score > best_score:
                best_score = avg_score
                best_rollout = rollout

        return best_rollout, all_rollouts

    def process_episodes(
            self,
            episodes,
            max_num=-1,
            target_episode_ids: Optional[set] = None,
            processed_scenarios: Optional[set] = None,
            processed_episodes: Optional[set] = None,
            existing_instances: Optional[list] = None,
            save_path="",
    ) -> list:
        target_episode_ids = set() if target_episode_ids is None else set(target_episode_ids)
        processed_scenarios = set() if processed_scenarios is None else set(processed_scenarios)
        processed_episodes = set() if processed_episodes is None else set(processed_episodes)
        existing_instances = [] if existing_instances is None else list(existing_instances)

        generated_instances = existing_instances.copy()
        self.logger.info(f">>> Processing Episodes... [max_num = {max_num}]")
        for episode in tqdm.tqdm(episodes):
            eid = episode["episode_id"]
            target_episode_ids = set(target_episode_ids) if target_episode_ids else set()
            if target_episode_ids and eid not in target_episode_ids:
                continue
            else:
                # if there are no target_episode_ids, this means we don't need to generate new data
                # (usually for outer_ep=0 resume data generation)
                if eid in processed_episodes:
                    self.logger.info(f"[Skip] episode_id {eid}")
                    continue
                scenario = episode["scenario"]
                if scenario in processed_scenarios:
                    self.logger.info(f"[Skip] scenario {scenario}")
                    continue

            conversations = episode["conversation"]
            if not conversations:
                continue
            processed_scenarios.add(scenario)
            _conversations = [x for x in conversations if x['turn_id'] < 5] # sample conversations when turn_id is less than 5
            tom_convs = random.sample(_conversations, min(1, len(_conversations)))
            self.logger.info(f"[Process] episode_id {eid}, selected turns: {len(tom_convs)}")

            episode_record = {
                "episode_id": eid,
                "scenario": episode["scenario"],
                "social_goals": episode["social_goals"],
                "agents_background": episode["agents_background"],
                "conversation": conversations,
                "tom_exist_conversation": []
            }

            for conv in tom_convs:
                if self.train_type == "uttr":
                    global_best, all_rollouts = self.generate_rollouts_without_ms(
                        item=conv,
                        scenario=episode["scenario"],
                        social_goals=episode["social_goals"]
                    )
                    conv.update({
                        "global_best_rollout": global_best,
                        "all_rollouts": all_rollouts
                    })
                else:
                    global_best, best_by_depth, all_rollouts = self.generate_rollouts(
                        item=conv,
                        scenario=episode["scenario"],
                        social_goals=episode["social_goals"]
                    )
                    conv.update({
                        "best_rollouts_by_depth": best_by_depth,
                        "global_best_rollout": global_best,
                        "all_rollouts": all_rollouts
                    })
                episode_record["tom_exist_conversation"].append(conv)

            generated_instances.append(episode_record)
            self.save_json(save_path, generated_instances)
            
            if 0 < max_num == len(generated_instances): # stop generating
                break

        self.logger.info("done!")
        return generated_instances

    def load_data_for_star_simulation(
            self,
            data_path: str = "data/gemini-tom-conversation-sotopia-pi-episodes.json"
    ):
        # === Load Input Data ===
        sample_filepath = os.path.join(self.project_root_dir, data_path)
        logging.info(f"[sample_filepath]: {sample_filepath}")
        assert os.path.isfile(sample_filepath)
        with open(sample_filepath, "r", encoding="utf-8") as f:
            sample_data = json.load(f)

        if self.do_eval:
            raise NotImplementedError

        self.train_data = sample_data[:-1] if self.do_eval else sample_data
        self.eval_data = [sample_data[-1]] if self.do_eval else []

        # Total instances used to build training instances per outer epoch
        init_num_instance_for_train = 150 if self.total_finetune_epochs > 1 else 1000
        # self.epoch_new_data = int(init_num_instance_for_train * (self.epoch_new_data / 100))
        # self.logger.info(f"epoch_new_data: {self.epoch_new_data}")
        self.logger.info(f">>> # of epoch_new_data: {int(init_num_instance_for_train * (self.epoch_new_data / 100))}")

        # ----------------- helpers -----------------
        def base_prefix():
            return f"data/all_rollouts_{self.model_name}{'_uttr' if self.train_type == 'uttr' else ''}_ms{self.ms_num}_uttr{self.uttr_num}_turn{self.rollout_turns}"

        def delta_path_for_epoch(ep: int) -> str:
            # per-epoch NEW data only
            # return f"{base_prefix()}-ranking_{self.do_sample_ranking}-new_data_{self.epoch_new_data}-ep{ep}-lor{self.lora_rank}-loa{self.lora_alpha}_delta.json"
            return f"{base_prefix()}-ranking_{self.do_sample_ranking}_delta.json"

        def mix_path_for_epoch(ep: int) -> str:
            # the mixed pool actually used in training for this epoch
            return f"{base_prefix()}-ranking_{self.do_sample_ranking}_mix.json"

        def allocate_counts(total: int, epoch: int) -> dict:
            """
            Return {ep -> count} that sums to `total`:
            - if epoch <= 10: w0 = 1 - 0.1*epoch, wi=0.1 for i=1...epoch
            - else: epochs (epoch-9...epoch) each get 0.1
            (It's OK if we end up using fewer than these counts due to shortages.)
            """
            weights = {}
            if epoch <= 10:
                w0 = max(0.0, 1.0 - 0.1 * epoch)
                if w0 > 0: weights[0] = w0
                for i in range(1, epoch + 1):
                    weights[i] = 0.1
            else:
                for i in range(epoch - 9, epoch + 1):
                    weights[i] = 0.1

            raw = {ep: weights[ep] * total for ep in weights}
            counts = {ep: int(math.floor(raw[ep])) for ep in raw}
            remainder = total - sum(counts.values())
            if remainder > 0:
                # Give leftovers to the largest fractions, tie-break toward newer epochs
                fracs = sorted([(raw[ep] - math.floor(raw[ep]), ep) for ep in raw],
                               key=lambda x: (x[0], x[1]), reverse=True)
                for _, ep in fracs[:remainder]:
                    counts[ep] += 1
            return counts

        def allocate_counts_new(total: int, epoch: int, epoch_new_data: int = 10) -> dict:
            """
            Return {ep -> count} that sums to `total`
            """
            assert isinstance(total, int) and total > 0
            assert isinstance(epoch, int) and epoch >= 0
            if epoch == 0:  # Use all the data we have for the first epoch
                return {0: total}

            ep2cnt = dict()
            assert isinstance(epoch_new_data, int) and 0 <= epoch_new_data <= 100
            if epoch_new_data == 0:  # the new data is not used
                quotient, remainder = divmod(total, epoch)
                for ep_idx in range(epoch):  # equally use the old data
                    ep2cnt[ep_idx] = quotient
                if remainder > 0:  # assign the leftover data to the most recent epoch (but not the current epoch)
                    ep2cnt[epoch - 1] += remainder
            elif epoch_new_data == 100:  # only use the new data
                return {epoch: total}
            else:
                # 1. For the current epoch, we use `epoch_new_data` percent of new data.
                new_ratio = float(epoch_new_data / 100)
                assert 0.0 <= new_ratio <= 1.0
                num_new_data = int(total * new_ratio)
                ep2cnt[epoch] = num_new_data

                # 2. As for the old data, we use them equally.
                num_old_data = total - num_new_data
                quotient, remainder = divmod(num_old_data, epoch)
                for ep_idx in range(epoch):  # equally assign the old data
                    ep2cnt[ep_idx] = quotient

                # 3. If there is leftover amount, use new data.
                if remainder > 0:  # assign the leftover data to the current epoch
                    ep2cnt[epoch] += remainder

            return ep2cnt

        def safe_sample(items: list, k: int) -> list:
            # if k <= 0 or not items:
            #     return []
            # return random.sample(items, min(k, len(items)))

            self.logger.info(f">>> safe_sample -- len(items) = {len(items)}; k = {k}; "
                             f"do_sample_ranking = {self.do_sample_ranking}")
            if isinstance(k, int) and isinstance(items, list) and len(items) > 0:
                # random.sample can not take k that is larger than len(items) or is negative, otherwise ValueError
                if k <= 0:
                    return []
                elif k >= len(items):
                    seen_episode = set()
                    seen_scenario = set()
                    out = []
                    for x in items:
                        eid = x.get("episode_id")
                        if eid in seen_episode:
                            continue
                        seen_episode.add(eid)

                        scneario = x.get("scenario")
                        if scneario in seen_scenario:
                            continue
                        seen_scenario.add(scneario)
                        out.append(x)
                    return out
                else:
                    if self.do_sample_ranking:
                        # Pick the top samples with highest eval score instead of random ones
                        items_with_scores = []
                        scores_only = []
                        for item in items:  # Note: `item` is `episode_record` in `process_episodes()`
                            item_best_avg_scores = []  # containing the best rollout score from all conv of this item

                            assert isinstance(item, dict) and "tom_exist_conversation" in item, item
                            tom_exist_conversation = item["tom_exist_conversation"]
                            assert isinstance(tom_exist_conversation, list), type(tom_exist_conversation)
                            for conv in tom_exist_conversation:  # deal with each conv
                                assert isinstance(conv, dict) and "global_best_rollout" in conv, type(conv)
                                global_best_rollout = conv["global_best_rollout"]  # get the best rollout
                                assert (isinstance(global_best_rollout, dict) and
                                        "speaker_eval_scores" in global_best_rollout), type(global_best_rollout)
                                speaker_eval_scores = global_best_rollout["speaker_eval_scores"]
                                assert (isinstance(speaker_eval_scores, dict) and
                                        len(speaker_eval_scores) > 0), type(speaker_eval_scores)
                                conv_scores = [float(eval_dict["score"]) for agent_name, eval_dict in
                                               speaker_eval_scores.items()]
                                cur_best_avg_score = float(np.mean(conv_scores).item())
                                item_best_avg_scores.append(cur_best_avg_score)  # avg score over the two agents

                            item_best_avg_score = float(np.mean(item_best_avg_scores).item())
                            assert 0.0 <= item_best_avg_score <= 10.0, item_best_avg_score
                            scores_only.append(item_best_avg_score)
                            items_with_scores.append((item, item_best_avg_score))  # item paired with its score

                        # Statistics of all scores
                        scores_max, scores_min = float(np.max(scores_only)), float(np.min(scores_only))
                        scores_avg, scores_std = float(np.mean(scores_only)), float(np.std(scores_only))
                        self.logger.info(f">>> do_sample_ranking -- statistics of all scores: "
                                         f"max = {scores_max:.2f}; min = {scores_min:.2f}; "
                                         f"avg = {scores_avg:.2f}; std = {scores_std:.2f}")

                        # Ranking/sorting
                        items_with_scores.sort(key=lambda x: x[1], reverse=True)
                        # Pick the top items
                        top_items_with_scores = items_with_scores[:min(k, len(items))]
                        top_items = [_item for _item, _score in top_items_with_scores]
                        top_scores = [_score for _item, _score in top_items_with_scores]

                        # Statistics of the top scores
                        top_scores_max, top_scores_min = float(np.max(top_scores)), float(np.min(top_scores))
                        top_scores_avg, top_scores_std = float(np.mean(top_scores)), float(np.std(top_scores))
                        self.logger.info(f">>> do_sample_ranking -- statistics of top scores: "
                                         f"max = {top_scores_max:.2f}; min = {top_scores_min:.2f}; "
                                         f"avg = {top_scores_avg:.2f}; std = {top_scores_std:.2f}")

                        return top_items
                    else:
                        return random.sample(items, min(k, len(items)))
            else:
                return []

        def _dedupe_by_episode(items):
            seen = set()
            out = []
            for x in items:
                eid = x.get("episode_id")
                if eid in seen:
                    continue
                seen.add(eid)
                out.append(x)
            return out

        # --------------- main flow ----------------
        self.logger.info(f">>> load_data_for_star_simulation [outer_epoch_num]: {self.outer_epoch_num}")
        if self.outer_epoch_num == 0:
            # === Prepare initial pool ===
            # "all" rollouts pool used at epoch 0 before sampling
            save_all = f"{base_prefix()}.json"

            output_instances = []
            all_generated_instances = self.load_json(save_all, default=[])
            all_generated_instances = _dedupe_by_episode(all_generated_instances)
            processed_episodes = {item['episode_id'] for item in all_generated_instances}
            processed_scenarios = {item.get('scenario') for item in all_generated_instances}

            self.logger.info(f"[Resume] SFT instances: {len(output_instances)}, "
                             f"full rollouts(all): {len(all_generated_instances)}")

            # Build/expand the big pool if too small
            if not all_generated_instances or len(all_generated_instances) < init_num_instance_for_train:
                all_generated_instances = self.process_episodes(
                    self.train_data, max_num=init_num_instance_for_train, target_episode_ids=set(),
                    processed_episodes=set(processed_episodes), processed_scenarios=set(processed_scenarios),
                    existing_instances=all_generated_instances, save_path=save_all
                )

            # Sample epoch-0 pool and persist as BOTH delta(0) and mix(0)
            ep0_pool = safe_sample(all_generated_instances, init_num_instance_for_train)
            ep0_pool = _dedupe_by_episode(ep0_pool)
            self.logger.info(f"[ep0 pool]: {len(ep0_pool)}")
            
            if self.total_finetune_epochs > 1:
                self.save_json(delta_path_for_epoch(0), ep0_pool)   # new data for ep0 == the initial pool
                self.save_json(mix_path_for_epoch(0), ep0_pool)     # training mix used at ep0

            # Build training instances from epoch 0 mix
            target_episode_ids = [x["episode_id"] for x in ep0_pool]
            output_instances = self.generate_training_instances(
                all_generated_instances=ep0_pool,
                target_episode_ids=target_episode_ids, train_type=self.train_type,
            )
            unsloth.FastLanguageModel.for_training(self.model)
            self.logger.info(f"[output_instances at epoch 0]: {len(output_instances)}")
            self.save_json(f"data/train_instances_{self.model_name}_{self.train_type}_ms{self.ms_num}_uttr{self.uttr_num}_turn{self.rollout_turns}_ep0.json", output_instances)
            return output_instances, []

        else:
            e = self.outer_epoch_num
            # counts = allocate_counts(init_num_instance_for_train, e)
            counts = allocate_counts_new(init_num_instance_for_train, e, epoch_new_data=self.epoch_new_data)
            self.logger.info(f"[mix plan for epoch {e}] " +
                             ", ".join([f"ep{ep}:{counts[ep]}" for ep in sorted(counts.keys())]))

            # ---- 1) Ensure/Resume NEW data for current epoch e into delta(e); no fallback if short ----
            new_count = counts.get(e, 0)
            new_delta_path = delta_path_for_epoch(e)
            existing_new = self.load_json(new_delta_path, default=[])
            existing_new = _dedupe_by_episode(existing_new)
            existing_episode_ids = {x["episode_id"] for x in existing_new}
            existing_scenarios = {x.get("scenario") for x in existing_new if "scenario" in x}

            if len(existing_new) < new_count and new_count > 0:
                need = new_count - len(existing_new)
                all_episode_ids = [x["episode_id"] for x in self.train_data]
                remaining_candidates = [eid for eid in all_episode_ids if eid not in existing_episode_ids]
                to_draw = min(need, len(remaining_candidates))
                target_episode_ids = random.sample(remaining_candidates, to_draw) if to_draw > 0 else []

                newly_created = self.process_episodes(
                    self.train_data,
                    max_num=len(target_episode_ids),
                    target_episode_ids=set(target_episode_ids),
                    processed_episodes=set(existing_episode_ids),
                    processed_scenarios=set(existing_scenarios),
                    existing_instances=existing_new,   # append in place
                    save_path=new_delta_path
                )
                existing_new = _dedupe_by_episode(existing_new + newly_created)

            # This epoch's new slice (maybe fewer than planned if we're short)
            new_generated_instances = safe_sample(existing_new, new_count) if new_count > 0 else []

            # ---- 2) Pull prior epochs strictly from their delta files, newest -> oldest ----
            prior_picks = []
            for ep in sorted([k for k in counts.keys() if k < e], reverse=True):  # descending
                want = counts[ep]
                if want <= 0:
                    continue
                # Prefer delta; fall back to mix if delta missing (back-compat)
                src = self.load_json(delta_path_for_epoch(ep), default=None)
                if src is None:
                    src = self.load_json(mix_path_for_epoch(ep), default=[])
                if not src:
                    self.logger.info(f"[warn] No per-epoch data found for ep{ep} (delta/mix).")
                    continue
                src = _dedupe_by_episode(src)
                prior_picks.extend(safe_sample(src, want))  # may yield fewer than want
                if len(prior_picks) >= (init_num_instance_for_train - len(new_generated_instances)):
                    break

            # ---- 3) Combine (no top-ups) ----
            combined = _dedupe_by_episode(new_generated_instances + prior_picks)
            random.shuffle(combined)

            self.logger.info(f"[mixed pool at epoch {e}]: {len(combined)}")

            # Save the mixed pool separately; DO NOT overwrite delta(e)
            self.save_json(mix_path_for_epoch(e), combined)

            # ---- 4) Build training instances from the mixed pool ----
            output_instances = self.generate_training_instances(
                all_generated_instances=combined, target_episode_ids=[], train_type=self.train_type,
            )
            unsloth.FastLanguageModel.for_training(self.model)
            self.logger.info(f"[output_instances at epoch {e}]: {len(output_instances)}")
            self.save_json(f"data/train_instances_{self.model_name}_{self.train_type}_ep{e}.json", output_instances)
            return output_instances, []

    def load_data(
            self,
            training_strategy: str = "sft",
            do_data_stat: bool = False
    ):
        match training_strategy:
            case "sft":
                # dataset = load_dataset("stanfordnlp/imdb", split="train", cache_dir=self.cache_dir)
                # dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train", cache_dir=self.cache_dir)
                # data_tom = self.load_data_for_sft(do_data_stat=do_data_stat)
                data_tom = self.load_data_for_star_simulation()
            case _:
                raise ValueError(f">>> `training_strategy` {training_strategy} not supported.")

        self.ds = data_tom
        return data_tom
