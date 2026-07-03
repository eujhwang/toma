# -*- coding: utf-8 -*-

import os
import json
# import random
# import pprint
from typing import Optional
# import collections
# from datasets import load_dataset


class DatasetLoader:

    def __init__(
            self,
            dataset_name: str,
            cache_dir: Optional[str] = None,
            data_dir: str = ".",
            eval_mode: str = "all",
    ):
        self.dataset_name = dataset_name
        self.eval_mode = eval_mode
        self.home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.home_dir, ".cache/huggingface")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        self.data_dir = data_dir
        print(f">>> dataset_name: {self.dataset_name}; cache_dir: {self.cache_dir}; data_dir: {data_dir}")

        self.ds = None

    def load_sotopia_episodes(self):
        # env_id_fp = "data/sotopia/env_ids.txt"
        print("[evaluation_mode]:", self.eval_mode)
        if self.eval_mode == "hard":
            data_fp = os.path.join(self.data_dir, "data/sotopia/env_pairs_dump_sotopia_hard.json")  # 70
        else:
            data_fp = os.path.join(self.data_dir, "data/sotopia/env_pairs_dump_sotopia_all.json")  # 450
        assert os.path.isfile(data_fp), f">>> {data_fp} does not exist!"

        data = json.load(open(data_fp, "r+"))
        out = []
        for item in data:
            environment_id = item['environment_id']
            scenario = item['scenario']
            relationship = item['relationship']
            # ['environment_id', 'relationship', 'scenario', 'pair_count', 'pairs']
            agent_pairs = item['pairs']
            for pair in agent_pairs:
                agents = pair['agents']
                cur_item = {
                    "environment_id": environment_id,
                    "agents": agents,
                    "scenario": scenario,
                    "relationship": relationship,
                }
                out.append(cur_item)
        return out

    def load_data(self):
        if self.dataset_name == "sotopia-episodes":
            data_tom = self.load_sotopia_episodes()
        # elif self.dataset_name == "sotopia-pi-episodes":
        #     data_tom = self.load_sotopia_pi_episodes()
        # elif self.dataset_name == "tombench":
        #     data_tom = self.load_tombench()
        else:
            raise ValueError(f">>> Dataset {self.dataset_name} not supported")

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
