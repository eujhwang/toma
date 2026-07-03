#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import json
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, Optional

import fire
import torch
import unsloth
import wandb
import yaml
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer

from utils.data_sotopia import SampleDataloader
from utils.init_functions import cuda_setup, logger_setup, random_setup
from utils.models import OPEN_MODEL_HF, PROPRIETARY_LLM

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


class TomTraining:
    def __init__(
            self,
            verbose: bool,
            logger,
            seed: int,
            cuda_dict: dict,
            project_root_dir: str,
            ckpt_dir: str,
            cache_dir: Optional[str] = None,
            config_dir: Optional[str] = "config/default",
            model_name: str = "qwen2.5-3b",
            simulator_name: str = "qwen2.5-14b-unsloth-4bit",
            run_id: str = "default_run",
            do_eval: bool = False,
            max_seq_len: Optional[int] = 4096,
            use_wandb: bool = False,
            debug: bool = False,
            train_type: str = "ms-uttr-sep",
            wandb_config: Optional[Dict[str, Any]] = None,
            lora_args: Optional[Dict[str, Any]] = None,
            finetune_epochs: int = 1,
            do_sample_ranking: bool = False,
            epoch_new_data: int = 100,
            no_re_initialize: bool = False,
            ms_num: int = 2,
            uttr_num: int = 2,
            rollout_turns: int = 0,
    ):
        if do_eval:
            raise ValueError("Evaluation is not supported in the released Sotopia SFT training path.")
        if not os.path.isdir(project_root_dir):
            raise ValueError(f"project_root_dir does not exist: {project_root_dir}")
        if not os.path.isdir(ckpt_dir):
            raise ValueError(f"ckpt_dir does not exist: {ckpt_dir}")

        self.verbose = verbose
        self.logger = logger
        self.seed = seed
        self.cuda_dict = cuda_dict
        self.project_root_dir = project_root_dir
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.simulator_name = simulator_name
        self.run_id = run_id
        self.use_wandb = use_wandb
        self.debug = debug
        self.train_type = train_type
        self.do_eval = False
        self.max_seq_len = max_seq_len
        self.finetune_epochs = max(1, int(finetune_epochs))
        self.do_sample_ranking = do_sample_ranking
        self.epoch_new_data = epoch_new_data
        self.no_re_initialize = no_re_initialize
        self.ms_num = ms_num
        self.uttr_num = uttr_num
        self.rollout_turns = rollout_turns
        self.wandb_config = wandb_config or {}
        self.lora_args = lora_args or {
            "lora_rank": 64,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
        }
        self._base_seed = int(seed)
        self._base_ckpt_root = ckpt_dir

        self.common_training_args: Dict[str, Any] = {}
        self.sft_trainer_args: Dict[str, Any] = {}
        self._load_training_config(config_dir)
        self._apply_preinit_overrides_from_wandb()

        self.cache_dir = self._resolve_cache_dir(cache_dir)
        os.environ["HF_HOME"] = self.cache_dir
        if self.verbose:
            self.logger.info(f">>> cache_dir: {self.cache_dir}")

        self.open_models = OPEN_MODEL_HF
        self.open_models_local = {
            k: (
                os.path.join(self.cache_dir, "models--" + "--".join(v.split("/")), "snapshots/model")
                if not os.path.exists("/model-weights/" + v.split("/")[-1])
                else "/model-weights/" + v.split("/")[-1]
            )
            for k, v in self.open_models.items()
        }
        self.logger.info(f">>> self.open_models_local: {self.open_models_local}")
        self.proprietary_models = PROPRIETARY_LLM

        self.tokenizer_train = self.initialize_tokenizer(
            model_name=self.model_name,
            padding_side="right",
            truncation_side="right",
        )
        self.tokenizer_eval = self.initialize_tokenizer(
            model_name=self.model_name,
            padding_side="left",
            truncation_side="left",
        )
        self._set_max_seq_len()

        self.model, self.tokenizer, self.simulator, self.simulator_tokenizer = self.initialize_model_simulator(
            model_name=self.model_name,
            simulator_name=self.simulator_name,
        )
        self.finetuned_model = None

    def _load_training_config(self, config_dir: Optional[str]) -> None:
        if not (isinstance(config_dir, str) and os.path.isdir(config_dir)):
            return

        config_files = {
            "common_training_args": self.common_training_args,
            "sft_trainer_args": self.sft_trainer_args,
        }
        for name, target in config_files.items():
            config_path = os.path.join(config_dir, f"{name}.yaml")
            if os.path.isfile(config_path):
                with open(config_path, "r", encoding="utf-8") as fp_in:
                    loaded = yaml.load(fp_in, Loader=yaml.FullLoader)
                    if isinstance(loaded, dict):
                        target.update(loaded)

    def _resolve_cache_dir(self, cache_dir: Optional[str]) -> str:
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            return cache_dir

        cache_dir = os.path.join(self.project_root_dir, ".cache/huggingface/")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _apply_preinit_overrides_from_wandb(self) -> None:
        if not self.wandb_config:
            return

        for key in ["model_name", "simulator_name", "train_type", "max_seq_len", "debug"]:
            if self.wandb_config.get(key) is not None:
                setattr(self, key, self.wandb_config[key])

        for key in ["lora_rank", "lora_alpha", "lora_dropout"]:
            if self.wandb_config.get(key) is not None:
                self.lora_args[key] = self.wandb_config[key]

    def _apply_training_overrides_from_wandb(self) -> None:
        if not self.wandb_config:
            return

        trainarg_keys = {
            "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
            "weight_decay", "num_train_epochs", "warmup_steps", "warmup_ratio",
            "lr_scheduler_type", "logging_steps", "save_steps", "save_strategy", "bf16",
            "fp16", "max_grad_norm", "max_steps",
        }
        for key in trainarg_keys:
            if self.wandb_config.get(key) is not None:
                self.common_training_args[key] = self.wandb_config[key]

        common_args = self.wandb_config.get("common_training_args")
        if isinstance(common_args, dict):
            self.common_training_args.update(common_args)

        sft_args = self.wandb_config.get("sft_trainer_args")
        if isinstance(sft_args, dict):
            self.sft_trainer_args.update(sft_args)

        for key in ["lora_rank", "lora_alpha", "lora_dropout"]:
            if self.wandb_config.get(key) is not None:
                self.lora_args[key] = self.wandb_config[key]

    def _set_max_seq_len(self) -> None:
        max_len = min(
            self.tokenizer_train.model_max_length,
            self.tokenizer_eval.model_max_length,
            self.tokenizer_train.max_len_single_sentence,
            self.tokenizer_eval.max_len_single_sentence,
        )
        self.max_seq_len = max_len if self.max_seq_len is None or self.max_seq_len <= 0 else min(self.max_seq_len, max_len)
        if self.verbose:
            self.logger.info(f">>> len(tokenizer_train.vocab) = {len(self.tokenizer_train.vocab)}")
            self.logger.info(f">>> len(tokenizer_eval.vocab) = {len(self.tokenizer_eval.vocab)}")
            self.logger.info(f">>> tokenizer.max_len_single_sentence = {max_len}")
            self.logger.info(f"max_seq_len = {self.max_seq_len}")

    def _get_model_path(self, model_name: str) -> str:
        if model_name not in self.open_models_local:
            raise ValueError(f"Unsupported model_name: {model_name}")

        model_path_local = self.open_models_local[model_name]
        if os.path.isdir(model_path_local):
            return model_path_local
        if model_name in self.open_models:
            return self.open_models[model_name]
        raise ValueError(f"Unsupported model_name: {model_name}")

    def initialize_tokenizer(
            self,
            model_name: str,
            padding_side: str = "left",
            truncation_side: str = "left",
    ):
        model_path = self._get_model_path(model_name)
        self.logger.info(f">>> Loading tokenizer: model_path = {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=self.cache_dir,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def initialize_model_simulator(self, model_name: str, simulator_name: str):
        model_path = self._get_model_path(model_name)
        simulator_path = self._get_model_path(simulator_name)
        self.logger.info(f">>> model_path: {model_path}")
        self.logger.info(f">>> simulator_path: {simulator_path}")

        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_path,
            device_map="cuda:0" if torch.cuda.device_count() > 1 else "auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            load_in_4bit=False,
            dtype="bfloat16",
            max_seq_length=self.max_seq_len,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_disable()
        unsloth.FastLanguageModel.for_training(model)

        simulator, simulator_tokenizer = unsloth.FastLanguageModel.from_pretrained(
            simulator_path,
            device_map="cuda:1" if torch.cuda.device_count() > 1 else "auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            load_in_4bit=True,
            dtype="bfloat16",
        )
        simulator.config.use_cache = False
        simulator.gradient_checkpointing_disable()
        unsloth.FastLanguageModel.for_inference(simulator)

        self._log_num_params(model, model_path)
        self._log_num_params(simulator, simulator_path)
        return model, tokenizer, simulator, simulator_tokenizer

    def _log_num_params(self, torch_model, torch_model_path: str) -> None:
        if not self.verbose:
            return
        total_params = sum(p.numel() for p in torch_model.parameters())
        train_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
        self.logger.info(f">>> Model loaded from `{torch_model_path}`")
        self.logger.info(f">>> Number of total parameters: {total_params}")
        self.logger.info(f">>> Number of trainable parameters: {train_params}")

    def finetune(self) -> None:
        from trainer.sft_config import SFTConfig
        from trainer.sft_trainer import SFTTrainer

        self._apply_training_overrides_from_wandb()
        report_to = "wandb" if self.use_wandb else "none"
        max_steps = self.common_training_args["max_steps"]
        base_learning_rate = self.common_training_args["learning_rate"]
        growth = 1.2

        for outer_ep in range(self.finetune_epochs):
            self.logger.info(f">>> [outer_ep {outer_ep} START] [Total finetune_epochs = {self.finetune_epochs}]")
            if self.use_wandb:
                wandb.log({"finetune-epoch": outer_ep + 1})

            if max_steps > 0:
                max_steps = int(round(max_steps * (growth ** outer_ep)))
                self.common_training_args["max_steps"] = max_steps
                if max_steps % 2 == 0:
                    base_learning_rate = base_learning_rate * (0.8 ** (outer_ep / 2))
                    self.common_training_args["learning_rate"] = base_learning_rate
            self.logger.info(
                f">>> [learning_rate]: {self.common_training_args['learning_rate']}, [max_steps]: {max_steps}"
            )

            cur_seed = self._base_seed + outer_ep
            random_setup(seed=cur_seed, has_cuda=self.cuda_dict["has_cuda"])
            self.seed = cur_seed

            epoch_dir = os.path.join(self._base_ckpt_root, f"ft_epoch_{outer_ep + 1:02d}")
            os.makedirs(epoch_dir, exist_ok=True)
            if self.verbose:
                self.logger.info(f">>> [Outer Finetune Epoch {outer_ep + 1}/{self.finetune_epochs}] -> {epoch_dir}")

            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            lora_rank = int(self.lora_args.get("lora_rank", 64))
            lora_alpha = int(self.lora_args.get("lora_alpha", 64))
            lora_dropout = float(self.lora_args.get("lora_dropout", 0))
            self.model = unsloth.FastModel.get_peft_model(
                self.model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=3407 + outer_ep,
                use_rslora=False,
                loftq_config=None,
            )
            self.model.train()

            dataset_loader = SampleDataloader(
                tokenizer_train=self.tokenizer_train,
                tokenizer_eval=self.tokenizer_eval,
                model=self.model if self.finetuned_model is None else self.finetuned_model,
                simulator=self.simulator,
                simulator_tokenizer=self.simulator_tokenizer,
                model_name=self.model_name,
                project_root_dir=self.project_root_dir,
                do_eval=False,
                debug=self.debug,
                train_type=self.train_type,
                outer_epoch_num=outer_ep,
                total_finetune_epochs=self.finetune_epochs,
                do_sample_ranking=self.do_sample_ranking,
                epoch_new_data=self.epoch_new_data,
                lora_alpha=lora_alpha,
                lora_rank=lora_rank,
                ms_num=self.ms_num,
                uttr_num=self.uttr_num,
                rollout_turns=self.rollout_turns,
            )
            data_train_list, _ = dataset_loader.load_data()
            data_train = Dataset.from_list(data_train_list)

            self.common_training_args["seed"] = cur_seed
            self.common_training_args["data_seed"] = cur_seed
            self.common_training_args["resume_from_checkpoint"] = None
            self.common_training_args["run_name"] = epoch_dir
            self.common_training_args["output_dir"] = epoch_dir
            self.common_training_args["report_to"] = report_to
            self.common_training_args["do_eval"] = False
            self.common_training_args["eval_on_start"] = False
            self.common_training_args["metric_for_best_model"] = "loss"
            self.common_training_args["eval_strategy"] = "no"
            self.common_training_args["eval_steps"] = None
            self.common_training_args["load_best_model_at_end"] = False

            self.sft_trainer_args["eos_token"] = self.tokenizer_train.eos_token
            self.sft_trainer_args["pad_token"] = self.tokenizer_train.pad_token
            self.sft_trainer_args["max_length"] = self.max_seq_len
            config = SFTConfig(
                **self.sft_trainer_args,
                **self.common_training_args,
            )
            self.logger.info(f"[common_training_args]: {self.common_training_args}")
            self.logger.info(f"[lora_args]: {self.lora_args}")

            trainer = SFTTrainer(
                model=self.model,
                judge=self.simulator,
                judge_tokenizer=self.simulator_tokenizer,
                args=config,
                data_collator=None,
                train_dataset=data_train,
                eval_dataset=None,
                processing_class=None,
                compute_loss_func=None,
                compute_metrics=None,
                callbacks=None,
                optimizers=(None, None),
                optimizer_cls_and_kwargs=None,
                preprocess_logits_for_metrics=None,
                peft_config=None,
                formatting_func=None,
                train_type=self.train_type,
            )

            hyper_param_dir = os.path.join(epoch_dir, "hyper_params")
            os.makedirs(hyper_param_dir, exist_ok=True)
            with open(os.path.join(hyper_param_dir, "common_training_args.json"), "w", encoding="utf-8") as fp_out:
                json.dump(self.common_training_args, fp_out, indent=4)
            with open(os.path.join(hyper_param_dir, "sft_trainer_args.json"), "w", encoding="utf-8") as fp_out:
                json.dump(self.sft_trainer_args, fp_out, indent=4)

            training_results = trainer.train(resume_from_checkpoint=None)
            self.logger.info(f">>> [Outer {outer_ep + 1}] Training finished. TrainOutput:\n{training_results}")

            last_ckpt_dir = self._save_epoch_model(trainer, epoch_dir)
            self._cleanup_after_epoch(trainer)
            del trainer, data_train, dataset_loader
            self.logger.info(f">>> [outer_ep {outer_ep} END] [Total finetune_epochs = {self.finetune_epochs}]")

            if (not self.no_re_initialize) and outer_ep < (self.finetune_epochs - 1):
                self._reload_model_for_next_epoch(last_ckpt_dir)

    def _save_epoch_model(self, trainer, epoch_dir: str) -> str:
        ckpt_fn_list = [fn for fn in os.listdir(epoch_dir) if fn.startswith("checkpoint-")]
        if not ckpt_fn_list:
            self.logger.info(">>> No checkpoint saved")
            best_ckpt_dir = os.path.join(epoch_dir, "last_model")
            os.makedirs(best_ckpt_dir, exist_ok=True)
            trainer.save_model(best_ckpt_dir)
            return best_ckpt_dir

        self.logger.info(f">>> Saved checkpoints: {ckpt_fn_list}")
        ckpt_fn_list.sort(key=lambda x: int(x.split("-")[-1]))
        last_ckpt_dir = os.path.join(epoch_dir, ckpt_fn_list[-1])
        best_ckpt_dir = os.path.join(epoch_dir, "last_model")
        shutil.copytree(last_ckpt_dir, best_ckpt_dir, dirs_exist_ok=True)
        trainer.save_model(best_ckpt_dir)
        return last_ckpt_dir

    def _cleanup_after_epoch(self, trainer) -> None:
        try:
            if hasattr(trainer, "accelerator"):
                trainer.accelerator.free_memory()
        except Exception as exc:
            self.logger.info(f">>> !!! Exception when `trainer.accelerator.free_memory()`\n{exc}")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _reload_model_for_next_epoch(self, last_ckpt_dir: str) -> None:
        if not os.path.isdir(last_ckpt_dir):
            raise ValueError(f"Missing checkpoint directory: {last_ckpt_dir}")

        del self.model, self.simulator
        if self.finetuned_model is not None:
            del self.finetuned_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.logger.info(f"Reinitialize the model to the base model... model_name: {self.model_name}")
        self.model, self.tokenizer, self.simulator, self.simulator_tokenizer = self.initialize_model_simulator(
            model_name=self.model_name,
            simulator_name=self.simulator_name,
        )
        self.logger.info(f"Load the previous best checkpoint as self.finetuned model... last_ckpt_dir: {last_ckpt_dir}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finetuned_model, _ = unsloth.FastLanguageModel.from_pretrained(
            self._get_model_path(self.model_name),
            device_map=device,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            load_in_4bit=False,
            dtype="bfloat16",
            max_seq_length=self.max_seq_len,
        )
        self.finetuned_model = PeftModel.from_pretrained(self.finetuned_model, last_ckpt_dir)
        unsloth.FastLanguageModel.for_inference(self.finetuned_model)


def _validate_release_scope(training_data: str, training_strategy: str, do_eval: bool) -> None:
    if training_data != "sotopia-pi-tom":
        raise ValueError("This release only supports --training_data=sotopia-pi-tom.")
    if training_strategy != "sft":
        raise ValueError("This release only supports --training_strategy=sft.")
    if do_eval:
        raise ValueError("This release only supports --do_eval=False.")


def main(
        task: int = 1,
        model_name: str = "qwen2.5-3b",
        simulator_name: str = "qwen2.5-14b-unsloth-4bit",
        cache_dir: Optional[str] = None,
        project_root_dir: Optional[str] = None,
        ckpt_root_dir: Optional[str] = None,
        config_dir: Optional[str] = "config/default/",
        seed: int = 42,
        cuda: Optional[str] = None,
        training_data: str = "sotopia-pi-tom",
        training_strategy: str = "sft",
        do_eval: bool = False,
        max_seq_len: Optional[int] = 4096,
        verbose: bool = False,
        wandb_key: Optional[str] = None,
        debug: bool = False,
        train_type: str = "ms-uttr-sep",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        finetune_epochs: int = 1,
        do_sample_ranking: bool = False,
        epoch_new_data: int = 100,
        no_re_initialize: bool = False,
        ms_num: int = 2,
        uttr_num: int = 2,
        rollout_turns: int = 0,
        **kwargs,
) -> None:
    """Run Sotopia SFT training for ToMA."""
    _validate_release_scope(training_data, training_strategy, do_eval)
    if int(task) != 1:
        raise ValueError("This release only supports --task=1.")
    if project_root_dir is None:
        project_root_dir = os.getcwd()

    timer_start = time.perf_counter()
    logger = logger_setup("ToM Training")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])
    logger.info(f">>> cuda_dict:\n{cuda_dict}")

    wandb_config_snapshot: Dict[str, Any] = {}
    if use_wandb:
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(project=wandb_project, entity=wandb_entity)
        wandb_config_snapshot = dict(wandb.config)
        logger.info(f">>> W&B sweep config snapshot: {wandb_config_snapshot}")

        model_name = wandb_config_snapshot.get("model_name", model_name)
        simulator_name = wandb_config_snapshot.get("simulator_name", simulator_name)
        train_type = wandb_config_snapshot.get("train_type", train_type)
        max_seq_len = wandb_config_snapshot.get("max_seq_len", max_seq_len)
        debug = wandb_config_snapshot.get("debug", debug)

    run_id = f"ToM---sft---sotopia-pi-tom---{model_name}--{train_type}"
    cur_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    ckpt_root = ckpt_root_dir if isinstance(ckpt_root_dir, str) and ckpt_root_dir else project_root_dir
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_dir = os.path.join(ckpt_root, "ckpt_tom", run_id, cur_time)
    while os.path.isdir(ckpt_dir):
        time.sleep(3)
        cur_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        ckpt_dir = os.path.join(ckpt_root, "ckpt_tom", run_id, cur_time)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f">>> New Running: ckpt_dir = {ckpt_dir}")

    kwargs_lora_rank = int(kwargs.pop("lora_rank", 64))
    kwargs_lora_alpha = int(kwargs.pop("lora_alpha", 64))
    kwargs_lora_dropout = float(kwargs.pop("lora_dropout", 0.0))
    if kwargs:
        logger.info(f">>> Training argument overrides from CLI kwargs: {kwargs}")

    tom_training = TomTraining(
        verbose=verbose,
        logger=logger,
        seed=seed,
        cuda_dict=cuda_dict,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        ckpt_dir=ckpt_dir,
        config_dir=config_dir,
        model_name=model_name,
        simulator_name=simulator_name,
        run_id=run_id,
        do_eval=False,
        max_seq_len=max_seq_len,
        use_wandb=use_wandb,
        debug=debug,
        train_type=train_type,
        wandb_config=wandb_config_snapshot,
        lora_args={
            "lora_rank": wandb_config_snapshot.get("lora_rank", kwargs_lora_rank),
            "lora_alpha": wandb_config_snapshot.get("lora_alpha", kwargs_lora_alpha),
            "lora_dropout": wandb_config_snapshot.get("lora_dropout", kwargs_lora_dropout),
        },
        finetune_epochs=finetune_epochs,
        do_sample_ranking=do_sample_ranking,
        epoch_new_data=min(100, max(0, int(epoch_new_data))),
        no_re_initialize=no_re_initialize,
        ms_num=ms_num,
        uttr_num=uttr_num,
        rollout_turns=rollout_turns,
    )
    tom_training.common_training_args.update(kwargs)
    tom_training.finetune()

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
