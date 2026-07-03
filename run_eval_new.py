#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Optional

import fire

from sotopia import SotopiaEvaluator
from utils.init_functions import cuda_setup, logger_setup, random_setup
from utils.models import OPEN_MODEL_HF, PROPRIETARY_LLM


def _validate_model_name(model_name: Optional[str], label: str) -> None:
    if not model_name:
        return
    if model_name not in OPEN_MODEL_HF and model_name not in PROPRIETARY_LLM:
        raise ValueError(f"Unsupported {label}: {model_name}")


def main(
        eval_task: str = "sotopia",
        gen_model: str = "qwen2.5-3b",
        gen_partner_model: Optional[str] = None,
        eval_model: str = "gpt-5-mini",
        ckpt_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        project_root_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        seed: int = 42,
        cuda: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
        sotopia_eval_mode: str = "all",
        sotopia_use_ms: bool = False,
        sotopia_use_strategy: bool = False,
        sotopia_use_cot: bool = False,
        sotopia_gen_only: bool = False,
        sotopia_eval_only: bool = False,
        sotopia_eval_filepath: Optional[str] = None,
        sotopia_eval_turns: int = -1,
        sotopia_generator_ckpt_path: Optional[str] = None,
        sotopia_partner_ckpt_path: Optional[str] = None,
) -> None:
    """Run Sotopia generation/evaluation for ToMA."""
    if eval_task != "sotopia":
        raise ValueError("This release only supports --eval_task=sotopia.")
    if project_root_dir is None:
        project_root_dir = os.getcwd()
    if output_dir is None:
        output_dir = os.path.join(project_root_dir, "results", "sotopia")

    _validate_model_name(gen_model, "gen_model")
    _validate_model_name(gen_partner_model, "gen_partner_model")
    _validate_model_name(eval_model, "eval_model")

    timer_start = time.perf_counter()
    logger = logger_setup("ToM Sotopia Evaluation")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])
    logger.info(f">>> cuda_dict:\n{cuda_dict}")

    if not isinstance(cache_dir, str) or not os.path.isdir(cache_dir):
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface")
        os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir

    generator_ckpt_path = sotopia_generator_ckpt_path or ckpt_path
    evaluator = SotopiaEvaluator(
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        verbose=verbose,
        debug=debug,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        output_dir=output_dir,
        eval_mode=sotopia_eval_mode,
        use_ms=sotopia_use_ms,
        use_strategy=sotopia_use_strategy,
        use_cot=sotopia_use_cot,
        gen_only=sotopia_gen_only,
        eval_only=sotopia_eval_only,
        eval_filepath=sotopia_eval_filepath,
    )
    evaluator.run_eval_open(
        generator_model_name=gen_model,
        gen_partner_model_name=gen_partner_model or gen_model,
        evaluator_model_name=eval_model,
        generator_ckpt_path=generator_ckpt_path,
        evaluator_ckpt_path=None,
        gen_partner_ckpt_path=sotopia_partner_ckpt_path,
        do_4bit=False,
        eval_turns=int(sotopia_eval_turns),
    )

    total_sec = time.perf_counter() - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
