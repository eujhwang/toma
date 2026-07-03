# -*- coding: utf-8 -*-

OPEN_MODEL_HF = {
    # Gemma
    "gemma3-4b": "google/gemma-3-4B-it",
    # Mistral
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    # LLaMA-3
    "llama3-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen-2.5
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    # Qwen-3
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    # unsloth models: https://docs.unsloth.ai/get-started/all-our-models
    # LLaMA-3 (unsloth)
    "llama3-1b-unsloth": "unsloth/Llama-3.2-1B-Instruct",
    "llama3-1b-unsloth-4bit": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "llama3-3b-unsloth": "unsloth/Llama-3.2-3B-Instruct",
    "llama3-3b-unsloth-4bit": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama3-8b-unsloth": "unsloth/Llama-3.1-8B-Instruct",
    "llama3-8b-unsloth-4bit": "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    # "llama3-8b-unsloth-4bit": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama3-70b-unsloth": "unsloth/Llama-3.3-70B-Instruct",
    "llama3-70b-unsloth-4bit": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    # Qwen-2.5 (unsloth)
    "qwen2.5-0.5b-unsloth": "unsloth/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b-unsloth-4bit": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "qwen2.5-1.5b-unsloth": "unsloth/Qwen2.5-1.5B-Instruct",
    "qwen2.5-1.5b-unsloth-4bit": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "qwen2.5-3b-unsloth": "unsloth/Qwen2.5-3B-Instruct",
    "qwen2.5-3b-unsloth-4bit": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "qwen2.5-7b-unsloth": "unsloth/Qwen2.5-7B-Instruct",
    "qwen2.5-7b-unsloth-4bit": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "qwen2.5-14b-unsloth": "unsloth/Qwen2.5-14B-Instruct",
    "qwen2.5-14b-unsloth-4bit": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "qwen2.5-32b-unsloth": "unsloth/Qwen2.5-32B-Instruct",
    "qwen2.5-32b-unsloth-4bit": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    "qwen2.5-72b-unsloth": "unsloth/Qwen2.5-72B-Instruct",
    "qwen2.5-72b-unsloth-4bit": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
    # Qwen-3 (unsloth)
    "qwen3-0.6b-unsloth": "unsloth/Qwen3-0.6B",
    "qwen3-0.6b-unsloth-4bit": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    "qwen3-1.7b-unsloth": "unsloth/Qwen3-1.7B",
    "qwen3-1.7b-unsloth-4bit": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    "qwen3-4b-unsloth": "unsloth/Qwen3-4B",
    "qwen3-4b-unsloth-4bit": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "qwen3-8b-unsloth": "unsloth/Qwen3-8B",
    "qwen3-8b-unsloth-4bit": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "qwen3-14b-unsloth": "unsloth/Qwen3-14B",
    "qwen3-14b-unsloth-4bit": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "qwen3-32b-unsloth": "unsloth/Qwen3-32B",
    "qwen3-32b-unsloth-4bit": "unsloth/Qwen3-32B-unsloth-bnb-4bit",
    "gpt-oss-20b": "unsloth/gpt-oss-20b",
}


PROPRIETARY_LLM = {
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash-8b",
}
