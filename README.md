# ToMA

Code for Sotopia SFT training and Sotopia evaluation.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Set up the original Sotopia and Sotopia-Pi data following their official
repositories/dataset pages. The raw Sotopia-Pi source file should be available
at:

```text
data/sotopia-pi/sotopia_pi_episodes.jsonl
```

Each JSONL row should follow the original Sotopia-Pi episode structure:

```json
{
  "episode_id": "example_episode_id",
  "environment_id": "example_environment_id",
  "agent_ids": ["Alice", "Bob"],
  "raw_messages": [
    [
      ["Environment", "Alice", ""],
      ["Environment", "Bob", ""],
      ["Alice", "Environment", "said: I felt ignored during the meeting."],
      ["Bob", "Environment", "did nothing"]
    ]
  ],
  "scenario": "Two friends are deciding how to handle a disagreement.",
  "agents_background": {
    "Alice": "Alice values direct communication but worries about conflict.",
    "Bob": "Bob wants to be helpful but often misses social cues."
  },
  "social_goals": {
    "Alice": "Resolve the disagreement without hurting the friendship.",
    "Bob": "Explain his concern honestly."
  },
  "social_interactions": "Alice and Bob discuss a misunderstanding from a meeting.",
  "reasoning": "The episode requires reasoning about each agent's beliefs, goals, and relationship.",
  "rewards": [
    [
      "Alice",
      {
        "relationship": { "score": 4 },
        "knowledge": { "score": 8 },
        "goal": { "score": 7 }
      }
    ]
  ]
}
```

The training pipeline generates ToMA mental-state and rollout data from the
Sotopia-Pi episodes and writes the training file used by `run_train.py`:

```text
data/gemini-tom-conversation-sotopia-pi-episodes.json
```

## Train

```bash
python run_train.py \
  --verbose \
  --use_wandb=True \
  --cache_dir /scratch/ejhwang/teach-tom/cache/ \
  --project_root_dir /project/aip-vshwartz/ejhwang/teach-tom/ \
  --ckpt_root_dir /scratch/ejhwang/teach-tom/cache/ \
  --config_dir config/default \
  --finetune_epochs=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=0.0001 \
  --logging_steps=10 \
  --lora_alpha=128 \
  --lora_dropout=0 \
  --lora_rank=32 \
  --lr_scheduler_type=cosine \
  --max_seq_len=4096 \
  --max_steps=-1 \
  --model_name=qwen2.5-3b \
  --ms_num=2 \
  --num_train_epochs=3 \
  --per_device_train_batch_size=2 \
  --rollout_turns=0 \
  --save_steps=0.1 \
  --simulator_name=qwen2.5-14b-unsloth-4bit \
  --task=1 \
  --train_type=ms-uttr-sep \
  --training_data=sotopia-pi-tom \
  --training_strategy=sft \
  --uttr_num=2 \
  --warmup_ratio=0 \
  --warmup_steps=10 \
  --weight_decay=0
```

## Evaluate

```bash
python run_eval_new.py \
  --verbose \
  --eval_task=sotopia \
  --gen_model=qwen2.5-3b \
  --gen_partner_model=qwen2.5-3b \
  --eval_model=gpt-5-mini \
  --cache_dir /scratch/ejhwang/teach-tom/cache/ \
  --project_root_dir /project/aip-vshwartz/ejhwang/teach-tom/ \
  --output_dir /scratch/ejhwang/teach-tom/results/sotopia \
  --sotopia_eval_mode=hard
```

To evaluate a fine-tuned checkpoint, add:

```bash
--sotopia_generator_ckpt_path /path/to/checkpoint-or-last_model
```

## Paper

If the code is helpful for your project, please cite [our paper](https://arxiv.org/abs/2502.18331) (Bibtex below).
```
@inproceedings{hwang-etal-2026-infusing,
    title = "Infusing Theory of Mind into Socially Intelligent {LLM} Agents",
    author = "Hwang, EunJeong  and
      Yin, Yuwei  and
      Carenini, Giuseppe  and
      West, Peter  and
      Shwartz, Vered",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.551/",
    doi = "10.18653/v1/2026.findings-acl.551",
    pages = "11327--11360",
    ISBN = "979-8-89176-395-1"
}
```
