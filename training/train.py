"""
Fine-tunes Mistral-7B-Instruct-v0.2 on support ticket data using:
-LoRA(PEFT) with 4-bit QLoRA quantisation
-SFTTrainer from TRL
-MLflow + Weights & Biases experiment tracking

Requirements:
  - GPU with 16GB+ VRAM (or 8GB with use_4bit=True)
  - Google Colab T4 works fine with use_4bit=True

Run:
    python training/train.py
    python training/train.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --epochs 3
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
import torch
import wandb
wandb.login(key="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",relogin=True)
from datasets import Dataset
from loguru import logger
from peft import LoraConfig,TaskType,get_peft_model, prepare_model_for_kbit_training
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer

#-----config defaults-----------------------
DEFAULT_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR     = "training/checkpoints"
DATA_DIR       = "data"
MAX_SEQ_LEN    = 512
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05

#-------------Helper-------------------------
def load_jsonl(path:str)->Dataset:
    rows=[]
    with open(path)as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return Dataset.from_list(rows)

def get_bnb_config(use_4bit: bool)-> BitsAndBytesConfig|None:
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def get_lora_config()->LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        inference_mode=False,
    )
#-------------------Main train Function--------------------------
def train(
        model_name:str=DEFAULT_MODEL,
        use_4bit:bool=True,
        epochs:int=3,
        batch_size:int=4,
        lr:float=2e-4,
        run_name:str="ticket-triage-lora-v1"
)->None:
    logger.info(f"Starting training: {model_name} | LoRA r={LORA_R} | 4bit={use_4bit} | epochs={epochs}")

# ── WandB ──────────────────────────────────────────────────────────────────
    wandb.init(
        project="ticket-triage-ai",
        name=run_name,
        config={
            "model": model_name, "epochs": epochs,
            "batch_size": batch_size, "lr": lr,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "use_4bit": use_4bit, "max_seq_len": MAX_SEQ_LEN,
        },
    )

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_experiment("ticket-triage-ai")
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "model": model_name, "epochs": epochs, "lr": lr,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "use_4bit": use_4bit,
    })

#-----------Tokeniser--------------------------------
    logger.info("Loading Tokeniser.........")
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="right"
 
    #------------Model----------------------------
    logger.info("Loading Model....")
    bnb_config=get_bnb_config(use_4bit)
    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.config.use_cache=False,
    model.config.pretraining_tp=1

    if use_4bit:
        model=prepare_model_for_kbit_training(model)

    #---------------LoRA-----------------------------
    lora_config=get_lora_config()
    model=get_peft_model(model,lora_config)
    model.print_trainable_parameters()

    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    mlflow.log_params({"trainable_params":trainable,"total_params":total})

    # ── Data ───────────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_dataset = load_jsonl(f"{DATA_DIR}/train.jsonl")
    val_dataset   = load_jsonl(f"{DATA_DIR}/val.jsonl")
    mlflow.log_params({"train_size": len(train_dataset), "val_size": len(val_dataset)})
    logger.info(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    
    # ── Training arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb", "mlflow"],
        run_name=run_name,
        dataloader_num_workers=2,
        group_by_length=True,
        optim="paged_adamw_32bit",
    )
    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

     #── Train ──────────────────────────────────────────────────────────────────
    logger.info("Training started...")
    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────────
    save_path = Path(OUTPUT_DIR) / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info(f"Model saved -> {save_path}")

    # ── Final eval metrics ─────────────────────────────────────────────────────
    final_metrics = trainer.evaluate()
    mlflow.log_metrics({k: round(v, 4) for k, v in final_metrics.items()})
    logger.info(f"Final eval: {final_metrics}")


    mlflow.end_run()
    wandb.finish()

    logger.info("Training complete.")
    logger.info("Next step: python training/evaluate.py")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B on support tickets")
    parser.add_argument("--model_name",  default=DEFAULT_MODEL,                     help="HuggingFace model ID")
    parser.add_argument("--use_4bit",    default=True,  type=lambda x: x == "True", help="4-bit QLoRA quantisation")
    parser.add_argument("--epochs",      default=3,     type=int)
    parser.add_argument("--batch_size",  default=4,     type=int)
    parser.add_argument("--lr",          default=2e-4,  type=float)
    parser.add_argument("--run_name",    default="ticket-triage-lora-v1")
    args = parser.parse_args()
    train(**vars(args))

