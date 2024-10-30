import numpy as np
import pandas as pd
import os
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    default_data_collator,
)
import ray
from ray import train
import ray.data
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig
import boto3
import shutil

ray.init()

# Configuration for a 16GB GPU
model_name = "Qwen/Qwen2.5-0.5B"
use_gpu = True
num_workers = 1
block_size = 512
batch_size = 1
#storage_path = "s3://your-bucket-here"  # Set your S3 bucket

# Load dataset
print("Loading tiny_shakespeare dataset")
current_dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)

ray_datasets = {
    "train": ray.data.from_huggingface(current_dataset["train"]),
    "validation": ray.data.from_huggingface(current_dataset["validation"]),
}

def split_text(batch: pd.DataFrame) -> pd.DataFrame:
    text = list(batch["text"])
    flat_text = "".join(text)
    split_text = [
        x.strip()
        for x in flat_text.split("\n")
        if x.strip() and not x.strip()[-1] == ":"
    ]
    return pd.DataFrame(split_text, columns=["text"])


def tokenize(batch: pd.DataFrame) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    ret = tokenizer(
        list(batch["text"]),
        truncation=True,
        max_length=block_size,
        padding="max_length",
        return_tensors="np",
    )
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)


processed_datasets = {
    key: (
        ds.map_batches(split_text, batch_format="pandas")
        .map_batches(tokenize, batch_format="pandas")
    )
    for key, ds in ray_datasets.items()
}

# Training function
def train_func(config):
    torch.backends.cuda.matmul.allow_tf32 = True

    deepspeed = {
        "fp16": {
            "enabled": "auto",
            "initial_scale_power": 8,
            "hysteresis": 4,
            "consecutive_hysteresis": True,
        },
        "bf16": {"enabled": "auto"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": 722534,
            "stage3_param_persistence_threshold": "auto",
            "gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }


    # Training arguments
    training_args = TrainingArguments(
        "output",
        logging_steps=1,
        save_strategy="steps",
        save_steps=config.get("steps_per_epoch"),
        max_steps=config.get("steps_per_epoch") * config.get("epochs"),
        per_device_train_batch_size=config.get("batch_size"),
        fp16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
    model.resize_token_embeddings(len(tokenizer))

    metric = evaluate.load("accuracy")
    train_ds = train.get_dataset_shard("train")
    eval_ds = train.get_dataset_shard("validation")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.iter_torch_batches(batch_size=config.get("batch_size")),
        eval_dataset=eval_ds.iter_torch_batches(batch_size=config.get("batch_size")),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()

# Trainer setup
train_ds_size = processed_datasets["train"].count()
steps_per_epoch = train_ds_size // (batch_size * num_workers)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"epochs": 1, "batch_size": batch_size, "steps_per_epoch": steps_per_epoch},
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    datasets=processed_datasets,
)

results = trainer.fit()
checkpoint = results.checkpoint

# Save checkpoint directory locally
checkpoint_dir = "checkpoint"
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
checkpoint.to_directory(checkpoint_dir)

# S3 bucket configuration
s3_bucket = "ray-bucket-model-output"
s3_client = boto3.client("s3")

# Upload checkpoint directory to S3
def upload_folder_to_s3(local_folder, bucket, s3_folder):
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = f"{s3_folder}/{relative_path}"

            s3_client.upload_file(local_path, bucket, s3_path)
            print(f"Uploaded {local_path} to s3://{bucket}/{s3_path}")

# Upload the checkpoint folder to S3
upload_folder_to_s3(checkpoint_dir, s3_bucket, "model_output")

ray.shutdown()