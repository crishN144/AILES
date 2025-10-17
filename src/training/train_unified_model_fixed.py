#!/usr/bin/env python3
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json
import yaml
import logging
import argparse
import time
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedLegalAITrainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing trainer on {self.device}")

    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_model_and_tokenizer(self):
        model_path = self.config['model']['base_model']
        
        logger.info(f"Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        logger.info(f"Loading model from: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def load_training_data(self, data_path: str):
        logger.info(f"Loading training data from {data_path}")
        
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        example = json.loads(line.strip())
                        if 'text' in example and example['text'].strip():
                            examples.append(example)
                    except:
                        continue
        
        logger.info(f"Loaded {len(examples)} examples")
        
        if len(examples) == 0:
            raise ValueError("No valid examples found")
        
        dataset = Dataset.from_list(examples)
        train_size = max(1, int(len(dataset) * 0.9))
        
        return DatasetDict({
            'train': dataset.select(range(train_size)),
            'validation': dataset.select(range(train_size, len(dataset)))
        })

    def tokenize_dataset(self, dataset):
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=2048,
                return_tensors=None,
                add_special_tokens=False
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

    def train(self, data_path: str):
        logger.info("Starting training")
        
        self.setup_model_and_tokenizer()
        dataset = self.load_training_data(data_path)
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        output_dir = Path("./models/unified_legal_ai")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            max_steps=400,
            per_device_train_batch_size=10,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_steps=40,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            bf16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_total_limit=2,
            report_to=None
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        final_path = output_dir / "final_model"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        logger.info(f"Training completed - model saved to {final_path}")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llama32_training_config.yaml")
    parser.add_argument("--data", required=True)
    
    args = parser.parse_args()
    
    trainer = UnifiedLegalAITrainer(args.config)
    trainer.train(args.data)

if __name__ == "__main__":
    main()
