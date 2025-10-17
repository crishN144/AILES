#!/usr/bin/env python3
"""
AILES Legal AI Model Training Script - Llama-3.2-3B-Instruct
Fully optimized for AIRE HPC with comprehensive error handling
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json
import os
import yaml
import logging
import argparse
import time
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class OptimizedLegalAITrainer:
    def __init__(self, config_path: str, component: str):
        self.component = component
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
        # Memory management
        self._setup_memory_optimization()
        
        logger.info(f"🚀 Initializing trainer for {component} component")
        logger.info(f"💻 Device: {self.device}")
        logger.info(f"🔥 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"🎯 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    def _setup_memory_optimization(self):
        """Setup memory optimization settings"""
        # Set memory fraction if specified in config
        if 'memory' in self.config and 'max_memory_mb' in self.config['memory']:
            max_memory = self.config['memory']['max_memory_mb']
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(max_memory / 48000)  # L40S has 48GB
                logger.info(f"🔧 Set GPU memory limit to {max_memory}MB")
        
        # Enable memory efficient attention
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Disable tokenizers parallelism to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate training configuration"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'lora', 'training', 'data', 'hardware']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with comprehensive error handling"""
        model_path = self.config['model']['base_model']
        
        # Validate model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Check for required files
        required_files = ['config.json']
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required model files: {missing_files}")
        
        logger.info(f"📚 Loading tokenizer from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=self.config['model'].get('local_files_only', True),
                trust_remote_code=True,
                use_fast=True,
                padding_side="right"  # Important for training
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info("✅ Added pad token from eos token")
            
            logger.info(f"✅ Tokenizer loaded successfully")
            logger.info(f"📊 Vocab size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            raise
        
        # Configure quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=False  # Keep on GPU for L40S
        )
        
        logger.info(f"🤖 Loading model from: {model_path}")
        try:
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=self.config['model'].get('local_files_only', True),
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_cache=False,  # Disable for training
                attn_implementation="eager",  # Most stable for training
                low_cpu_mem_usage=True
            )
            
            logger.info("✅ Base model loaded successfully")
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            logger.info("✅ Model prepared for k-bit training")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Configure LoRA with validation
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias="none",
            inference_mode=False
        )
        
        logger.info("🔧 Applying LoRA configuration")
        try:
            self.model = get_peft_model(self.model, lora_config)
            logger.info("✅ LoRA applied successfully")
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
        except Exception as e:
            logger.error(f"❌ Failed to apply LoRA: {e}")
            raise
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")

    def load_training_data(self, data_path: str) -> DatasetDict:
        """Load and validate training data with comprehensive checks"""
        logger.info(f"📊 Loading training data from {data_path}")
        
        # Verify file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        # Check file is not empty
        if os.path.getsize(data_path) == 0:
            raise ValueError(f"Training data file is empty: {data_path}")
        
        # Load JSONL data with validation
        examples = []
        line_errors = 0
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        example = json.loads(line.strip())
                        
                        # Validate required fields
                        required_fields = ['instruction', 'input', 'output']
                        if all(field in example for field in required_fields):
                            examples.append(example)
                        else:
                            logger.warning(f"⚠️ Line {line_num}: Missing required fields")
                            line_errors += 1
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Line {line_num}: Invalid JSON - {e}")
                        line_errors += 1
        
        logger.info(f"📈 Loaded {len(examples)} valid examples")
        if line_errors > 0:
            logger.warning(f"⚠️ Skipped {line_errors} invalid lines")
        
        if len(examples) == 0:
            raise ValueError(f"No valid training examples found in {data_path}")
        
        if len(examples) < 10:
            logger.warning(f"⚠️ Very few training examples ({len(examples)}). Training may not be effective.")
        
        # Create instruction-following format with validation
        formatted_examples = []
        format_errors = 0
        
        for i, example in enumerate(examples):
            try:
                formatted_text = self._format_example_for_training(example)
                
                # Validate formatted text length
                if len(formatted_text) > 0:
                    formatted_examples.append({"text": formatted_text})
                else:
                    format_errors += 1
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to format example {i}: {e}")
                format_errors += 1
        
        logger.info(f"✅ Formatted {len(formatted_examples)} examples")
        if format_errors > 0:
            logger.warning(f"⚠️ Failed to format {format_errors} examples")
        
        if len(formatted_examples) == 0:
            raise ValueError("No examples could be formatted successfully")
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        
        # Split into train/validation with minimum validation size
        train_size = max(1, int(len(dataset) * self.config['data']['train_split']))
        val_size = len(dataset) - train_size
        
        if val_size < 1:
            val_size = 1
            train_size = len(dataset) - 1
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        
        logger.info(f"🎯 Training examples: {len(train_dataset)}")
        logger.info(f"✅ Validation examples: {len(val_dataset)}")
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })

    def _format_example_for_training(self, example: Dict[str, Any]) -> str:
        """Format example for Llama-3.2 with validation"""
        # Validate required fields
        if not all(key in example for key in ['instruction', 'input', 'output']):
            raise ValueError(f"Missing required fields in example: {list(example.keys())}")
        
        # Clean and validate fields
        instruction = str(example['instruction']).strip()
        input_text = str(example['input']).strip()
        output_text = str(example['output']).strip()
        
        if not instruction or not input_text or not output_text:
            raise ValueError("One or more fields are empty after cleaning")
        
        # Parse and reformat JSON output for better learning
        try:
            parsed_output = json.loads(output_text)
            output_text = json.dumps(parsed_output, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # Keep as string if not valid JSON
            pass
        
        # Clean the input data if it's JSON
        try:
            parsed_input = json.loads(input_text)
            input_text = json.dumps(parsed_input, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # Keep as string if not valid JSON
            pass
        
        # Use Llama-3.2 instruct format
        formatted = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output_text}<|eot_id|>"
        )
        
        return formatted

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Tokenize dataset with memory optimization"""
        logger.info("🔄 Tokenizing dataset...")
        
        def tokenize_function(examples):
            try:
                # Tokenize with proper settings
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.config['data']['max_seq_length'],
                    return_tensors=None,
                    add_special_tokens=False  # Already included in formatting
                )
                
                # For causal LM, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()
                
                return tokenized
                
            except Exception as e:
                logger.error(f"❌ Tokenization error: {e}")
                raise
        
        try:
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=1,  # Single process for stability
                remove_columns=dataset["train"].column_names,
                desc="Tokenizing dataset",
                load_from_cache_file=False  # Force recomputation
            )
            
            logger.info("✅ Tokenization complete")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"❌ Dataset tokenization failed: {e}")
            raise

    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments with validation"""
        output_dir = Path(self.config['training']['output_dir']) / f"{self.component}_llama32"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config['training']['num_train_epochs'],
            max_steps=self.config['training']['max_steps'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            eval_strategy=self.config['training']['eval_strategy'],
            save_strategy=self.config['training']['save_strategy'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            bf16=self.config['hardware']['bf16'],
            fp16=self.config['hardware']['fp16'],
            dataloader_num_workers=self.config['hardware']['dataloader_num_workers'],
            remove_unused_columns=self.config['hardware']['remove_unused_columns'],
            gradient_checkpointing=self.config['hardware']['gradient_checkpointing'],
            dataloader_pin_memory=self.config['hardware'].get('dataloader_pin_memory', False),
            report_to=None,
            run_name=f"ailes-{self.component}-{self._get_timestamp()}",
            push_to_hub=False,
            save_total_limit=self.config['training']['save_total_limit'],
            save_safetensors=self.config['training']['save_safetensors'],
            ddp_find_unused_parameters=False,
            logging_dir=str(logs_dir),
            # Memory optimization
            max_grad_norm=self.config.get('memory', {}).get('gradient_clipping', 1.0),
            # Disable some features that can cause issues
            prediction_loss_only=True,
        )
        
        return args

    def _get_timestamp(self) -> str:
        """Get current timestamp for run naming"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(self, data_path: str):
        """Main training function with comprehensive error handling"""
        start_time = time.time()
        logger.info(f"🚀 Starting training for {self.component} component")
        logger.info(f"📁 Data path: {data_path}")
        
        try:
            # Setup model and tokenizer
            self.setup_model_and_tokenizer()
            
            # Load and prepare data
            dataset = self.load_training_data(data_path)
            tokenized_dataset = self.tokenize_dataset(dataset)
            
            # Setup training arguments
            training_args = self.setup_training_arguments()
            
            # Setup data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
                return_tensors="pt"
            )
            
            # Setup callbacks
            callbacks = []
            if self.config.get('early_stopping'):
                callbacks.append(EarlyStoppingCallback(
                    early_stopping_patience=self.config['early_stopping']['patience'],
                    early_stopping_threshold=self.config['early_stopping']['threshold']
                ))
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                data_collator=data_collator,
                callbacks=callbacks,
                tokenizer=self.tokenizer,
            )
            
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Start training
            logger.info("🔥 Beginning training...")
            train_result = trainer.train()
            
            logger.info("✅ Training completed successfully")
            
            # Save final model
            final_model_path = Path(training_args.output_dir) / "final_model"
            trainer.save_model(str(final_model_path))
            
            # Save tokenizer explicitly
            self.tokenizer.save_pretrained(str(final_model_path))
            
            training_time = time.time() - start_time
            logger.info(f"⏱️ Training completed in {training_time/60:.1f} minutes")
            logger.info(f"💾 Model saved to {final_model_path}")
            
            # Save training metrics
            self._save_training_metrics(trainer, training_args.output_dir, training_time, train_result)
            
            # Final cleanup
            del trainer
            del self.model
            del self.tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            
            # Cleanup on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            raise

    def _save_training_metrics(self, trainer, output_dir: str, training_time: float, train_result):
        """Save comprehensive training metrics"""
        metrics_file = Path(output_dir) / "training_metrics.json"
        
        try:
            # Get final metrics from training result
            log_history = trainer.state.log_history
            final_train_loss = None
            final_eval_loss = None
            
            # Find the last recorded losses
            for log in reversed(log_history):
                if final_train_loss is None and "train_loss" in log:
                    final_train_loss = log["train_loss"]
                if final_eval_loss is None and "eval_loss" in log:
                    final_eval_loss = log["eval_loss"]
                if final_train_loss and final_eval_loss:
                    break
            
            metrics = {
                "component": self.component,
                "model_name": "llama-3.2-3b-instruct",
                "final_train_loss": final_train_loss,
                "final_eval_loss": final_eval_loss,
                "total_training_steps": trainer.state.global_step,
                "training_time_minutes": training_time / 60,
                "training_time_seconds": training_time,
                "examples_processed": len(trainer.train_dataset),
                "validation_examples": len(trainer.eval_dataset),
                "learning_rate": self.config['training']['learning_rate'],
                "batch_size": self.config['training']['per_device_train_batch_size'],
                "gradient_accumulation_steps": self.config['training']['gradient_accumulation_steps'],
                "effective_batch_size": (
                    self.config['training']['per_device_train_batch_size'] * 
                    self.config['training']['gradient_accumulation_steps']
                ),
                "lora_r": self.config['lora']['r'],
                "lora_alpha": self.config['lora']['lora_alpha'],
                "max_seq_length": self.config['data']['max_seq_length'],
                "config": self.config,
                "timestamp": self._get_timestamp(),
                "success": True
            }
            
            # Add train result metrics if available
            if hasattr(train_result, 'metrics'):
                metrics.update(train_result.metrics)
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"📊 Training metrics saved to {metrics_file}")
            if final_train_loss:
                logger.info(f"📈 Final train loss: {final_train_loss:.4f}")
            if final_eval_loss:
                logger.info(f"📉 Final eval loss: {final_eval_loss:.4f}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save training metrics: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train AILES Legal AI Model with Llama-3.2-3B")
    parser.add_argument("--component", required=True, choices=["chatbot", "predictor", "explainer"],
                        help="AI component to train")
    parser.add_argument("--config", default="configs/training_config.yaml",
                        help="Path to training configuration file")
    parser.add_argument("--data", required=True,
                        help="Path to training data JSONL file")
    
    args = parser.parse_args()
    
    # Comprehensive validation
    if not os.path.exists(args.config):
        logger.error(f"❌ Config file not found: {args.config}")
        sys.exit(1)
        
    if not os.path.exists(args.data):
        logger.error(f"❌ Data file not found: {args.data}")
        sys.exit(1)
    
    logger.info(f"🎯 Training {args.component} component")
    logger.info(f"⚙️ Config: {args.config}")
    logger.info(f"📊 Data: {args.data}")
    
    try:
        # Create trainer and start training
        trainer = OptimizedLegalAITrainer(args.config, args.component)
        success = trainer.train(args.data)
        
        if success:
            logger.info(f"🎉 {args.component} training completed successfully!")
            sys.exit(0)
        else:
            logger.error(f"❌ {args.component} training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        sys.exit(1)

if __name__ == "__main__":
    main()
