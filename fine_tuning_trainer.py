#!/usr/bin/env python3
"""
Advanced LLM Fine-tuning for Image Captioning
Supports LoRA, QLoRA, and full fine-tuning approaches
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
from tqdm import tqdm
import wandb
from datetime import datetime
import numpy as np

class ImageCaptionFineTuner:
    """
    Advanced fine-tuning system for image captioning models
    Supports multiple techniques: LoRA, QLoRA, Full Fine-tuning
    """
    
    def __init__(self, 
                 base_model="Salesforce/blip-image-captioning-base",
                 fine_tuning_method="lora",
                 device=None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model
        self.fine_tuning_method = fine_tuning_method
        
        print(f"Initializing fine-tuner with {fine_tuning_method} on {self.device}")
        self._setup_model()
    
    def _setup_model(self):
        """Setup base model and processor"""
        self.processor = BlipProcessor.from_pretrained(self.base_model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.fine_tuning_method == "lora":
            self._setup_lora()
        elif self.fine_tuning_method == "qlora":
            self._setup_qlora()
        
        self.model.to(self.device)
    
    def _setup_lora(self):
        """Setup LoRA (Low-Rank Adaptation) configuration"""
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,  # Rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA setup complete. Trainable parameters: {self.model.num_parameters()}")
    
    def _setup_qlora(self):
        """Setup QLoRA (Quantized LoRA) configuration"""
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Reload model with quantization
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Apply LoRA to quantized model
        self._setup_lora()
        print("QLoRA setup complete with 4-bit quantization")
    
    def prepare_dataset(self, dataset_name, split="train", max_samples=None):
        """
        Prepare dataset for fine-tuning
        Supports multiple datasets: COCO, Flickr30k, Custom
        """
        if dataset_name.lower() == "coco":
            from dataset_handler import COCOCaptionDataset
            dataset = COCOCaptionDataset(split=split, max_samples=max_samples, processor=self.processor)
        elif dataset_name.lower() == "flickr30k":
            from dataset_handler import FlickrDataset
            dataset = FlickrDataset(split=split, processor=self.processor)
        elif dataset_name.lower() == "custom":
            dataset = self._load_custom_dataset(split, max_samples)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return dataset
    
    def _load_custom_dataset(self, split, max_samples):
        """Load custom dataset from local files"""
        # Implementation for custom dataset loading
        # Expected format: JSON with image paths and captions
        pass
    
    def fine_tune(self, 
                  dataset_name="coco",
                  output_dir="./fine_tuned_model",
                  num_epochs=3,
                  batch_size=8,
                  learning_rate=5e-5,
                  warmup_steps=500,
                  max_samples=1000,
                  use_wandb=False):
        """
        Fine-tune the model on specified dataset
        """
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{dataset_name}_{self.fine_tuning_method}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="image-captioning-finetuning",
                name=f"{dataset_name}_{self.fine_tuning_method}_{timestamp}",
                config={
                    "model": self.base_model_name,
                    "method": self.fine_tuning_method,
                    "dataset": dataset_name,
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                }
            )
        
        # Prepare datasets
        print(f"Loading {dataset_name} dataset...")
        train_dataset = self.prepare_dataset(dataset_name, "train", max_samples)
        val_dataset = self.prepare_dataset(dataset_name, "validation", max_samples//4)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if use_wandb else None,
            fp16=True if self.device == "cuda" else False,
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )
        
        # Custom trainer for image captioning
        trainer = ImageCaptionTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processor=self.processor
        )
        
        # Start training
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "base_model": self.base_model_name,
            "fine_tuning_method": self.fine_tuning_method,
            "dataset": dataset_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "output_dir": output_dir,
            "timestamp": timestamp
        }
        
        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Fine-tuning completed! Model saved to: {output_dir}")
        return output_dir

class ImageCaptionTrainer(Trainer):
    """Custom trainer for image captioning tasks"""
    
    def __init__(self, processor, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation for image captioning"""
        
        pixel_values = inputs.get("pixel_values")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss