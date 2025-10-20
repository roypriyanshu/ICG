#!/usr/bin/env python3
"""
Dataset configurations and custom dataset handlers
Supports multiple datasets with unified interface
"""

import json
import os
from typing import List, Dict, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset

class DatasetConfig:
    """Configuration class for different datasets"""
    
    SUPPORTED_DATASETS = {
        "coco": {
            "name": "MS-COCO 2017",
            "description": "Microsoft Common Objects in Context - 118k training images",
            "splits": ["train", "validation"],
            "captions_per_image": 5,
            "total_images": 118287,
            "hf_dataset": "ydshieh/coco_dataset_script",
            "hf_config": "2017"
        },
        "flickr30k": {
            "name": "Flickr30k",
            "description": "Flickr 30k dataset - 31k images with 5 captions each",
            "splits": ["train", "validation", "test"],
            "captions_per_image": 5,
            "total_images": 31783,
            "hf_dataset": "nlphuji/flickr30k",
            "hf_config": None
        },
        "conceptual_captions": {
            "name": "Conceptual Captions 3M",
            "description": "Google's Conceptual Captions - 3.3M image-caption pairs",
            "splits": ["train", "validation"],
            "captions_per_image": 1,
            "total_images": 3318333,
            "hf_dataset": "conceptual_captions",
            "hf_config": None
        },
        "wit": {
            "name": "Wikipedia Image Text (WIT)",
            "description": "Wikipedia-based Image Text dataset - 37M image-text pairs",
            "splits": ["train"],
            "captions_per_image": 1,
            "total_images": 37000000,
            "hf_dataset": "wikimedia/wit_base",
            "hf_config": None
        },
        "visual_genome": {
            "name": "Visual Genome",
            "description": "Dense annotations of visual concepts - 108k images",
            "splits": ["train"],
            "captions_per_image": "variable",
            "total_images": 108077,
            "hf_dataset": "visual_genome",
            "hf_config": None
        }
    }
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        return cls.SUPPORTED_DATASETS.get(dataset_name.lower(), {})
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all supported datasets"""
        return list(cls.SUPPORTED_DATASETS.keys())

class UnifiedCaptionDataset(Dataset):
    """
    Unified dataset class that works with multiple caption datasets
    Provides consistent interface regardless of source dataset
    """
    
    def __init__(self, 
                 dataset_name: str,
                 split: str = "train",
                 max_samples: int = None,
                 processor = None,
                 transform = None):
        
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.processor = processor
        self.transform = transform
        
        # Get dataset configuration
        self.config = DatasetConfig.get_dataset_info(self.dataset_name)
        if not self.config:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        print(f"Loading {self.config['name']} ({split} split)...")
        self._load_dataset(max_samples)
    
    def _load_dataset(self, max_samples: int = None):
        """Load dataset from HuggingFace or custom source"""
        
        try:
            if self.dataset_name == "coco":
                self.dataset = load_dataset(
                    self.config["hf_dataset"], 
                    self.config["hf_config"], 
                    split=self.split
                )
            elif self.dataset_name == "flickr30k":
                self.dataset = load_dataset(
                    self.config["hf_dataset"], 
                    split=self.split
                )
            elif self.dataset_name == "conceptual_captions":
                self.dataset = load_dataset(
                    self.config["hf_dataset"], 
                    split=self.split
                )
            else:
                # For other datasets, implement custom loading
                self.dataset = self._load_custom_dataset()
            
            # Limit samples if specified
            if max_samples and max_samples < len(self.dataset):
                self.dataset = self.dataset.select(range(max_samples))
            
            print(f"Loaded {len(self.dataset)} samples")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def _load_custom_dataset(self):
        """Load custom dataset format"""
        # Implement custom dataset loading logic here
        # Expected format: JSON with image paths and captions
        raise NotImplementedError(f"Custom loading for {self.dataset_name} not implemented")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get dataset item with unified format"""
        
        item = self.dataset[idx]
        
        # Standardize format based on dataset
        if self.dataset_name == "coco":
            image = item['image']
            captions = [ann['caption'] for ann in item['annotations']]
            
        elif self.dataset_name == "flickr30k":
            image = item['image']
            captions = item['caption'] if isinstance(item['caption'], list) else [item['caption']]
            
        elif self.dataset_name == "conceptual_captions":
            image = item['image']
            captions = [item['caption']]
            
        else:
            # Default format
            image = item.get('image')
            captions = item.get('captions', [item.get('caption', '')])
        
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            else:
                raise ValueError(f"Invalid image format in dataset item {idx}")
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Process with model processor if available
        if self.processor:
            # Use first caption for training
            caption = captions[0] if captions else ""
            
            processed = self.processor(
                image, 
                text=caption, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            )
            
            return {
                'image': image,
                'pixel_values': processed['pixel_values'].squeeze(),
                'input_ids': processed['input_ids'].squeeze() if 'input_ids' in processed else None,
                'attention_mask': processed['attention_mask'].squeeze() if 'attention_mask' in processed else None,
                'caption': caption,
                'all_captions': captions
            }
        
        return {
            'image': image,
            'caption': captions[0] if captions else "",
            'all_captions': captions
        }

class CustomDatasetCreator:
    """
    Helper class to create custom datasets from various sources
    """
    
    @staticmethod
    def from_folder(image_folder: str, 
                   captions_file: str = None,
                   caption_format: str = "json") -> List[Dict]:
        """
        Create dataset from folder of images with captions
        
        Args:
            image_folder: Path to folder containing images
            captions_file: Path to captions file (JSON, CSV, or TXT)
            caption_format: Format of captions file
        
        Returns:
            List of dataset items
        """
        
        dataset_items = []
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = []
        
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        # Load captions if provided
        captions_dict = {}
        if captions_file and os.path.exists(captions_file):
            if caption_format.lower() == "json":
                with open(captions_file, 'r') as f:
                    captions_data = json.load(f)
                    
                # Handle different JSON formats
                if isinstance(captions_data, dict):
                    captions_dict = captions_data
                elif isinstance(captions_data, list):
                    for item in captions_data:
                        if 'image' in item and 'caption' in item:
                            captions_dict[item['image']] = item['caption']
            
            elif caption_format.lower() == "csv":
                df = pd.read_csv(captions_file)
                if 'image' in df.columns and 'caption' in df.columns:
                    captions_dict = dict(zip(df['image'], df['caption']))
        
        # Create dataset items
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            caption = captions_dict.get(image_file, f"Image: {image_file}")
            
            dataset_items.append({
                'image_path': image_path,
                'caption': caption,
                'image_name': image_file
            })
        
        return dataset_items
    
    @staticmethod
    def from_urls(urls_and_captions: List[Dict]) -> List[Dict]:
        """
        Create dataset from list of URLs and captions
        
        Args:
            urls_and_captions: List of {'url': str, 'caption': str} dicts
        
        Returns:
            List of dataset items
        """
        
        dataset_items = []
        
        for item in urls_and_captions:
            if 'url' in item and 'caption' in item:
                dataset_items.append({
                    'image_url': item['url'],
                    'caption': item['caption']
                })
        
        return dataset_items
    
    @staticmethod
    def save_dataset(dataset_items: List[Dict], 
                    output_file: str,
                    format: str = "json"):
        """
        Save dataset to file
        
        Args:
            dataset_items: List of dataset items
            output_file: Output file path
            format: Output format (json, csv)
        """
        
        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(dataset_items, f, indent=2)
        
        elif format.lower() == "csv":
            df = pd.DataFrame(dataset_items)
            df.to_csv(output_file, index=False)
        
        print(f"Dataset saved to {output_file} ({len(dataset_items)} items)")

# Example usage and dataset recommendations
RECOMMENDED_DATASETS = {
    "beginner": {
        "dataset": "flickr30k",
        "reason": "Smaller size, high quality annotations, good for learning",
        "samples": 1000
    },
    "research": {
        "dataset": "coco",
        "reason": "Standard benchmark, comprehensive evaluation possible",
        "samples": 5000
    },
    "production": {
        "dataset": "conceptual_captions",
        "reason": "Large scale, diverse content, real-world distribution",
        "samples": 50000
    },
    "multilingual": {
        "dataset": "wit",
        "reason": "Multiple languages, Wikipedia-based, factual content",
        "samples": 10000
    }
}