import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import json
import os
from tqdm import tqdm

class COCOCaptionDataset(Dataset):
    """
    COCO Caption Dataset handler for training and evaluation
    Uses the MS-COCO dataset which is the gold standard for image captioning
    """
    
    def __init__(self, split="train", max_samples=None, processor=None):
        self.split = split
        self.processor = processor
        
        print(f"Loading COCO {split} dataset...")
        # Load COCO dataset from HuggingFace
        self.dataset = load_dataset("ydshieh/coco_dataset_script", "2017", split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples from COCO {split} set")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get image
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get captions (COCO has multiple captions per image)
        captions = [ann['caption'] for ann in item['annotations']]
        
        # For training, we typically use the first caption
        # For evaluation, we might want all captions
        caption = captions[0] if captions else ""
        
        if self.processor:
            # Process image for model input
            processed = self.processor(image, text=caption, return_tensors="pt", padding=True, truncation=True)
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
            'caption': caption,
            'all_captions': captions
        }

class FlickrDataset(Dataset):
    """
    Alternative dataset: Flickr30k for image captioning
    Smaller but high-quality dataset
    """
    
    def __init__(self, split="train", processor=None):
        self.split = split
        self.processor = processor
        
        print(f"Loading Flickr30k {split} dataset...")
        self.dataset = load_dataset("nlphuji/flickr30k", split=split)
        print(f"Loaded {len(self.dataset)} samples from Flickr30k {split} set")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Flickr30k has 5 captions per image
        captions = item['caption']
        caption = captions[0] if isinstance(captions, list) else captions
        
        if self.processor:
            processed = self.processor(image, text=caption, return_tensors="pt", padding=True, truncation=True)
            return {
                'image': image,
                'pixel_values': processed['pixel_values'].squeeze(),
                'input_ids': processed['input_ids'].squeeze() if 'input_ids' in processed else None,
                'caption': caption,
                'all_captions': captions if isinstance(captions, list) else [captions]
            }
        
        return {
            'image': image,
            'caption': caption,
            'all_captions': captions if isinstance(captions, list) else [captions]
        }