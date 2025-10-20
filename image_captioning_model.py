import torch
import torch.nn as nn
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    GPT2LMHeadModel, GPT2Tokenizer,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

class ImageCaptioningModel:
    """
    Advanced Image Captioning Model using state-of-the-art architectures
    Supports multiple model backends: BLIP, ViT-GPT2, and custom implementations
    """
    
    def __init__(self, model_type="blip", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        print(f"Initializing {model_type} model on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the specified model architecture"""
        if self.model_type == "blip":
            self._load_blip_model()
        elif self.model_type == "vit_gpt2":
            self._load_vit_gpt2_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_blip_model(self):
        """Load BLIP (Bootstrapping Language-Image Pre-training) model"""
        model_name = "Salesforce/blip-image-captioning-large"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_vit_gpt2_model(self):
        """Load Vision Transformer + GPT2 model"""
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  
  
    def preprocess_image(self, image_input):
        """
        Preprocess image from various input formats
        Args:
            image_input: PIL Image, numpy array, or URL string
        Returns:
            PIL Image object
        """
        if isinstance(image_input, str):
            # Handle URL
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input)
                image = Image.open(BytesIO(response.content))
            else:
                # Handle file path
                image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Unsupported image input format")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def generate_caption(self, image_input, max_length=50, num_beams=5, temperature=1.0):
        """
        Generate caption for input image
        Args:
            image_input: Image in various formats
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
        Returns:
            Generated caption string
        """
        image = self.preprocess_image(image_input)
        
        with torch.no_grad():
            if self.model_type == "blip":
                return self._generate_blip_caption(image, max_length, num_beams)
            elif self.model_type == "vit_gpt2":
                return self._generate_vit_gpt2_caption(image, max_length, num_beams)
    
    def _generate_blip_caption(self, image, max_length, num_beams):
        """Generate caption using BLIP model"""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        out = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False
        )
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _generate_vit_gpt2_caption(self, image, max_length, num_beams):
        """Generate caption using ViT-GPT2 model"""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        output_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False
        )
        
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption    

    def generate_multiple_captions(self, image_input, num_captions=3, max_length=50):
        """Generate multiple diverse captions for the same image"""
        image = self.preprocess_image(image_input)
        captions = []
        
        with torch.no_grad():
            if self.model_type == "blip":
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                
                for i in range(num_captions):
                    out = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=5,
                        do_sample=True,
                        temperature=0.7 + i * 0.3,
                        early_stopping=True
                    )
                    caption = self.processor.decode(out[0], skip_special_tokens=True)
                    captions.append(caption)
            
            elif self.model_type == "vit_gpt2":
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
                
                for i in range(num_captions):
                    output_ids = self.model.generate(
                        pixel_values,
                        max_length=max_length,
                        num_beams=5,
                        do_sample=True,
                        temperature=0.7 + i * 0.3,
                        early_stopping=True
                    )
                    caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    captions.append(caption)
        
        return captions
    
    def visualize_prediction(self, image_input, caption=None):
        """Visualize image with generated caption"""
        image = self.preprocess_image(image_input)
        
        if caption is None:
            caption = self.generate_caption(image)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Generated Caption: {caption}", fontsize=14, wrap=True)
        plt.tight_layout()
        plt.show()
        
        return caption