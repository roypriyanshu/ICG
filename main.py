#!/usr/bin/env python3
"""
Advanced Image Captioning System
Supports multiple state-of-the-art models and comprehensive evaluation
"""

import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from image_captioning_model import ImageCaptioningModel
from dataset_handler import COCOCaptionDataset, FlickrDataset
from evaluation_metrics import CaptionEvaluator
import json
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Image Captioning System')
    parser.add_argument('--model', type=str, default='blip', 
                       choices=['blip', 'vit_gpt2'],
                       help='Model architecture to use')
    parser.add_argument('--image', type=str, help='Path to image file or URL')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate model on dataset')
    parser.add_argument('--dataset', type=str, default='coco',
                       choices=['coco', 'flickr30k'],
                       help='Dataset to use for evaluation')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print(f"Loading {args.model} model...")
    model = ImageCaptioningModel(model_type=args.model)
    
    if args.image:
        # Single image captioning
        print(f"Generating caption for: {args.image}")
        
        try:
            # Generate single caption
            caption = model.generate_caption(args.image)
            print(f"Generated Caption: {caption}")
            
            # Generate multiple diverse captions
            multiple_captions = model.generate_multiple_captions(args.image, num_captions=3)
            print("\nMultiple Caption Variations:")
            for i, cap in enumerate(multiple_captions, 1):
                print(f"{i}. {cap}")
            
            # Visualize result
            model.visualize_prediction(args.image, caption)
            
        except Exception as e:
            print(f"Error processing image: {e}")
    
    elif args.evaluate:
        # Dataset evaluation
        print(f"Evaluating on {args.dataset} dataset...")
        evaluate_model(model, args.dataset, args.num_samples, args.output_dir)
    
    else:
        # Interactive mode
        interactive_mode(model)

def evaluate_model(model, dataset_name, num_samples, output_dir):
    """Evaluate model on specified dataset"""
    
    # Load dataset
    if dataset_name == 'coco':
        dataset = COCOCaptionDataset(split='validation', max_samples=num_samples)
    elif dataset_name == 'flickr30k':
        dataset = FlickrDataset(split='test', max_samples=num_samples)
    
    # Initialize evaluator
    evaluator = CaptionEvaluator()
    
    generated_captions = []
    reference_captions = []
    
    print("Generating captions...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        image = sample['image']
        ref_captions = sample['all_captions']
        
        # Generate caption
        try:
            gen_caption = model.generate_caption(image)
            generated_captions.append(gen_caption)
            reference_captions.append(ref_captions)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Evaluate
    print("Computing evaluation metrics...")
    avg_metrics, all_metrics = evaluator.evaluate_dataset(generated_captions, reference_captions)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, score in avg_metrics.items():
        print(f"{metric}: {score:.4f}")
    
    # Save results
    results = {
        'model': model.model_type,
        'dataset': dataset_name,
        'num_samples': len(generated_captions),
        'average_metrics': avg_metrics,
        'generated_captions': generated_captions[:10],  # Save first 10 for inspection
        'reference_captions': reference_captions[:10]
    }
    
    results_file = os.path.join(output_dir, f'{model.model_type}_{dataset_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

def interactive_mode(model):
    """Interactive mode for testing different images"""
    print("\n" + "="*60)
    print("Interactive Image Captioning Mode")
    print("Enter image path/URL or 'quit' to exit")
    print("="*60)
    
    while True:
        image_input = input("\nImage path/URL: ").strip()
        
        if image_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not image_input:
            continue
        
        try:
            # Generate caption
            caption = model.generate_caption(image_input)
            print(f"Caption: {caption}")
            
            # Ask if user wants multiple captions
            multi = input("Generate multiple captions? (y/n): ").strip().lower()
            if multi in ['y', 'yes']:
                captions = model.generate_multiple_captions(image_input, num_captions=3)
                print("\nAlternative captions:")
                for i, cap in enumerate(captions, 1):
                    print(f"{i}. {cap}")
            
            # Ask if user wants to visualize
            viz = input("Show image with caption? (y/n): ").strip().lower()
            if viz in ['y', 'yes']:
                model.visualize_prediction(image_input, caption)
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()