#!/usr/bin/env python3
"""
Streamlit UI for Image Captioning and Fine-tuning
Complete interface for training, evaluation, and inference
"""

import streamlit as st
import torch
from PIL import Image
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64

# Import our modules
try:
    from image_captioning_model import ImageCaptioningModel
    from fine_tuning_trainer import ImageCaptionFineTuner
    from evaluation_metrics import CaptionEvaluator
    from dataset_handler import COCOCaptionDataset, FlickrDataset
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Advanced Image Captioning System",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ Advanced Image Captioning System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📸 Image Captioning", "🔧 Model Fine-tuning", "📊 Evaluation", "📚 Dataset Explorer"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📸 Image Captioning":
        show_captioning_page()
    elif page == "🔧 Model Fine-tuning":
        show_finetuning_page()
    elif page == "📊 Evaluation":
        show_evaluation_page()
    elif page == "📚 Dataset Explorer":
        show_dataset_page()

def show_home_page():
    """Home page with system overview"""
    
    st.markdown("## Welcome to the Advanced Image Captioning System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - **Multiple Models**: BLIP, ViT-GPT2
        - **Fine-tuning**: LoRA, QLoRA, Full
        - **Datasets**: COCO, Flickr30k, Custom
        - **Evaluation**: BLEU, METEOR, CIDEr
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        1. **Caption Images**: Upload and get captions
        2. **Fine-tune Models**: Train on custom data
        3. **Evaluate Performance**: Comprehensive metrics
        4. **Explore Datasets**: Browse training data
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Supported Datasets
        - **MS-COCO**: 118k images, 5 captions each
        - **Flickr30k**: 31k images, high quality
        - **Custom**: Upload your own dataset
        - **Real-time**: Process live images
        """)
    
    # System status
    st.markdown("## System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gpu_available = torch.cuda.is_available()
        st.metric("GPU Available", "✅ Yes" if gpu_available else "❌ No")
    
    with col2:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        st.metric("GPU Count", device_count)
    
    with col3:
        st.metric("Models Available", "2 (BLIP, ViT-GPT2)")
    
    with col4:
        st.metric("Datasets", "3 (COCO, Flickr30k, Custom)")

def show_captioning_page():
    """Image captioning interface"""
    
    st.markdown("## 📸 Image Captioning")
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_type = st.selectbox(
            "Select Model:",
            ["blip", "vit_gpt2"],
            help="BLIP generally provides better quality captions"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_length = st.slider("Max Caption Length", 10, 100, 50)
            num_beams = st.slider("Beam Search Size", 1, 10, 5)
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0)
            num_captions = st.slider("Number of Captions", 1, 5, 1)
    
    with col2:
        # Image input options
        input_method = st.radio(
            "Image Input Method:",
            ["Upload File", "URL", "Camera", "Sample Images"]
        )
        
        image = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
        
        elif input_method == "URL":
            url = st.text_input("Enter image URL:")
            if url:
                try:
                    import requests
                    response = requests.get(url)
                    image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        
        elif input_method == "Camera":
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                image = Image.open(camera_image)
        
        elif input_method == "Sample Images":
            sample_urls = {
                "Dog": "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=500",
                "Landscape": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",
                "Food": "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e?w=500",
                "City": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=500"
            }
            
            selected_sample = st.selectbox("Choose sample image:", list(sample_urls.keys()))
            if st.button("Load Sample"):
                try:
                    import requests
                    response = requests.get(sample_urls[selected_sample])
                    image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    st.error(f"Error loading sample: {e}")
    
    # Process image if available
    if image:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        
        with col2:
            if st.button("Generate Caption(s)", type="primary"):
                with st.spinner("Loading model and generating captions..."):
                    try:
                        # Initialize model
                        model = ImageCaptioningModel(model_type=model_type)
                        
                        if num_captions == 1:
                            # Single caption
                            caption = model.generate_caption(
                                image, 
                                max_length=max_length, 
                                num_beams=num_beams,
                                temperature=temperature
                            )
                            
                            st.markdown("### Generated Caption:")
                            st.markdown(f'<div class="success-box"><strong>{caption}</strong></div>', 
                                      unsafe_allow_html=True)
                        
                        else:
                            # Multiple captions
                            captions = model.generate_multiple_captions(
                                image, 
                                num_captions=num_captions,
                                max_length=max_length
                            )
                            
                            st.markdown("### Generated Captions:")
                            for i, caption in enumerate(captions, 1):
                                st.markdown(f"**{i}.** {caption}")
                        
                        # Caption analysis
                        st.markdown("### Caption Analysis:")
                        caption_text = caption if num_captions == 1 else captions[0]
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Word Count", len(caption_text.split()))
                        with col_b:
                            st.metric("Character Count", len(caption_text))
                        with col_c:
                            st.metric("Sentences", caption_text.count('.') + caption_text.count('!') + caption_text.count('?'))
                    
                    except Exception as e:
                        st.error(f"Error generating caption: {e}")

def show_finetuning_page():
    """Fine-tuning interface"""
    
    st.markdown("## 🔧 Model Fine-tuning")
    
    # Fine-tuning configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Model Configuration")
        
        base_model = st.selectbox(
            "Base Model:",
            [
                "Salesforce/blip-image-captioning-base",
                "Salesforce/blip-image-captioning-large",
                "nlpconnect/vit-gpt2-image-captioning"
            ]
        )
        
        fine_tuning_method = st.selectbox(
            "Fine-tuning Method:",
            ["lora", "qlora", "full"],
            help="LoRA is recommended for efficiency"
        )
        
        dataset_name = st.selectbox(
            "Dataset:",
            ["coco", "flickr30k", "custom"],
            help="Choose dataset for fine-tuning"
        )
    
    with col2:
        st.markdown("### Training Parameters")
        
        num_epochs = st.slider("Number of Epochs", 1, 10, 3)
        batch_size = st.slider("Batch Size", 1, 32, 8)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
            value=5e-5,
            format_func=lambda x: f"{x:.0e}"
        )
        max_samples = st.slider("Max Training Samples", 100, 10000, 1000)
        
        use_wandb = st.checkbox("Use Weights & Biases logging")
    
    # Custom dataset upload
    if dataset_name == "custom":
        st.markdown("### Custom Dataset")
        st.info("Upload a JSON file with format: [{'image_path': 'path/to/image.jpg', 'caption': 'description'}]")
        
        uploaded_dataset = st.file_uploader(
            "Upload dataset JSON:",
            type=['json']
        )
        
        if uploaded_dataset:
            try:
                dataset_data = json.load(uploaded_dataset)
                st.success(f"Loaded {len(dataset_data)} samples")
                
                # Show sample
                if dataset_data:
                    st.markdown("**Sample entry:**")
                    st.json(dataset_data[0])
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    
    # Start fine-tuning
    if st.button("Start Fine-tuning", type="primary"):
        if dataset_name == "custom" and 'uploaded_dataset' not in locals():
            st.error("Please upload a custom dataset first")
            return
        
        with st.spinner("Initializing fine-tuning..."):
            try:
                # Initialize fine-tuner
                fine_tuner = ImageCaptionFineTuner(
                    base_model=base_model,
                    fine_tuning_method=fine_tuning_method
                )
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Start fine-tuning
                status_text.text("Starting fine-tuning process...")
                
                output_dir = fine_tuner.fine_tune(
                    dataset_name=dataset_name,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_samples=max_samples,
                    use_wandb=use_wandb
                )
                
                progress_bar.progress(100)
                status_text.text("Fine-tuning completed!")
                
                st.success(f"Model fine-tuned successfully! Saved to: {output_dir}")
                
                # Show training info
                with open(f"{output_dir}/training_info.json", "r") as f:
                    training_info = json.load(f)
                
                st.json(training_info)
            
            except Exception as e:
                st.error(f"Fine-tuning failed: {e}")

def show_evaluation_page():
    """Model evaluation interface"""
    
    st.markdown("## 📊 Model Evaluation")
    
    # Evaluation configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_path = st.text_input(
            "Model Path:",
            value="Salesforce/blip-image-captioning-base",
            help="Path to model (local or HuggingFace)"
        )
        
        dataset_name = st.selectbox(
            "Evaluation Dataset:",
            ["coco", "flickr30k"],
            help="Dataset for evaluation"
        )
        
        num_samples = st.slider("Number of Samples", 10, 1000, 100)
    
    with col2:
        metrics_to_compute = st.multiselect(
            "Metrics to Compute:",
            ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"],
            default=["BLEU-4", "METEOR", "CIDEr"]
        )
        
        save_results = st.checkbox("Save Results", value=True)
        show_examples = st.checkbox("Show Example Predictions", value=True)
    
    if st.button("Start Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            try:
                # Initialize model and evaluator
                model = ImageCaptioningModel(model_type="blip")  # Adjust based on model_path
                evaluator = CaptionEvaluator()
                
                # Load dataset
                if dataset_name == "coco":
                    dataset = COCOCaptionDataset(split="validation", max_samples=num_samples)
                else:
                    dataset = FlickrDataset(split="test", max_samples=num_samples)
                
                # Run evaluation
                progress_bar = st.progress(0)
                generated_captions = []
                reference_captions = []
                
                for i in range(len(dataset)):
                    sample = dataset[i]
                    image = sample['image']
                    ref_caps = sample['all_captions']
                    
                    # Generate caption
                    gen_caption = model.generate_caption(image)
                    generated_captions.append(gen_caption)
                    reference_captions.append(ref_caps)
                    
                    progress_bar.progress((i + 1) / len(dataset))
                
                # Compute metrics with report generation
                avg_metrics, all_metrics, report_data = evaluator.evaluate_dataset(
                    generated_captions, reference_captions, save_report=True, report_dir="reports"
                )
                
                # Display results
                st.markdown("### Evaluation Results")
                
                # Main metrics summary
                main_metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
                col1, col2, col3, col4 = st.columns(4)
                
                for i, metric in enumerate(main_metrics):
                    if metric in avg_metrics:
                        with [col1, col2, col3, col4][i]:
                            mean_val = avg_metrics[metric]['mean']
                            std_val = avg_metrics[metric]['std']
                            st.metric(
                                metric, 
                                f"{mean_val:.4f}",
                                delta=f"±{std_val:.4f}"
                            )
                
                # Detailed metrics table
                st.markdown("#### Detailed Metrics")
                metrics_summary = []
                for metric_name, stats in avg_metrics.items():
                    if isinstance(stats, dict):
                        metrics_summary.append({
                            'Metric': metric_name,
                            'Mean': f"{stats['mean']:.4f}",
                            'Std': f"{stats['std']:.4f}",
                            'Min': f"{stats['min']:.4f}",
                            'Max': f"{stats['max']:.4f}",
                            'Median': f"{stats['median']:.4f}"
                        })
                
                metrics_df = pd.DataFrame(metrics_summary)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Display generated plots
                st.markdown("#### Evaluation Plots")
                
                if 'plots' in report_data['report_files']:
                    plot_files = report_data['report_files']['plots']
                    
                    # Show performance radar chart
                    if 'performance_radar' in plot_files:
                        st.markdown("**Performance Summary**")
                        with open(plot_files['performance_radar'], 'r') as f:
                            st.components.v1.html(f.read(), height=500)
                    
                    # Show distributions
                    if 'distributions' in plot_files:
                        st.markdown("**Metrics Distributions**")
                        with open(plot_files['distributions'], 'r') as f:
                            st.components.v1.html(f.read(), height=600)
                    
                    # Show correlation matrix
                    if 'correlation' in plot_files:
                        st.markdown("**Metrics Correlation**")
                        with open(plot_files['correlation'], 'r') as f:
                            st.components.v1.html(f.read(), height=500)
                
                # Report files download
                st.markdown("#### Download Reports")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if 'summary' in report_data['report_files']:
                        with open(report_data['report_files']['summary'], 'r') as f:
                            st.download_button(
                                "📄 Summary Report",
                                f.read(),
                                file_name=f"summary_{report_data['timestamp']}.txt"
                            )
                
                with col_b:
                    if 'analysis' in report_data['report_files']:
                        with open(report_data['report_files']['analysis'], 'r') as f:
                            st.download_button(
                                "📊 Detailed Analysis",
                                f.read(),
                                file_name=f"analysis_{report_data['timestamp']}.json"
                            )
                
                with col_c:
                    # Create CSV export of metrics
                    csv_data = pd.DataFrame(all_metrics).to_csv(index=False)
                    st.download_button(
                        "📈 Raw Metrics CSV",
                        csv_data,
                        file_name=f"metrics_{report_data['timestamp']}.csv"
                    )
                
                # Show examples if requested
                if show_examples:
                    st.markdown("### Example Predictions")
                    
                    for i in range(min(5, len(generated_captions))):
                        with st.expander(f"Example {i+1}"):
                            col_a, col_b = st.columns([1, 2])
                            
                            with col_a:
                                sample = dataset[i]
                                st.image(sample['image'], use_column_width=True)
                            
                            with col_b:
                                st.markdown(f"**Generated:** {generated_captions[i]}")
                                st.markdown("**References:**")
                                for j, ref in enumerate(reference_captions[i]):
                                    st.markdown(f"  {j+1}. {ref}")
                
                # Save results if requested
                if save_results:
                    results = {
                        "model_path": model_path,
                        "dataset": dataset_name,
                        "num_samples": num_samples,
                        "metrics": avg_metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
                    
                    st.success(f"Results saved to {results_file}")
            
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

def show_dataset_page():
    """Dataset exploration interface"""
    
    st.markdown("## 📚 Dataset Explorer")
    
    dataset_name = st.selectbox(
        "Select Dataset:",
        ["coco", "flickr30k"]
    )
    
    split = st.selectbox(
        "Dataset Split:",
        ["train", "validation", "test"]
    )
    
    max_samples = st.slider("Max Samples to Load", 10, 1000, 100)
    
    if st.button("Load Dataset"):
        with st.spinner("Loading dataset..."):
            try:
                if dataset_name == "coco":
                    dataset = COCOCaptionDataset(split=split, max_samples=max_samples)
                else:
                    dataset = FlickrDataset(split=split, max_samples=max_samples)
                
                st.success(f"Loaded {len(dataset)} samples from {dataset_name} {split} set")
                
                # Dataset statistics
                st.markdown("### Dataset Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", len(dataset))
                with col2:
                    st.metric("Dataset", dataset_name.upper())
                with col3:
                    st.metric("Split", split.title())
                with col4:
                    st.metric("Captions per Image", "5")
                
                # Sample browser
                st.markdown("### Sample Browser")
                
                sample_idx = st.slider("Sample Index", 0, len(dataset)-1, 0)
                sample = dataset[sample_idx]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(sample['image'], caption=f"Sample {sample_idx}", use_column_width=True)
                
                with col2:
                    st.markdown("**Captions:**")
                    for i, caption in enumerate(sample['all_captions']):
                        st.markdown(f"{i+1}. {caption}")
                
                # Caption length analysis
                if st.checkbox("Analyze Caption Lengths"):
                    caption_lengths = []
                    for i in range(min(100, len(dataset))):
                        sample = dataset[i]
                        for caption in sample['all_captions']:
                            caption_lengths.append(len(caption.split()))
                    
                    fig = px.histogram(
                        x=caption_lengths,
                        title="Caption Length Distribution (words)",
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()