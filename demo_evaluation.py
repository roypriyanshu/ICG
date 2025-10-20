#!/usr/bin/env python3
"""
Demo script for enhanced evaluation system
Shows comprehensive evaluation with report generation
"""

from evaluation_metrics import CaptionEvaluator
from report_generator import ReportGenerator
import json

def demo_evaluation():
    """Demonstrate the enhanced evaluation system"""
    
    print("🔍 Image Captioning Evaluation Demo")
    print("=" * 50)
    
    # Sample data for demonstration
    generated_captions = [
        "a dog sitting on grass in a park",
        "a woman walking down the street",
        "a red car parked on the road",
        "children playing in the playground",
        "a beautiful sunset over the ocean",
        "a cat sleeping on a couch",
        "people eating at a restaurant",
        "a bird flying in the sky",
        "a train moving through the countryside",
        "flowers blooming in a garden"
    ]
    
    reference_captions = [
        ["a brown dog sits in the grass", "dog in a park setting", "canine resting outdoors"],
        ["woman walking on sidewalk", "person strolling down street", "lady walking in city"],
        ["red vehicle parked", "car on the roadside", "automobile in parking area"],
        ["kids at playground", "children having fun outside", "young people playing"],
        ["sunset over water", "beautiful evening sky", "sun setting on horizon"],
        ["cat resting indoors", "feline sleeping comfortably", "pet cat on furniture"],
        ["diners at restaurant", "people having meal", "customers eating food"],
        ["bird in flight", "flying creature in air", "avian soaring through sky"],
        ["locomotive in motion", "train traveling", "railway transport moving"],
        ["garden with flowers", "blooming plants", "colorful flowers in bloom"]
    ]
    
    # Initialize evaluator
    evaluator = CaptionEvaluator()
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    avg_metrics, all_metrics, report_data = evaluator.evaluate_dataset(
        generated_captions, 
        reference_captions, 
        save_report=True, 
        report_dir="reports"
    )
    
    # Display key results
    print("\n📊 Key Results:")
    print("-" * 30)
    
    main_metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
    for metric in main_metrics:
        if metric in avg_metrics:
            mean_val = avg_metrics[metric]['mean']
            std_val = avg_metrics[metric]['std']
            print(f"{metric:12}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Show report files generated
    print(f"\n📁 Report Files Generated:")
    print("-" * 30)
    
    if 'report_files' in report_data:
        for file_type, file_path in report_data['report_files'].items():
            if isinstance(file_path, dict):
                print(f"{file_type.upper()}:")
                for sub_type, sub_path in file_path.items():
                    print(f"  - {sub_type}: {sub_path}")
            else:
                print(f"- {file_type}: {file_path}")
    
    # Generate additional comprehensive report
    print("\n🔧 Generating comprehensive report...")
    report_gen = ReportGenerator()
    
    comprehensive_report = report_gen.generate_comprehensive_report(
        avg_metrics=avg_metrics,
        all_metrics=all_metrics,
        model_name="BLIP-Demo",
        dataset_name="Sample-Data",
        additional_info={
            "demo_mode": True,
            "sample_size": len(generated_captions)
        }
    )
    
    print("\n✅ Demo completed!")
    print(f"📊 Check the 'reports' folder for all generated files")
    print(f"🌐 Open the HTML report for interactive visualization")
    
    return report_data, comprehensive_report

if __name__ == "__main__":
    demo_evaluation()