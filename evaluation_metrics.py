import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import Counter
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class CaptionEvaluator:
    """
    Comprehensive evaluation metrics for image captioning with report generation
    Implements BLEU, METEOR, ROUGE-L, CIDEr, and additional metrics
    """
    
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def preprocess_caption(self, caption):
        """Preprocess caption for evaluation"""
        caption = caption.lower()
        caption = re.sub(r'[^\w\s]', '', caption)
        return caption.split()
    
    def compute_bleu(self, generated_caption, reference_captions, max_n=4):
        """Compute BLEU scores (BLEU-1 to BLEU-4)"""
        gen_tokens = self.preprocess_caption(generated_caption)
        ref_tokens = [self.preprocess_caption(ref) for ref in reference_captions]
        
        bleu_scores = {}
        for n in range(1, max_n + 1):
            weights = [1.0/n] * n + [0.0] * (4-n)
            score = sentence_bleu(
                ref_tokens, 
                gen_tokens, 
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            bleu_scores[f'BLEU-{n}'] = score
        
        return bleu_scores
    
    def compute_meteor(self, generated_caption, reference_captions):
        """Compute METEOR score"""
        gen_tokens = self.preprocess_caption(generated_caption)
        
        meteor_scores = []
        for ref_caption in reference_captions:
            ref_tokens = self.preprocess_caption(ref_caption)
            score = meteor_score([ref_tokens], gen_tokens)
            meteor_scores.append(score)
        
        return max(meteor_scores)
    
    def compute_rouge_l(self, generated_caption, reference_captions):
        """Compute ROUGE-L score (Longest Common Subsequence)"""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        gen_tokens = self.preprocess_caption(generated_caption)
        
        rouge_scores = []
        for ref_caption in reference_captions:
            ref_tokens = self.preprocess_caption(ref_caption)
            
            lcs_len = lcs_length(gen_tokens, ref_tokens)
            
            if len(gen_tokens) == 0 or len(ref_tokens) == 0:
                rouge_scores.append(0.0)
                continue
            
            precision = lcs_len / len(gen_tokens)
            recall = lcs_len / len(ref_tokens)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            rouge_scores.append(f1)
        
        return max(rouge_scores)
    
    def compute_cider(self, generated_caption, reference_captions):
        """Simplified CIDEr score computation"""
        def compute_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        gen_tokens = self.preprocess_caption(generated_caption)
        
        cider_scores = []
        for ref_caption in reference_captions:
            ref_tokens = self.preprocess_caption(ref_caption)
            
            total_score = 0
            for n in range(1, 5):
                gen_ngrams = Counter(compute_ngrams(gen_tokens, n))
                ref_ngrams = Counter(compute_ngrams(ref_tokens, n))
                
                overlap = sum((gen_ngrams & ref_ngrams).values())
                total_ngrams = sum(gen_ngrams.values())
                
                if total_ngrams > 0:
                    total_score += overlap / total_ngrams
            
            cider_scores.append(total_score / 4)
        
        return max(cider_scores)
    
    def compute_additional_metrics(self, generated_caption, reference_captions):
        """Compute additional evaluation metrics"""
        metrics = {}
        
        # Caption length metrics
        gen_length = len(generated_caption.split())
        ref_lengths = [len(ref.split()) for ref in reference_captions]
        avg_ref_length = np.mean(ref_lengths)
        
        metrics['caption_length'] = gen_length
        metrics['length_ratio'] = gen_length / avg_ref_length if avg_ref_length > 0 else 0
        
        # Vocabulary diversity
        gen_words = set(generated_caption.lower().split())
        ref_words = set()
        for ref in reference_captions:
            ref_words.update(ref.lower().split())
        
        metrics['vocabulary_overlap'] = len(gen_words & ref_words) / len(gen_words | ref_words) if len(gen_words | ref_words) > 0 else 0
        metrics['unique_words'] = len(gen_words)
        
        # Semantic similarity
        metrics['semantic_score'] = self._compute_semantic_similarity(generated_caption, reference_captions)
        
        return metrics
    
    def _compute_semantic_similarity(self, generated_caption, reference_captions):
        """Compute semantic similarity using word overlap"""
        gen_words = set(generated_caption.lower().split())
        
        similarities = []
        for ref in reference_captions:
            ref_words = set(ref.lower().split())
            
            intersection = len(gen_words & ref_words)
            union = len(gen_words | ref_words)
            jaccard = intersection / union if union > 0 else 0
            
            similarities.append(jaccard)
        
        return max(similarities) if similarities else 0
    
    def evaluate_caption(self, generated_caption, reference_captions):
        """Compute all evaluation metrics for a single caption"""
        metrics = {}
        
        # Standard metrics
        bleu_scores = self.compute_bleu(generated_caption, reference_captions)
        metrics.update(bleu_scores)
        
        metrics['METEOR'] = self.compute_meteor(generated_caption, reference_captions)
        metrics['ROUGE-L'] = self.compute_rouge_l(generated_caption, reference_captions)
        metrics['CIDEr'] = self.compute_cider(generated_caption, reference_captions)
        
        # Additional metrics
        additional_metrics = self.compute_additional_metrics(generated_caption, reference_captions)
        metrics.update(additional_metrics)
        
        return metrics
    
    def evaluate_dataset(self, generated_captions, reference_captions_list, save_report=True, report_dir="reports"):
        """
        Comprehensive evaluation with report generation
        """
        all_metrics = []
        detailed_results = []
        
        print("Computing evaluation metrics...")
        for i, (gen_cap, ref_caps) in enumerate(zip(generated_captions, reference_captions_list)):
            metrics = self.evaluate_caption(gen_cap, ref_caps)
            all_metrics.append(metrics)
            
            detailed_results.append({
                'sample_id': i,
                'generated_caption': gen_cap,
                'reference_captions': ref_caps,
                'metrics': metrics
            })
        
        # Compute statistics
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            avg_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        # Generate report if requested
        if save_report:
            report_data = self.generate_evaluation_report(
                avg_metrics, all_metrics, detailed_results, report_dir
            )
            return avg_metrics, all_metrics, report_data
        
        return avg_metrics, all_metrics
    
    def generate_evaluation_report(self, avg_metrics, all_metrics, detailed_results, report_dir="reports"):
        """Generate comprehensive evaluation report with plots"""
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"evaluation_report_{timestamp}"
        
        report_data = {
            'timestamp': timestamp,
            'summary_metrics': avg_metrics,
            'total_samples': len(all_metrics),
            'report_files': {}
        }
        
        # Generate plots
        plot_files = self._generate_evaluation_plots(all_metrics, report_dir, report_name)
        report_data['report_files']['plots'] = plot_files
        
        # Generate detailed analysis
        analysis_file = self._generate_detailed_analysis(avg_metrics, all_metrics, detailed_results, report_dir, report_name)
        report_data['report_files']['analysis'] = analysis_file
        
        # Generate summary report
        summary_file = self._generate_summary_report(avg_metrics, report_dir, report_name)
        report_data['report_files']['summary'] = summary_file
        
        # Save complete report
        report_file = os.path.join(report_dir, f"{report_name}_complete.json")
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📊 Evaluation report generated: {report_dir}/{report_name}_*")
        return report_data
    
    def _generate_evaluation_plots(self, all_metrics, report_dir, report_name):
        """Generate comprehensive evaluation plots"""
        plot_files = {}
        
        # Convert metrics to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # 1. Metrics Distribution Plot
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'Caption Length', 'Vocabulary Overlap']
        )
        
        metrics_to_plot = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'caption_length', 'vocabulary_overlap']
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for metric, (row, col) in zip(metrics_to_plot, positions):
            if metric in df.columns:
                fig.add_trace(
                    go.Histogram(x=df[metric], name=metric, showlegend=False),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Distribution of Evaluation Metrics",
            height=600,
            showlegend=False
        )
        
        plot_file = os.path.join(report_dir, f"{report_name}_distributions.html")
        fig.write_html(plot_file)
        plot_files['distributions'] = plot_file
        
        # 2. Metrics Correlation Heatmap
        correlation_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        corr_df = df[correlation_metrics].corr()
        
        fig_corr = px.imshow(
            corr_df,
            title="Correlation Matrix of Evaluation Metrics",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        plot_file = os.path.join(report_dir, f"{report_name}_correlation.html")
        fig_corr.write_html(plot_file)
        plot_files['correlation'] = plot_file
        
        # 3. BLEU Scores Comparison
        bleu_cols = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
        bleu_means = [df[col].mean() for col in bleu_cols]
        bleu_stds = [df[col].std() for col in bleu_cols]
        
        fig_bleu = go.Figure()
        fig_bleu.add_trace(go.Bar(
            x=bleu_cols,
            y=bleu_means,
            error_y=dict(type='data', array=bleu_stds),
            name='BLEU Scores'
        ))
        
        fig_bleu.update_layout(
            title="BLEU Scores (1-4) with Standard Deviation",
            xaxis_title="BLEU Metric",
            yaxis_title="Score"
        )
        
        plot_file = os.path.join(report_dir, f"{report_name}_bleu_comparison.html")
        fig_bleu.write_html(plot_file)
        plot_files['bleu_comparison'] = plot_file
        
        # 4. Caption Length Analysis
        fig_length = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Caption Length Distribution', 'Length vs BLEU-4']
        )
        
        fig_length.add_trace(
            go.Histogram(x=df['caption_length'], name='Caption Length'),
            row=1, col=1
        )
        
        fig_length.add_trace(
            go.Scatter(
                x=df['caption_length'], 
                y=df['BLEU-4'],
                mode='markers',
                name='Length vs BLEU-4'
            ),
            row=1, col=2
        )
        
        fig_length.update_layout(title="Caption Length Analysis", height=400)
        
        plot_file = os.path.join(report_dir, f"{report_name}_length_analysis.html")
        fig_length.write_html(plot_file)
        plot_files['length_analysis'] = plot_file
        
        # 5. Performance Summary Radar Chart
        main_metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        values = [df[metric].mean() for metric in main_metrics]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=main_metrics,
            fill='toself',
            name='Performance'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]
                )),
            title="Performance Summary (Main Metrics)"
        )
        
        plot_file = os.path.join(report_dir, f"{report_name}_performance_radar.html")
        fig_radar.write_html(plot_file)
        plot_files['performance_radar'] = plot_file
        
        return plot_files
    
    def _generate_detailed_analysis(self, avg_metrics, all_metrics, detailed_results, report_dir, report_name):
        """Generate detailed analysis report"""
        analysis_file = os.path.join(report_dir, f"{report_name}_detailed_analysis.json")
        
        # Compute additional statistics
        df = pd.DataFrame(all_metrics)
        
        analysis = {
            'summary_statistics': {},
            'performance_analysis': {},
            'best_worst_examples': {},
            'recommendations': []
        }
        
        # Summary statistics for each metric
        for metric in df.columns:
            analysis['summary_statistics'][metric] = {
                'mean': float(df[metric].mean()),
                'median': float(df[metric].median()),
                'std': float(df[metric].std()),
                'min': float(df[metric].min()),
                'max': float(df[metric].max()),
                'q25': float(df[metric].quantile(0.25)),
                'q75': float(df[metric].quantile(0.75))
            }
        
        # Performance analysis
        bleu4_mean = df['BLEU-4'].mean()
        meteor_mean = df['METEOR'].mean()
        cider_mean = df['CIDEr'].mean()
        
        analysis['performance_analysis'] = {
            'overall_performance': 'excellent' if bleu4_mean > 0.3 else 'good' if bleu4_mean > 0.2 else 'needs_improvement',
            'bleu4_performance': {
                'score': float(bleu4_mean),
                'benchmark': 'state-of-the-art' if bleu4_mean > 0.35 else 'competitive' if bleu4_mean > 0.25 else 'baseline'
            },
            'meteor_performance': {
                'score': float(meteor_mean),
                'benchmark': 'excellent' if meteor_mean > 0.28 else 'good' if meteor_mean > 0.22 else 'fair'
            },
            'cider_performance': {
                'score': float(cider_mean),
                'benchmark': 'excellent' if cider_mean > 1.0 else 'good' if cider_mean > 0.8 else 'fair'
            }
        }
        
        # Best and worst examples
        bleu4_scores = df['BLEU-4'].values
        best_indices = np.argsort(bleu4_scores)[-5:]  # Top 5
        worst_indices = np.argsort(bleu4_scores)[:5]  # Bottom 5
        
        analysis['best_worst_examples'] = {
            'best_examples': [
                {
                    'sample_id': int(idx),
                    'bleu4_score': float(bleu4_scores[idx]),
                    'generated_caption': detailed_results[idx]['generated_caption'],
                    'reference_captions': detailed_results[idx]['reference_captions']
                }
                for idx in best_indices
            ],
            'worst_examples': [
                {
                    'sample_id': int(idx),
                    'bleu4_score': float(bleu4_scores[idx]),
                    'generated_caption': detailed_results[idx]['generated_caption'],
                    'reference_captions': detailed_results[idx]['reference_captions']
                }
                for idx in worst_indices
            ]
        }
        
        # Generate recommendations
        recommendations = []
        
        if bleu4_mean < 0.25:
            recommendations.append("Consider fine-tuning the model on domain-specific data")
        
        if df['caption_length'].mean() > 15:
            recommendations.append("Captions are relatively long - consider adjusting max_length parameter")
        
        if df['vocabulary_overlap'].mean() < 0.3:
            recommendations.append("Low vocabulary overlap with references - consider vocabulary expansion")
        
        if df['METEOR'].mean() < 0.22:
            recommendations.append("Low METEOR score suggests semantic issues - review training data quality")
        
        analysis['recommendations'] = recommendations
        
        # Save analysis
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis_file
    
    def _generate_summary_report(self, avg_metrics, report_dir, report_name):
        """Generate human-readable summary report"""
        summary_file = os.path.join(report_dir, f"{report_name}_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("IMAGE CAPTIONING EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total samples evaluated: {len(avg_metrics)}\n\n")
            
            f.write("MAIN METRICS SUMMARY:\n")
            f.write("-" * 30 + "\n")
            
            main_metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
            for metric in main_metrics:
                if metric in avg_metrics:
                    mean_val = avg_metrics[metric]['mean']
                    std_val = avg_metrics[metric]['std']
                    f.write(f"{metric:12}: {mean_val:.4f} ± {std_val:.4f}\n")
            
            f.write("\nBLEU SCORES BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            
            bleu_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
            for metric in bleu_metrics:
                if metric in avg_metrics:
                    mean_val = avg_metrics[metric]['mean']
                    f.write(f"{metric:12}: {mean_val:.4f}\n")
            
            f.write("\nADDITIONAL METRICS:\n")
            f.write("-" * 30 + "\n")
            
            additional_metrics = ['caption_length', 'vocabulary_overlap', 'semantic_score']
            for metric in additional_metrics:
                if metric in avg_metrics:
                    mean_val = avg_metrics[metric]['mean']
                    f.write(f"{metric:18}: {mean_val:.4f}\n")
            
            # Performance assessment
            f.write("\nPERFORMANCE ASSESSMENT:\n")
            f.write("-" * 30 + "\n")
            
            bleu4_score = avg_metrics.get('BLEU-4', {}).get('mean', 0)
            if bleu4_score > 0.35:
                f.write("Overall Performance: EXCELLENT (State-of-the-art)\n")
            elif bleu4_score > 0.25:
                f.write("Overall Performance: GOOD (Competitive)\n")
            elif bleu4_score > 0.15:
                f.write("Overall Performance: FAIR (Baseline)\n")
            else:
                f.write("Overall Performance: NEEDS IMPROVEMENT\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        return summary_file
    
    def generate_comparison_report(self, results_list, model_names, report_dir="reports"):
        """Generate comparison report between multiple models"""
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_name = f"model_comparison_{timestamp}"
        
        # Create comparison DataFrame
        comparison_data = []
        for i, (avg_metrics, model_name) in enumerate(zip(results_list, model_names)):
            row = {'Model': model_name}
            for metric_name, metric_stats in avg_metrics.items():
                if isinstance(metric_stats, dict) and 'mean' in metric_stats:
                    row[metric_name] = metric_stats['mean']
                else:
                    row[metric_name] = metric_stats
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Generate comparison plots
        main_metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=main_metrics
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, (row, col) in zip(main_metrics, positions):
            if metric in df_comparison.columns:
                fig.add_trace(
                    go.Bar(
                        x=df_comparison['Model'],
                        y=df_comparison[metric],
                        name=metric,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=600
        )
        
        plot_file = os.path.join(report_dir, f"{comparison_name}_comparison.html")
        fig.write_html(plot_file)
        
        # Save comparison data
        csv_file = os.path.join(report_dir, f"{comparison_name}_data.csv")
        df_comparison.to_csv(csv_file, index=False)
        
        print(f"📊 Model comparison report generated: {report_dir}/{comparison_name}_*")
        
        return {
            'comparison_plot': plot_file,
            'comparison_data': csv_file,
            'dataframe': df_comparison
        }