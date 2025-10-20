#!/usr/bin/env python3
"""
Comprehensive Report Generator for Image Captioning Evaluation
Generates detailed reports with plots and analysis
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from jinja2 import Template

class ReportGenerator:
    """
    Advanced report generator for image captioning evaluation results
    """
    
    def __init__(self, report_dir="reports"):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_comprehensive_report(self, 
                                    avg_metrics, 
                                    all_metrics, 
                                    model_name="Unknown",
                                    dataset_name="Unknown",
                                    additional_info=None):
        """
        Generate a comprehensive evaluation report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{model_name}_{dataset_name}_{timestamp}"
        
        # Create report structure
        report_data = {
            'metadata': {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'timestamp': timestamp,
                'total_samples': len(all_metrics),
                'additional_info': additional_info or {}
            },
            'metrics': avg_metrics,
            'files': {}
        }
        
        # Generate all report components
        report_data['files']['plots'] = self._generate_all_plots(all_metrics, report_name)
        report_data['files']['analysis'] = self._generate_statistical_analysis(avg_metrics, all_metrics, report_name)
        report_data['files']['html_report'] = self._generate_html_report(report_data, report_name)
        report_data['files']['pdf_summary'] = self._generate_pdf_summary(avg_metrics, all_metrics, report_name)
        
        # Save master report file
        master_file = os.path.join(self.report_dir, f"{report_name}_master.json")
        with open(master_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📊 Comprehensive report generated: {self.report_dir}/{report_name}_*")
        return report_data
    
    def _generate_all_plots(self, all_metrics, report_name):
        """Generate all evaluation plots"""
        df = pd.DataFrame(all_metrics)
        plot_files = {}
        
        # 1. Main Metrics Dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['BLEU Scores', 'Semantic Metrics', 'Length Analysis', 'Quality Distribution'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # BLEU scores
        bleu_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
        bleu_values = [df[metric].mean() for metric in bleu_metrics if metric in df.columns]
        bleu_labels = [metric for metric in bleu_metrics if metric in df.columns]
        
        fig.add_trace(
            go.Bar(x=bleu_labels, y=bleu_values, name='BLEU Scores', showlegend=False),
            row=1, col=1
        )
        
        # Semantic metrics
        semantic_metrics = ['METEOR', 'ROUGE-L', 'CIDEr']
        semantic_values = [df[metric].mean() for metric in semantic_metrics if metric in df.columns]
        semantic_labels = [metric for metric in semantic_metrics if metric in df.columns]
        
        fig.add_trace(
            go.Bar(x=semantic_labels, y=semantic_values, name='Semantic Metrics', showlegend=False),
            row=1, col=2
        )
        
        # Length analysis
        if 'caption_length' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['caption_length'], name='Caption Length', showlegend=False),
                row=2, col=1
            )
        
        # Quality distribution (BLEU-4)
        if 'BLEU-4' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['BLEU-4'], name='BLEU-4 Distribution', showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Evaluation Dashboard - {report_name}",
            height=800,
            showlegend=False
        )
        
        dashboard_file = os.path.join(self.report_dir, f"{report_name}_dashboard.html")
        fig.write_html(dashboard_file)
        plot_files['dashboard'] = dashboard_file
        
        # 2. Correlation Heatmap
        correlation_metrics = [col for col in df.columns if col in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']]
        if len(correlation_metrics) > 1:
            corr_matrix = df[correlation_metrics].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Metrics Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                text_auto=True
            )
            
            corr_file = os.path.join(self.report_dir, f"{report_name}_correlation.html")
            fig_corr.write_html(corr_file)
            plot_files['correlation'] = corr_file
        
        # 3. Performance Radar Chart
        main_metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        radar_values = []
        radar_labels = []
        
        for metric in main_metrics:
            if metric in df.columns:
                radar_values.append(df[metric].mean())
                radar_labels.append(metric)
        
        if radar_values:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=radar_labels,
                fill='toself',
                name='Performance'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(radar_values) * 1.2]
                    )
                ),
                title="Performance Radar Chart"
            )
            
            radar_file = os.path.join(self.report_dir, f"{report_name}_radar.html")
            fig_radar.write_html(radar_file)
            plot_files['radar'] = radar_file
        
        # 4. Box Plots for Distribution Analysis
        fig_box = make_subplots(
            rows=2, cols=2,
            subplot_titles=['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        )
        
        box_metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, (row, col) in zip(box_metrics, positions):
            if metric in df.columns:
                fig_box.add_trace(
                    go.Box(y=df[metric], name=metric, showlegend=False),
                    row=row, col=col
                )
        
        fig_box.update_layout(
            title="Metrics Distribution Analysis",
            height=600
        )
        
        box_file = os.path.join(self.report_dir, f"{report_name}_distributions.html")
        fig_box.write_html(box_file)
        plot_files['distributions'] = box_file
        
        # 5. Matplotlib plots for PDF inclusion
        self._generate_matplotlib_plots(df, report_name)
        
        return plot_files
    
    def _generate_matplotlib_plots(self, df, report_name):
        """Generate matplotlib plots for PDF reports"""
        
        # Main metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Evaluation Results - {report_name}', fontsize=16)
        
        # BLEU scores
        bleu_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
        bleu_values = [df[metric].mean() for metric in bleu_metrics if metric in df.columns]
        bleu_labels = [metric for metric in bleu_metrics if metric in df.columns]
        
        axes[0,0].bar(bleu_labels, bleu_values, color='skyblue')
        axes[0,0].set_title('BLEU Scores')
        axes[0,0].set_ylabel('Score')
        
        # Other metrics
        other_metrics = ['METEOR', 'ROUGE-L', 'CIDEr']
        other_values = [df[metric].mean() for metric in other_metrics if metric in df.columns]
        other_labels = [metric for metric in other_metrics if metric in df.columns]
        
        axes[0,1].bar(other_labels, other_values, color='lightcoral')
        axes[0,1].set_title('Semantic Metrics')
        axes[0,1].set_ylabel('Score')
        
        # Caption length distribution
        if 'caption_length' in df.columns:
            axes[1,0].hist(df['caption_length'], bins=20, color='lightgreen', alpha=0.7)
            axes[1,0].set_title('Caption Length Distribution')
            axes[1,0].set_xlabel('Length (words)')
            axes[1,0].set_ylabel('Frequency')
        
        # BLEU-4 distribution
        if 'BLEU-4' in df.columns:
            axes[1,1].hist(df['BLEU-4'], bins=20, color='gold', alpha=0.7)
            axes[1,1].set_title('BLEU-4 Score Distribution')
            axes[1,1].set_xlabel('BLEU-4 Score')
            axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.report_dir, f"{report_name}_summary_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    def _generate_statistical_analysis(self, avg_metrics, all_metrics, report_name):
        """Generate detailed statistical analysis"""
        
        analysis = {
            'descriptive_statistics': {},
            'performance_benchmarks': {},
            'outlier_analysis': {},
            'recommendations': []
        }
        
        df = pd.DataFrame(all_metrics)
        
        # Descriptive statistics
        for metric in df.columns:
            analysis['descriptive_statistics'][metric] = {
                'mean': float(df[metric].mean()),
                'median': float(df[metric].median()),
                'std': float(df[metric].std()),
                'min': float(df[metric].min()),
                'max': float(df[metric].max()),
                'q25': float(df[metric].quantile(0.25)),
                'q75': float(df[metric].quantile(0.75)),
                'skewness': float(df[metric].skew()),
                'kurtosis': float(df[metric].kurtosis())
            }
        
        # Performance benchmarks
        bleu4_mean = df['BLEU-4'].mean() if 'BLEU-4' in df.columns else 0
        meteor_mean = df['METEOR'].mean() if 'METEOR' in df.columns else 0
        cider_mean = df['CIDEr'].mean() if 'CIDEr' in df.columns else 0
        
        analysis['performance_benchmarks'] = {
            'bleu4_benchmark': self._get_performance_level(bleu4_mean, [0.15, 0.25, 0.35]),
            'meteor_benchmark': self._get_performance_level(meteor_mean, [0.18, 0.25, 0.32]),
            'cider_benchmark': self._get_performance_level(cider_mean, [0.6, 0.9, 1.2]),
            'overall_grade': self._calculate_overall_grade(bleu4_mean, meteor_mean, cider_mean)
        }
        
        # Outlier analysis
        for metric in ['BLEU-4', 'METEOR', 'CIDEr']:
            if metric in df.columns:
                Q1 = df[metric].quantile(0.25)
                Q3 = df[metric].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[metric] < lower_bound) | (df[metric] > upper_bound)]
                
                analysis['outlier_analysis'][metric] = {
                    'num_outliers': len(outliers),
                    'outlier_percentage': len(outliers) / len(df) * 100,
                    'outlier_indices': outliers.index.tolist()
                }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(df)
        
        # Save analysis
        analysis_file = os.path.join(self.report_dir, f"{report_name}_statistical_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis_file
    
    def _get_performance_level(self, score, thresholds):
        """Get performance level based on score and thresholds"""
        if score >= thresholds[2]:
            return "Excellent"
        elif score >= thresholds[1]:
            return "Good"
        elif score >= thresholds[0]:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _calculate_overall_grade(self, bleu4, meteor, cider):
        """Calculate overall performance grade"""
        # Weighted average of normalized scores
        bleu4_norm = min(bleu4 / 0.4, 1.0)  # Normalize to 0.4 as excellent
        meteor_norm = min(meteor / 0.35, 1.0)  # Normalize to 0.35 as excellent
        cider_norm = min(cider / 1.3, 1.0)  # Normalize to 1.3 as excellent
        
        overall_score = (bleu4_norm * 0.4 + meteor_norm * 0.3 + cider_norm * 0.3)
        
        if overall_score >= 0.85:
            return "A"
        elif overall_score >= 0.75:
            return "B"
        elif overall_score >= 0.65:
            return "C"
        elif overall_score >= 0.55:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, df):
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        if 'BLEU-4' in df.columns:
            bleu4_mean = df['BLEU-4'].mean()
            if bleu4_mean < 0.2:
                recommendations.append("BLEU-4 score is low. Consider fine-tuning on domain-specific data or using a larger model.")
        
        if 'caption_length' in df.columns:
            avg_length = df['caption_length'].mean()
            if avg_length > 20:
                recommendations.append("Captions are quite long. Consider reducing max_length parameter.")
            elif avg_length < 5:
                recommendations.append("Captions are very short. Consider increasing max_length parameter.")
        
        if 'vocabulary_overlap' in df.columns:
            vocab_overlap = df['vocabulary_overlap'].mean()
            if vocab_overlap < 0.3:
                recommendations.append("Low vocabulary overlap with references. Consider vocabulary expansion or domain adaptation.")
        
        if 'METEOR' in df.columns and 'BLEU-4' in df.columns:
            meteor_mean = df['METEOR'].mean()
            bleu4_mean = df['BLEU-4'].mean()
            if meteor_mean / bleu4_mean > 1.2:
                recommendations.append("METEOR score is relatively high compared to BLEU. Model shows good semantic understanding.")
        
        return recommendations
    
    def _generate_html_report(self, report_data, report_name):
        """Generate comprehensive HTML report"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Captioning Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .metric-card { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .excellent { color: #28a745; }
                .good { color: #17a2b8; }
                .fair { color: #ffc107; }
                .poor { color: #dc3545; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Image Captioning Evaluation Report</h1>
                <p><strong>Model:</strong> {{ metadata.model_name }}</p>
                <p><strong>Dataset:</strong> {{ metadata.dataset_name }}</p>
                <p><strong>Generated:</strong> {{ metadata.timestamp }}</p>
                <p><strong>Total Samples:</strong> {{ metadata.total_samples }}</p>
            </div>
            
            <h2>Performance Summary</h2>
            <div class="metric-card">
                <h3>Main Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Score</th><th>Performance Level</th></tr>
                    {% for metric, stats in metrics.items() %}
                    {% if metric in ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr'] %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ "%.4f"|format(stats.mean) }}</td>
                        <td>{{ get_performance_class(metric, stats.mean) }}</td>
                    </tr>
                    {% endif %}
                    {% endfor %}
                </table>
            </div>
            
            <h2>Detailed Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Median</th>
                </tr>
                {% for metric, stats in metrics.items() %}
                <tr>
                    <td>{{ metric }}</td>
                    <td>{{ "%.4f"|format(stats.mean) }}</td>
                    <td>{{ "%.4f"|format(stats.std) }}</td>
                    <td>{{ "%.4f"|format(stats.min) }}</td>
                    <td>{{ "%.4f"|format(stats.max) }}</td>
                    <td>{{ "%.4f"|format(stats.median) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Generated Files</h2>
            <ul>
                {% for file_type, file_path in files.items() %}
                <li><strong>{{ file_type.title() }}:</strong> {{ file_path }}</li>
                {% endfor %}
            </ul>
        </body>
        </html>
        """
        
        template = Template(html_template)
        
        def get_performance_class(metric, score):
            if metric == 'BLEU-4':
                if score >= 0.35: return '<span class="excellent">Excellent</span>'
                elif score >= 0.25: return '<span class="good">Good</span>'
                elif score >= 0.15: return '<span class="fair">Fair</span>'
                else: return '<span class="poor">Poor</span>'
            # Add other metrics...
            return '<span class="good">Good</span>'
        
        html_content = template.render(
            metadata=report_data['metadata'],
            metrics=report_data['metrics'],
            files=report_data['files'],
            get_performance_class=get_performance_class
        )
        
        html_file = os.path.join(self.report_dir, f"{report_name}_report.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        return html_file
    
    def _generate_pdf_summary(self, avg_metrics, all_metrics, report_name):
        """Generate PDF summary report"""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            pdf_file = os.path.join(self.report_dir, f"{report_name}_summary.pdf")
            
            with PdfPages(pdf_file) as pdf:
                # Title page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.text(0.5, 0.8, 'Image Captioning Evaluation Report', 
                       ha='center', va='center', fontsize=24, weight='bold')
                ax.text(0.5, 0.7, f'Report: {report_name}', 
                       ha='center', va='center', fontsize=16)
                ax.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Metrics summary
                df = pd.DataFrame(all_metrics)
                self._generate_matplotlib_plots(df, report_name)
                
                # Include the generated plot
                plot_file = os.path.join(self.report_dir, f"{report_name}_summary_plot.png")
                if os.path.exists(plot_file):
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    img = plt.imread(plot_file)
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
            
            return pdf_file
            
        except ImportError:
            print("Warning: matplotlib not available for PDF generation")
            return None