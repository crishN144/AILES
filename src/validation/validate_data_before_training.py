#!/usr/bin/env python3
"""
AILES Legal AI - Report Visualization Generation
Generates professional charts for the Mistral XML testing report
"""

import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
import numpy as np

# Set aesthetic style for all plots
plt.style.use('default')
sns.set_palette("viridis")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_workflow_diagram():
    """Generate professional workflow flowchart using Graphviz"""
    
    dot = Digraph(comment='AILES XML Processing Workflow')
    
    # Configure graph aesthetics
    dot.attr(rankdir='LR')  # Left to right layout
    dot.attr('node', 
             shape='box',
             style='rounded,filled',
             fontname='Arial',
             fontsize='12',
             margin='0.3')
    dot.attr('edge',
             fontname='Arial',
             fontsize='10',
             color='#2E4057',
             penwidth='2')
    
    # Define nodes with color scheme
    nodes = [
        ('A', 'XML Input\n(4,611 files)', '#E8F4FD'),
        ('B', 'Preprocess\n(Truncate, Clean)', '#B8E6B8'), 
        ('C', 'Mistral Generation\n(3 pair types)', '#FFD93D'),
        ('D', 'Validate\n(Length, Quality)', '#FFB4B4'),
        ('E', 'JSONL Output\n(Training pairs)', '#DDA0DD')
    ]
    
    for node_id, label, color in nodes:
        dot.node(node_id, label, fillcolor=color)
    
    # Add edges with labels
    edges = [
        ('A', 'B', '25k char limit'),
        ('B', 'C', 'Direct XML feed'),
        ('C', 'D', 'Quality check'),
        ('D', 'E', '66.7% success')
    ]
    
    for start, end, label in edges:
        dot.edge(start, end, label=label)
    
    # Render the diagram
    dot.render('/users/bgxp240/ailes_legal_ai/figures/testing_workflow_flowchart', 
               format='png', cleanup=True)
    
    print("✅ Workflow diagram generated: testing_workflow_flowchart.png")
    return dot

def create_success_rate_chart():
    """Generate aesthetic success rate bar chart"""
    
    # Data preparation
    pair_types = ['Fact_Extraction', 'Legal_Analysis', 'Structured_Extraction']
    success_rates = [100, 100, 0]
    
    # Color mapping for different outcomes
    colors = ['#2E8B57', '#4682B4', '#DC143C']  # Green, Blue, Red
    
    # Create figure with larger size for clarity
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart for better readability
    bars = ax.barh(pair_types, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Styling improvements
    ax.set_xlabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pair Type', fontsize=14, fontweight='bold')
    ax.set_title('Mistral-Nemo Success Rates by Pair Type\nAILES Legal AI Testing Results', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, success_rates)):
        width = bar.get_width()
        label_x = width + 1 if width > 0 else 5
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{value}%', ha='left' if width > 0 else 'left', 
                va='center', fontweight='bold', fontsize=12)
    
    # Add status indicators
    status_labels = ['✅ Success', '✅ Success', '❌ Failed']
    for i, (bar, status) in enumerate(zip(bars, status_labels)):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
                status, ha='left', va='center', fontsize=11, 
                fontweight='bold', style='italic')
    
    # Grid and formatting
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 120)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add subtitle with key metrics
    fig.suptitle('Overall Success Rate: 66.7% (10/15 pairs generated)', 
                 fontsize=12, style='italic', y=0.02)
    
    plt.tight_layout()
    plt.savefig('/users/bgxp240/ailes_legal_ai/figures/success_rate_bar.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Success rate chart generated: success_rate_bar.png")

def create_case_types_pie_chart():
    """Generate case types distribution pie chart"""
    
    case_types = ['Child Arrangements', 'Care Proceedings', 'Financial Remedies', 
                  'Adoption', 'International Abduction', 'Other']
    counts = [1195, 1184, 627, 551, 461, 1593]
    percentages = [count/sum(counts)*100 for count in counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate pie chart with modern colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(case_types)))
    
    wedges, texts, autotexts = ax.pie(counts, labels=case_types, autopct='%1.1f%%',
                                      startangle=90, colors=colors,
                                      textprops={'fontsize': 10},
                                      wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    
    # Enhance text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title('Case Type Distribution in XML Corpus\n4,611 UK Family Law Judgments (2003-2024)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend with counts
    legend_labels = [f'{case_type}\n({count:,} cases)' for case_type, count in zip(case_types, counts)]
    ax.legend(wedges, legend_labels, title="Case Types", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/users/bgxp240/ailes_legal_ai/figures/case_types_pie.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Case types pie chart generated: case_types_pie.png")

def create_scaling_projections():
    """Generate scaling projections bar chart"""
    
    scenarios = ['Baseline\n(66.7%)', 'Optimized\n(75%)', 'Improved\n(85%)']
    pairs = [9222, 10378, 11758]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(scenarios, pairs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, pairs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 100,
                f'{value:,}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    ax.set_title('Projected Training Pairs from Full Corpus\n4,611 XML Files → Training Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Valid Training Pairs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Improvement Scenario', fontsize=14, fontweight='bold')
    
    # Formatting
    ax.set_ylim(0, 13000)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add target line
    ax.axhline(y=10000, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(1, 10200, 'Target: 10,000+ pairs', ha='center', 
            fontweight='bold', color='red', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('/users/bgxp240/ailes_legal_ai/figures/estimated_pairs_bar.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Scaling projections chart generated: estimated_pairs_bar.png")

def main():
    """Generate all visualization assets for the AILES report"""
    
    print("🎨 Generating AILES Legal AI Report Visualizations...")
    print("=" * 50)
    
    try:
        # Create output directory if it doesn't exist
        import os
        os.makedirs('/users/bgxp240/ailes_legal_ai/figures', exist_ok=True)
        
        # Generate all visualizations
        create_workflow_diagram()
        create_success_rate_chart()
        create_case_types_pie_chart() 
        create_scaling_projections()
        
        print("=" * 50)
        print("✅ All visualizations generated successfully!")
        print("📁 Files saved to: /users/bgxp240/ailes_legal_ai/figures/")
        print("\nGenerated files:")
        print("- testing_workflow_flowchart.png")
        print("- success_rate_bar.png") 
        print("- case_types_pie.png")
        print("- estimated_pairs_bar.png")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        print("Please ensure you have the required dependencies:")
        print("pip install matplotlib seaborn graphviz")

if __name__ == "__main__":
    main()