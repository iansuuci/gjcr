#!/usr/bin/env python3
"""
Visualization script for Grok experiment results.
Creates comprehensive visualizations of the experiment outcomes.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = "grok_experiment_results.json"
with open(results_file, 'r') as f:
    data = json.load(f)

summary = data['summary']
detailed_results = data['detailed_results']

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ==================== Plot 1: Success Rates ====================
ax1 = fig.add_subplot(gs[0, 0])
conditions = ['only_code', 'only_comments', 'both', 'nothing']
condition_labels = ['Only Code', 'Only Comments', 'Both', 'Nothing']
success_rates = [summary['success_rates'][c] * 100 for c in conditions]

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax1.bar(condition_labels, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Success Rates by Condition', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Add value labels on bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add total problems annotation
ax1.text(0.02, 0.98, f'Total Problems: {summary["total_problems"]}',
         transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ==================== Plot 2: Effects ====================
ax2 = fig.add_subplot(gs[0, 1])
effects = summary['effects']
effect_names = ['Δ Code', 'Δ Comments', 'Δ CoT']
effect_values = [effects['delta_code'] * 100, effects['delta_comments'] * 100, effects['delta_cot'] * 100]

effect_colors = ['#06A77D' if v > 0 else '#D00000' for v in effect_values]
bars2 = ax2.bar(effect_names, effect_values, color=effect_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Effect (%)', fontsize=12, fontweight='bold')
ax2.set_title('Effects: Improvement Over Baseline', fontsize=14, fontweight='bold', pad=15)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Add value labels
for bar, val in zip(bars2, effect_values):
    height = bar.get_height()
    y_pos = height + (2 if height > 0 else -4)
    ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{val:+.1f}%',
             ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')

# Add explanation text
ax2.text(0.5, 0.02, 'Δ Code = X - X\'\'\'\nΔ Comments = X\' - X\'\'\'\nΔ CoT = X\'\'\' - X\'\'',
         transform=ax2.transAxes, fontsize=9,
         ha='center', va='bottom', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# ==================== Plot 3: Adherence Scores ====================
ax3 = fig.add_subplot(gs[1, 0])
adherence = summary['average_adherence']
adherence_values = [adherence[c] * 100 for c in conditions]

bars3 = ax3.bar(condition_labels, adherence_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Adherence Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('Average Adherence Scores', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylim(0, 105)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

# Add value labels
for bar, score in zip(bars3, adherence_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{score:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# ==================== Plot 4: Per-Problem Performance ====================
ax4 = fig.add_subplot(gs[1, 1])
num_problems = len(detailed_results)
problem_indices = list(range(1, num_problems + 1))

# Create matrix: rows = problems, cols = conditions
performance_matrix = np.zeros((num_problems, len(conditions)))
for i, result in enumerate(detailed_results):
    for j, cond in enumerate(conditions):
        performance_matrix[i, j] = 1 if result[f'{cond}_correct'] else 0

im = ax4.imshow(performance_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
ax4.set_xlabel('Condition', fontsize=12, fontweight='bold')
ax4.set_ylabel('Problem Index', fontsize=12, fontweight='bold')
ax4.set_title('Per-Problem Performance Heatmap', fontsize=14, fontweight='bold', pad=15)
ax4.set_xticks(range(len(conditions)))
ax4.set_xticklabels(condition_labels, rotation=45, ha='right')
ax4.set_yticks(range(0, num_problems, max(1, num_problems // 10)))
ax4.set_yticklabels(range(1, num_problems + 1, max(1, num_problems // 10)))

# Add colorbar
cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
cbar.set_label('Correct (1) / Incorrect (0)', fontsize=10)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Incorrect', 'Correct'])

# ==================== Plot 5: Cumulative Success Over Problems ====================
ax5 = fig.add_subplot(gs[2, 0])
cumulative_correct = {cond: [] for cond in conditions}
running_total = {cond: 0 for cond in conditions}

for result in detailed_results:
    for cond in conditions:
        if result[f'{cond}_correct']:
            running_total[cond] += 1
        cumulative_correct[cond].append(running_total[cond] / (result['problem_index']) * 100)

for i, cond in enumerate(conditions):
    ax5.plot(problem_indices, cumulative_correct[cond], 
             label=condition_labels[i], color=colors[i], linewidth=2.5, marker='o', markersize=3)

ax5.set_xlabel('Problem Number', fontsize=12, fontweight='bold')
ax5.set_ylabel('Cumulative Success Rate (%)', fontsize=12, fontweight='bold')
ax5.set_title('Cumulative Success Rate Over Problems', fontsize=14, fontweight='bold', pad=15)
ax5.legend(loc='best', fontsize=10)
ax5.grid(alpha=0.3, linestyle='--')
ax5.set_axisbelow(True)

# ==================== Plot 6: Condition Comparison Summary ====================
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

# Create summary table
summary_data = [
    ['Condition', 'Success Rate', 'Adherence', 'Effect'],
    ['Only Code', f"{summary['success_rates']['only_code']*100:.1f}%", 
     f"{summary['average_adherence']['only_code']*100:.1f}%", 
     f"{summary['effects']['delta_code']*100:+.1f}%"],
    ['Only Comments', f"{summary['success_rates']['only_comments']*100:.1f}%", 
     f"{summary['average_adherence']['only_comments']*100:.1f}%", 
     f"{summary['effects']['delta_comments']*100:+.1f}%"],
    ['Both', f"{summary['success_rates']['both']*100:.1f}%", 
     f"{summary['average_adherence']['both']*100:.1f}%", 
     f"{summary['effects']['delta_cot']*100:+.1f}%"],
    ['Nothing (Baseline)', f"{summary['success_rates']['nothing']*100:.1f}%", 
     f"{summary['average_adherence']['nothing']*100:.1f}%", 
     'N/A'],
]

table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0],
                  cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(len(summary_data[0])):
    table[(0, i)].set_facecolor('#4A90E2')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows with alternating colors
for i in range(1, len(summary_data)):
    for j in range(len(summary_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')
        else:
            table[(i, j)].set_facecolor('white')

ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

# Add overall title
fig.suptitle('Grok Experiment Results: Code vs Comments vs Both vs Nothing', 
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
output_file = 'grok_experiment_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Visualization saved to: {output_file}")

# Also create a simpler comparison chart
fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(14, 6))

# Side-by-side comparison
x = np.arange(len(condition_labels))
width = 0.35

bars1 = ax7.bar(x - width/2, success_rates, width, label='Success Rate', 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax7.bar(x + width/2, adherence_values, width, label='Adherence', 
                color=[c.replace('0.8', '0.6') for c in colors], alpha=0.8, 
                edgecolor='black', linewidth=1.5, hatch='///')

ax7.set_xlabel('Condition', fontsize=12, fontweight='bold')
ax7.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax7.set_title('Success Rate vs Adherence Score', fontsize=14, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(condition_labels, rotation=45, ha='right')
ax7.legend(fontsize=11)
ax7.grid(axis='y', alpha=0.3, linestyle='--')
ax7.set_axisbelow(True)
ax7.set_ylim(0, 105)

# Effects bar chart
bars3 = ax8.bar(effect_names, effect_values, color=effect_colors, alpha=0.8, 
                edgecolor='black', linewidth=1.5)
ax8.set_ylabel('Effect (%)', fontsize=12, fontweight='bold')
ax8.set_title('Effects: Change from Baseline', fontsize=14, fontweight='bold')
ax8.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax8.grid(axis='y', alpha=0.3, linestyle='--')
ax8.set_axisbelow(True)

for bar, val in zip(bars3, effect_values):
    height = bar.get_height()
    y_pos = height + (2 if height > 0 else -4)
    ax8.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{val:+.1f}%',
             ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')

plt.tight_layout()
output_file2 = 'grok_experiment_comparison.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Comparison chart saved to: {output_file2}")

plt.show()

