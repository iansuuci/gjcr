import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Load the data from JSON files
files_info = [
    # Grok models
    ("results/grok_AIME_experiment_results.json", "Grok", "AIME"),
    ("results/grok_gsm_experiment_results.json", "Grok", "GSM8K"),
    ("results/grok_hmmt_experiment_results.json", "Grok", "HMMT"),
    # DeepSeek models
    ("results/deepseek_aime_experiment_results.json", "DeepSeek", "AIME"),
    ("results/deepseek_gsm_experiment_results.json", "DeepSeek", "GSM8K"),
    ("results/deepseek_hmmt_experiment_results.json", "DeepSeek", "HMMT"),
    # Gemini models
    ("results/gemini_aime_experiment_results.json", "Gemini", "AIME"),
    ("results/gemini_gsm_experiment_results.json", "Gemini", "GSM8K"),
    ("results/gemini_hmmt_experiment_results.json", "Gemini", "HMMT"),
    # OpenAI models
    ("results/openai_aime_experiment_results.json", "OpenAI", "AIME"),
    ("results/openai_gsm_experiment_results.json", "OpenAI", "GSM8K"),
    ("results/openai_hmmt_experiment_results.json", "OpenAI", "HMMT"),
]

# Collect data
data = []
for filepath, model, dataset in files_info:
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Handle different JSON formats
    if 'success_rates' in results['summary']:
        success_rates = results['summary']['success_rates']
    elif 'correct_counts' in results['summary']:
        # Convert correct_counts to success_rates
        total = results['summary']['total']
        success_rates = {k: v / total for k, v in results['summary']['correct_counts'].items()}
    else:
        continue
    
    for category, rate in success_rates.items():
        data.append({
            'model': model,
            'dataset': dataset,
            'category': category,
            'success_rate': rate,
            'label': f"{model}\n{dataset}"
        })

# Categories
categories = ['only_code', 'only_comments', 'both', 'nothing', 'cot']
category_labels = ['Only Code', 'Only Comments', 'Both', 'Nothing', 'CoT']

# Create grouped labels for x-axis
model_dataset_pairs = [
    ("Grok", "AIME"),
    ("Grok", "GSM8K"),
    ("Grok", "HMMT"),
    ("DeepSeek", "AIME"),
    ("DeepSeek", "GSM8K"),
    ("DeepSeek", "HMMT"),
    ("Gemini", "AIME"),
    ("Gemini", "GSM8K"),
    ("Gemini", "HMMT"),
    ("OpenAI", "AIME"),
    ("OpenAI", "GSM8K"),
    ("OpenAI", "HMMT"),
]

# Set up the figure with a dark, sophisticated theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(22, 9))

# Custom colors - vibrant palette against dark background
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Bar positioning
x = np.arange(len(model_dataset_pairs))
width = 0.14
multiplier = 0

# Plot bars for each category
for idx, category in enumerate(categories):
    rates = []
    for model, dataset in model_dataset_pairs:
        # Find the matching data point
        for d in data:
            if d['model'] == model and d['dataset'] == dataset and d['category'] == category:
                rates.append(d['success_rate'])
                break
    
    offset = width * multiplier
    bars = ax.bar(x + offset, rates, width, label=category_labels[idx], 
                  color=colors[idx], edgecolor='white', linewidth=0.5, alpha=0.9)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, color='white',
                    fontweight='bold', rotation=90)
    
    multiplier += 1

# Customize the chart
ax.set_xlabel('Model & Dataset', fontsize=14, fontweight='bold', color='white', labelpad=15)
ax.set_ylabel('Success Rate', fontsize=14, fontweight='bold', color='white', labelpad=15)
ax.set_title('Experiment Success Rates by Model, Dataset, and Category', 
             fontsize=18, fontweight='bold', color='white', pad=20)

# Set x-axis labels
x_labels = [f'{model}\n{dataset}' for model, dataset in model_dataset_pairs]
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=12, fontweight='medium')

# Set y-axis range
ax.set_ylim(0, 1.25)

# Add grid
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Customize legend
legend = ax.legend(loc='upper right', framealpha=0.9, fancybox=True, 
                   shadow=True, fontsize=10, title='Category', title_fontsize=12)
legend.get_frame().set_facecolor('#2C3E50')

# Add subtle background gradient effect
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#16213e')

# Adjust spines
for spine in ax.spines.values():
    spine.set_color('#4a4a6a')
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('results/success_rates_visualization.png', dpi=300, 
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print("Visualization saved to results/success_rates_visualization.png")
