"""
Plot LOYO (Leave-One-Year-Out) cross-validation results with train set sizes annotated.
Visualizes predictions, errors, skill scores, and train set progression.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the results
script_dir = Path(__file__).parent
data_path = script_dir / "loyo_results.csv"
df = pd.read_csv(data_path)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Actual vs Predicted with error bars and train size labels
ax1 = plt.subplot(2, 3, 1)
x_pos = np.arange(len(df))
ax1.scatter(x_pos, df['Target'], label='Actual', s=100, alpha=0.7, color='blue', marker='o')
ax1.scatter(x_pos, df['Prediction'], label='Predicted', s=100, alpha=0.7, color='red', marker='x')

# Add error bars
errors = np.abs(df['Error'])
ax1.errorbar(x_pos, df['Prediction'], yerr=errors, fmt='none', ecolor='red', alpha=0.3, capsize=3)

# Annotate with train sizes
for i, (idx, row) in enumerate(df.iterrows()):
    ax1.text(i, row['Target'] + 0.5, f"n={int(row['Train_Size_Used'])}", 
             fontsize=8, ha='center', color='blue', fontweight='bold')
    ax1.text(i, row['Prediction'] - 0.7, f"n={int(row['Train_Size_Used'])}", 
             fontsize=8, ha='center', color='red', fontweight='bold')

ax1.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
ax1.set_title('Actual vs Predicted Values\n(with Train Set Sizes)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['Year'])

# 2. Absolute Error
ax2 = plt.subplot(2, 3, 2)
abs_errors = np.abs(df['Error'])
colors = ['green' if e < 2 else 'orange' if e < 4 else 'red' for e in abs_errors]
bars = ax2.bar(x_pos, abs_errors, color=colors, alpha=0.7, edgecolor='black')

# Add train size labels on bars
for i, (idx, row) in enumerate(df.iterrows()):
    height = abs_errors.iloc[i]
    ax2.text(i, height + 0.1, f"n={int(row['Train_Size_Used'])}", 
             fontsize=7, ha='center', fontweight='bold')

ax2.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Error by Year\n(with Train Set Sizes)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df['Year'])

# 3. Skill Score
ax3 = plt.subplot(2, 3, 3)
colors_skill = ['green' if s > 0.8 else 'orange' if s > 0 else 'red' for s in df['Skill_Score']]
bars = ax3.bar(x_pos, df['Skill_Score'], color=colors_skill, alpha=0.7, edgecolor='black')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.axhline(y=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5)

# Add train size labels on bars
for i, (idx, row) in enumerate(df.iterrows()):
    height = df['Skill_Score'].iloc[i]
    y_pos = height + (0.05 if height > 0 else -0.1)
    ax3.text(i, y_pos, f"n={int(row['Train_Size_Used'])}", 
             fontsize=7, ha='center', fontweight='bold')

ax3.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Skill Score', fontsize=11, fontweight='bold')
ax3.set_title('Skill Score by Year\n(with Train Set Sizes)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(df['Year'])
ax3.set_ylim([min(df['Skill_Score']) - 0.3, max(df['Skill_Score']) + 0.3])

# 4. Train Set Size Progression
ax4 = plt.subplot(2, 3, 4)
line = ax4.plot(x_pos, df['Train_Size_Used'], 'o-', linewidth=2, markersize=8, color='purple')
for i, (idx, row) in enumerate(df.iterrows()):
    ax4.text(i, row['Train_Size_Used'] + 0.3, f"{int(row['Train_Size_Used'])}", 
             fontsize=8, ha='center', fontweight='bold', color='purple')

ax4.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Train Set Size (n)', fontsize=11, fontweight='bold')
ax4.set_title('Training Set Size Growth', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(df['Year'])

# 5. Squared Error vs Baseline
ax5 = plt.subplot(2, 3, 5)
width = 0.35
ax5.bar(x_pos - width/2, df['Squared_Error'], width, label='Squared Error', alpha=0.8, edgecolor='black')
ax5.bar(x_pos + width/2, df['Baseline_Squared_Error'], width, label='Baseline Error', alpha=0.8, edgecolor='black')

# Add train sizes to center of bars
for i, (idx, row) in enumerate(df.iterrows()):
    max_height = max(df['Squared_Error'].iloc[i], df['Baseline_Squared_Error'].iloc[i])
    ax5.text(i, max_height + 1, f"n={int(row['Train_Size_Used'])}", 
             fontsize=7, ha='center', fontweight='bold')

ax5.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Squared Error', fontsize=11, fontweight='bold')
ax5.set_title('Model vs Baseline Error Comparison\n(with Train Set Sizes)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(df['Year'])

# 6. Summary Statistics Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Calculate summary statistics
summary_stats = [
    ['Metric', 'Value'],
    ['---', '---'],
    [f'Number of Test Years', f"{len(df)}"],
    [f'Train Size Range', f"{int(df['Train_Size_Used'].min())}-{int(df['Train_Size_Used'].max())}"],
    [f'Mean Absolute Error', f"{df['Error'].abs().mean():.3f}"],
    [f'Mean Squared Error', f"{df['Squared_Error'].mean():.3f}"],
    [f'Mean Skill Score', f"{df['Skill_Score'].mean():.3f}"],
    [f'% Good Predictions (Skill > 0.8)', f"{(df['Skill_Score'] > 0.8).sum() / len(df) * 100:.1f}%"],
    [f'% Positive Skill Scores', f"{(df['Skill_Score'] > 0).sum() / len(df) * 100:.1f}%"],
]

table = ax6.table(cellText=summary_stats, cellLoc='left', loc='center', 
                   colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(len(summary_stats[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style separator row
for i in range(len(summary_stats[0])):
    table[(1, i)].set_facecolor('#e0e0e0')

ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(script_dir / 'loyo_results_annotated.png', dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {script_dir / 'loyo_results_annotated.png'}")

# Also save individual plots for each metric
fig2, ((ax_err, ax_skill), (ax_train, ax_baseline)) = plt.subplots(2, 2, figsize=(14, 10))

# Error progression
ax_err.plot(x_pos, df['Error'].abs(), 'o-', linewidth=2, markersize=8, color='red')
for i, (idx, row) in enumerate(df.iterrows()):
    ax_err.text(i, abs(row['Error']) + 0.2, f"n={int(row['Train_Size_Used'])}", 
                fontsize=8, ha='center', fontweight='bold', rotation=0)
ax_err.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax_err.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
ax_err.set_title('Error Progression with Train Set Sizes', fontsize=12, fontweight='bold')
ax_err.grid(True, alpha=0.3)
ax_err.set_xticks(x_pos)
ax_err.set_xticklabels(df['Year'])

# Skill score progression
ax_skill.plot(x_pos, df['Skill_Score'], 'o-', linewidth=2, markersize=8, color='green')
ax_skill.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (>0.8)')
ax_skill.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline')
for i, (idx, row) in enumerate(df.iterrows()):
    ax_skill.text(i, row['Skill_Score'] + 0.08, f"n={int(row['Train_Size_Used'])}", 
                  fontsize=8, ha='center', fontweight='bold')
ax_skill.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax_skill.set_ylabel('Skill Score', fontsize=11, fontweight='bold')
ax_skill.set_title('Skill Score Progression with Train Set Sizes', fontsize=12, fontweight='bold')
ax_skill.legend(fontsize=10)
ax_skill.grid(True, alpha=0.3)
ax_skill.set_xticks(x_pos)
ax_skill.set_xticklabels(df['Year'])

# Train size growth
ax_train.plot(x_pos, df['Train_Size_Used'], 'o-', linewidth=2, markersize=10, color='purple')
for i, (idx, row) in enumerate(df.iterrows()):
    ax_train.text(i, row['Train_Size_Used'] + 0.3, f"{int(row['Train_Size_Used'])}", 
                  fontsize=9, ha='center', fontweight='bold', color='purple')
ax_train.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax_train.set_ylabel('Train Set Size (n)', fontsize=11, fontweight='bold')
ax_train.set_title('Training Set Size Growth', fontsize=12, fontweight='bold')
ax_train.grid(True, alpha=0.3)
ax_train.set_xticks(x_pos)
ax_train.set_xticklabels(df['Year'])

# Improvement over baseline
improvement = 1 - (df['Squared_Error'] / df['Baseline_Squared_Error'])
colors_imp = ['green' if imp > 0.5 else 'orange' if imp > 0 else 'red' for imp in improvement]
ax_baseline.bar(x_pos, improvement * 100, color=colors_imp, alpha=0.7, edgecolor='black')
for i, (idx, row) in enumerate(df.iterrows()):
    imp_val = improvement.iloc[i] * 100
    y_pos = imp_val + (2 if imp_val > 0 else -5)
    ax_baseline.text(i, y_pos, f"n={int(row['Train_Size_Used'])}", 
                     fontsize=8, ha='center', fontweight='bold')
ax_baseline.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax_baseline.set_xlabel('Year (Test Index)', fontsize=11, fontweight='bold')
ax_baseline.set_ylabel('Improvement over Baseline (%)', fontsize=11, fontweight='bold')
ax_baseline.set_title('Model Performance vs Baseline\n(with Train Set Sizes)', fontsize=12, fontweight='bold')
ax_baseline.grid(True, alpha=0.3, axis='y')
ax_baseline.set_xticks(x_pos)
ax_baseline.set_xticklabels(df['Year'])

plt.tight_layout()
plt.savefig(script_dir / 'loyo_results_detailed.png', dpi=300, bbox_inches='tight')
print(f"✓ Detailed plot saved to: {script_dir / 'loyo_results_detailed.png'}")

# Print summary
print("\n" + "="*60)
print("LOYO CROSS-VALIDATION RESULTS SUMMARY")
print("="*60)
print(df.to_string())
print("\n" + "-"*60)
print(f"Mean Absolute Error: {df['Error'].abs().mean():.4f}")
print(f"Mean Squared Error: {df['Squared_Error'].mean():.4f}")
print(f"Mean Skill Score: {df['Skill_Score'].mean():.4f}")
print(f"High Skill Predictions (>0.8): {(df['Skill_Score'] > 0.8).sum()}/{len(df)}")
print(f"Positive Skill Scores: {(df['Skill_Score'] > 0).sum()}/{len(df)}")
print("-"*60)

plt.show()
