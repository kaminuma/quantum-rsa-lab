#!/usr/bin/env python3
"""Generate Support Mass vs 2Q Gate Count figure for arXiv paper."""

import matplotlib.pyplot as plt

# Data
data = {
    'a=6 (r=2)': {'2q_gates': 2, 'support_mass': 96.6, 'std': 1.2, 'qubits': 3},
    'a=8 (r=4)': {'2q_gates': 8, 'support_mass': 47.6, 'std': 2.2, 'qubits': 5},
}

# Extract values
labels = list(data.keys())
x = [data[k]['2q_gates'] for k in labels]
y = [data[k]['support_mass'] for k in labels]
yerr = [data[k]['std'] for k in labels]

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot with error bars
colors = ['#2ecc71', '#e74c3c']  # green for a=6, red for a=8
for i, (xi, yi, yerri, label) in enumerate(zip(x, y, yerr, labels)):
    ax.errorbar(xi, yi, yerr=yerri, fmt='o', markersize=10, capsize=5,
                color=colors[i], label=label, linewidth=2)

# Add horizontal line at 50% (uniform distribution baseline)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Uniform baseline (50%)')

# Labels and title
ax.set_xlabel('Logical 2Q Gate Count', fontsize=12)
ax.set_ylabel('Support Mass $S$ [%]', fontsize=12)
ax.set_title('Support Mass vs Circuit Complexity\n($N=35$, Rigetti Ankaa-3)', fontsize=12)

# Axis settings
ax.set_xlim(0, 10)
ax.set_ylim(0, 105)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_yticks([0, 25, 50, 75, 100])

# Grid
ax.grid(True, alpha=0.3)

# Legend
ax.legend(loc='upper right', fontsize=10)

# Annotations
ax.annotate('96.6%', (2, 96.6), textcoords="offset points", xytext=(15, -5),
            fontsize=10, fontweight='bold')
ax.annotate('47.6%', (8, 47.6), textcoords="offset points", xytext=(15, 5),
            fontsize=10, fontweight='bold')

# Add arrow showing the drop
ax.annotate('', xy=(8, 50), xytext=(2, 94),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
ax.text(5, 75, '−49pp', fontsize=10, color='gray', ha='center')

plt.tight_layout()

# Save
output_path = 'docs/figures/support_mass_vs_complexity.pdf'
import os
os.makedirs('docs/figures', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
print(f"Saved: {output_path.replace('.pdf', '.png')}")

plt.show()
