import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# THEME CONFIGURATION
# ==========================================
# Set this to True if you are using a Dark Jupyter Theme (e.g., JupyterLab Dark)
# Set this to False if you are using the default Light/White Theme
USE_DARK_THEME = False 

if USE_DARK_THEME:
    # --- DARK THEME PALETTE ---
    BACKGROUND_COLOUR = '#1E1E1E'       # Dark Grey Background
    FONT_COLOUR = '#F1F1F1'             # Almost White Text
    GRID_COLOUR = '#DDDDDD'             # Light Grey Gridlines
    
    COLOR_TUMOUR = '#E63946'            # Vibrant Red
    COLOR_LIVER = '#457B9D'             # Muted Blue
    COLOR_TIME = '#E9C46A'              # Sand/Yellow
    COLOR_OUTLIER = '#FF0000'           # Bright Red for failures
    MEAN_COLOUR = '#FFD700'             # Gold/Yellow for Mean marker

else:
    # --- LIGHT THEME PALETTE ---
    BACKGROUND_COLOUR = '#FFFFFF'       # Pure White Background
    FONT_COLOUR = '#2D3748'             # Deep Charcoal (Professional & Readable)
    GRID_COLOUR = '#E2E8F0'             # Very Subtle Blue-Grey Gridlines
    
    COLOR_TUMOUR = '#C53030'            # Deeper Red (Better contrast on white)
    COLOR_LIVER = '#2B6CB0'             # Stronger Blue (Better contrast on white)
    COLOR_TIME = '#B7791F'              # Darker Amber (Visible on white)
    COLOR_OUTLIER = '#E53E3E'           # Standard Red for outliers
    MEAN_COLOUR = '#D69E2E'             # Mustard/Gold (Visible on white)


@dataclass
class ModelTrainingResult:
    model_name: str
    tumour_dice: float
    liver_dice: float
    mean_dice: float
    time: str
    epochs: int

def parse_time_to_hours(time_str: str) -> float:
    """Parses 'Xh Ym Zs' format into decimal hours."""
    match = re.match(r"(\d+)h\s*(\d+)m\s*(\d+)s", time_str)
    if match:
        h, m, s = map(int, match.groups())
        return h + m / 60.0 + s / 3600.0
    return 0.0

def autolabel(rects, ax):
    """Annotates bars with their exact numerical values."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=FONT_COLOUR)

def _drop_seconds(time_str: str) -> str:
    """Converts 'Xh Ym Zs' to 'Xh Ym' for cleaner display."""
    match = re.match(r"(\d+h\s*\d+m)\s*\d+s", time_str)
    if match:
        return match.group(1)
    return time_str


def plot_results(model_list):
    models = [r.model_name for r in model_list]
    tumour_dice = [r.tumour_dice for r in model_list]
    liver_dice = [r.liver_dice for r in model_list]

    times_hours = [parse_time_to_hours(r.time) for r in model_list]
    times_str = [_drop_seconds(r.time) for r in model_list]


    x = np.arange(len(models))
    width = 0.2

    fig2, (ax2_top, ax2_bot) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.2},
        layout='constrained'  # <--- Add this
    )

    # --- Top Plot: Dice Scores ---
    rects2 = ax2_top.bar(x - width, liver_dice, width, label='Liver Dice', color=COLOR_LIVER, edgecolor='white', linewidth=1.2)
    rects1 = ax2_top.bar(x, tumour_dice, width, label='Tumour Dice', color=COLOR_TUMOUR, edgecolor='white', linewidth=1.2)

    ax2_top.set_ylabel('Dice Score', fontsize=11, fontweight='bold', color=FONT_COLOUR)
    ax2_top.set_title('Model Performance vs. Training Time', fontsize=13, fontweight='bold', pad=10, color=FONT_COLOUR)

    ax2_top.set_xticks(x, labels=models, fontsize=11, fontweight='bold', color=FONT_COLOUR)

    ax2_top.set_ylim(0, 1.15)
    ax2_top.legend(frameon=False, fontsize=10, loc='upper left', labelcolor=FONT_COLOUR)

    style_ax(ax2_top)
    autolabel(rects2, ax2_top)
    autolabel(rects1, ax2_top)

    # --- Bottom Plot: Training Time ---
    ax2_bot.bar(x, times_hours, width=0.5, color=COLOR_TIME, edgecolor='white', linewidth=1.2)
    ax2_bot.set_xticks(x, labels=[]) 
    ax2_bot.set_ylabel('Time (hours)', fontsize=11, fontweight='bold', color=FONT_COLOUR)
    ax2_bot.set_xticklabels([]) # Hide to avoid duplicating model names
    ax2_bot.set_ylim(0, max(times_hours) * 1.4)


    style_ax(ax2_bot)

    y_max = ax2_bot.get_ylim()[1]
    offset = y_max * 0.05  # 5% of the total Y-axis height

    for i, v in enumerate(times_hours):
        ax2_bot.text(i, v + offset, f"{times_str[i]}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=FONT_COLOUR)

    return fig2

def style_ax(ax):
    """Removes top/right spines and adds subtle gridlines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#555555', labelcolor=FONT_COLOUR)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color=GRID_COLOUR)
    ax.set_axisbelow(True)

def resolve_overlaps(positions, min_gap):
    """
    Adjusts a list of Y positions so they are at least min_gap apart.
    Processes from bottom to top, pushing higher labels up if they collide.
    """
    sorted_indices = sorted(range(len(positions)), key=lambda k: positions[k])
    adjusted = list(positions)
    for i in range(1, len(sorted_indices)):
        curr_idx = sorted_indices[i]
        prev_idx = sorted_indices[i-1]
        if adjusted[curr_idx] - adjusted[prev_idx] < min_gap:
            adjusted[curr_idx] = adjusted[prev_idx] + min_gap
    return adjusted

def _draw_metric_boxplot(ax, data_list, colors, title, ylabel, y_lim=None):
    # 1. Draw the boxplot
    bp = ax.boxplot(data_list, patch_artist=True, widths=0.4,
                    showfliers=False,
                    boxprops=dict(linewidth=1.5, edgecolor='white'),
                    whiskerprops=dict(linewidth=1.5, color='white', linestyle='--'),
                    capprops=dict(linewidth=1.5, color='white'),
                    medianprops=dict(linewidth=2, color=FONT_COLOUR))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # 2. Determine final Y-limits to calculate accurate min_gap
    if y_lim is not None:
        final_y_min, final_y_max = y_lim
    else:
        all_data = np.concatenate([d for d in data_list if len(d) > 0])
        max_val = np.nanmax(all_data) if len(all_data) > 0 else 1.0
        final_y_min = 0
        final_y_max = max_val * 1.35 if max_val > 0 else 1.0

    y_range = final_y_max - final_y_min
    min_gap = y_range * 0.04  # 4% of the total Y-axis height

    # 3. Overlay strip plot and add statistical labels
    for i, data in enumerate(data_list, 1):
        if len(data) == 0:
            continue  # Skip empty data to avoid errors

        # Calculate bounds first so we know which points are outliers
        q1_val, q3_val = np.percentile(data, [25, 75])
        iqr = q3_val - q1_val
        upper_bound = q3_val + 1.5 * iqr
        lower_bound = q1_val - 1.5 * iqr

        # Identify outliers in BOTH tails
        is_outlier = (data > upper_bound) | (data < lower_bound)
        
        # Filter out the outliers from the main dataset for the grey jitter
        non_outlier_data = data[~is_outlier]

        # Jitter ONLY for non-outlier points (Grey)
        if len(non_outlier_data) > 0:
            x_jitter = np.random.normal(i, 0.06, size=len(non_outlier_data))
            ax.scatter(x_jitter, non_outlier_data, alpha=0.5, color='#333333', s=20, zorder=2, edgecolors='none')

        # Re-plot ALL outliers specifically in RED
        outlier_values = data[is_outlier]
        if len(outlier_values) > 0:
            x_outliers = np.random.normal(i, 0.06, size=len(outlier_values))
            ax.scatter(x_outliers, outlier_values, color=COLOR_OUTLIER, s=60, zorder=5, 
                        edgecolors='white', linewidth=1.5)

        # --- STATISTICAL LABELS ---
        med_val = np.median(data)
        mean_val = np.mean(data)
        
        # Distinct white diamond marker for the mean
        ax.scatter(i, mean_val, marker='D', color='white', edgecolors=FONT_COLOUR, s=40, zorder=6)

        # Resolve overlapping labels dynamically
        raw_ys = [q3_val, med_val, q1_val, mean_val]
        adjusted_ys = resolve_overlaps(raw_ys, min_gap)

        labels_info = [
            {'y': adjusted_ys[0], 'orig_y': q3_val, 'text': f'Q3: {q3_val:.3f}', 'color': FONT_COLOUR},
            {'y': adjusted_ys[1], 'orig_y': med_val, 'text': f'Med: {med_val:.3f}', 'color': FONT_COLOUR},
            {'y': adjusted_ys[2], 'orig_y': q1_val, 'text': f'Q1: {q1_val:.3f}', 'color': FONT_COLOUR},
            {'y': adjusted_ys[3], 'orig_y': mean_val, 'text': f'Mean: {mean_val:.3f}', 'color': MEAN_COLOUR},
        ]

        label_x = i + 0.35
        for item in labels_info:
            ax.text(label_x, item['y'], item['text'], va='center', ha='left', 
                    color=item['color'], fontsize=9, 
                    fontweight='bold' if item['color'] == MEAN_COLOUR else 'normal')
            
            # Draw a faint dotted line if the label was shifted to avoid overlap
            if abs(item['y'] - item['orig_y']) > 1e-5:
                ax.plot([i + 0.26, label_x - 0.02], [item['orig_y'], item['y']], 
                        color='#777777', linestyle=':', linewidth=1, alpha=0.8)

    # 4. Finalize axes
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10, color=FONT_COLOUR)
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color=FONT_COLOUR)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Liver', 'Tumour'], fontsize=11, fontweight='bold', color=FONT_COLOUR)

    # Expanded X-limits to ensure the text labels on the right are not clipped
    ax.set_xlim(0.5, 3.2) 
    ax.set_ylim(final_y_min, final_y_max)

    style_ax(ax)
# End draw metric boxplot

# ==========================================
# STATISTICAL BOXPLOTS (WITH SMART LABELS)
# ==========================================
def plot_test_boxplots(df: pd.DataFrame, model_name: str = "Model", attach_hd95: bool=False):
    """
    Generates boxplots with overlaid strip plots (jitter). 
    Outliers are highlighted in red. Labels for Q1, Median, Q3, and Mean 
    are added to the right of each box. Overlapping labels are automatically 
    shifted apart and connected with a dotted line.
    """
    # Drop NaNs safely (happens when a structure is absent)
    d_liv = df['dice_liver'].dropna().values
    d_tum = df['dice_tumour'].dropna().values
    h_liv = df['hd95_liver_mm'].dropna().values
    h_tum = df['hd95_tumour_mm'].dropna().values

    if attach_hd95:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
        _draw_metric_boxplot(ax2, [h_liv, h_tum], [COLOR_LIVER, COLOR_TUMOUR],
                            f'{model_name} Test Set: Boundary Distance (HD95)',
                            '95th Percentile Hausdorff Distance (mm)')
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))

    _draw_metric_boxplot(ax1, [d_liv, d_tum], [COLOR_LIVER, COLOR_TUMOUR],
                        f'{model_name} Test Set: Dice Score Distribution',
                        'Dice Similarity Coefficient', y_lim=[-0.05, 1.1])

    return fig
