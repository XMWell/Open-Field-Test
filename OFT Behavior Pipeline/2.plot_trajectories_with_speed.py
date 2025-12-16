"""
Trajectory Visualization with Speed Coding for Open Field Test

Description:
    This script generates trajectory plots from DeepLabCut CSV files.
    The trajectory lines are color-coded based on the animal's speed using a custom colormap.
    It also generates a separate legend/colorbar PDF using plotnine.

    - Filters low-confidence tracking points.
    - Smooths speed data using a rolling window.
    - Visualizes movement path with color gradients representing speed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import os
import warnings
from plotnine import *

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# --- CONFIGURATION PARAMETERS ---
# ==========================================

# 1. Input Files
# List the full paths to the CSV files you want to analyze.
CSV_FILE_PATHS = [
    # "path/to/your/file1.csv",
    # "path/to/your/file2.csv",
]

# 2. Output Settings
# Name of the folder to save results (created in the same directory as this script)
OUTPUT_FOLDER_NAME = "trajectory_results"

# 3. Analysis Parameters
BODYPART_TO_PLOT = 'HEAD'      # Body part to track
LIKELIHOOD_THRESHOLD = 0.60    # Confidence threshold
SMOOTHING_WINDOW_SIZE = 15     # Window size for rolling mean speed smoothing

# 4. Visualization Settings
VIDEO_WIDTH = 500              # Arena width in pixels
VIDEO_HEIGHT = 500             # Arena height in pixels
SPEED_MAX = 22                 # Maximum speed for color normalization (cm/s or pixel/frame unit)
LINE_WIDTH = 1                 # Thickness of the trajectory line

# 5. Color Scheme
# Custom hex colors for the gradient (Blue -> Red)
CUSTOM_COLORS_OPTIMIZED = [
    '#0F467F', '#317CB6', '#6EAED1',
    '#DD6F58', '#B52330', '#6F011E'
]

# Create colormap object
COLOR_MAP = mcolors.LinearSegmentedColormap.from_list("custom_speed_cmap_optimized", CUSTOM_COLORS_OPTIMIZED)

# ==========================================
# --- FUNCTION DEFINITIONS ---
# ==========================================

def calculate_speed(x, y):
    """
    Calculate instantaneous speed (Euclidean distance between consecutive frames).
    """
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    speed = np.sqrt(dx**2 + dy**2)
    return speed


def create_standard_colorbar(output_dir):
    """
    Generate a standalone colorbar/legend using plotnine.
    The scale represents 0% to 100% of the defined SPEED_MAX.
    """
    print("Generating standard colorbar...")
    
    # 1. Create dummy data for the gradient
    dummy_data = pd.DataFrame({
        'x': np.repeat(0, 100),
        'y': np.linspace(0, 100, 100),
        'speed_percent': np.linspace(0, 100, 100)
    })
    
    # 2. Define breaks for the legend
    custom_breaks = [0, 25, 50, 75, 100]
    
    # 3. Create the plot
    p = (ggplot(dummy_data, aes(x='x', y='y', color='speed_percent'))
         + geom_point(size=0.1, alpha=0)  # Invisible points to generate legend
         + scale_color_gradientn(
             colors=CUSTOM_COLORS_OPTIMIZED,
             name="",
             limits=[0, 100],
             breaks=custom_breaks
         )
         + theme_void()
         + theme(
             legend_position='right',
             legend_title=element_blank(),
             legend_text=element_text(size=10, color='black', margin={'l': 5}), 
             legend_ticks=element_line(color='black'),
             legend_key_width=15,
             legend_key_height=100,
             figure_size=(0.5, 2.5), 
             plot_margin=0,
             panel_spacing=0
         ))
    
    # 4. Save
    output_path = os.path.join(output_dir, "colorbar_standard_percent.pdf")
    try:
        p.save(output_path, dpi=300, verbose=False, bbox_inches='tight')
        print(f"  -> Colorbar saved: {output_path}")
    except Exception as e:
        print(f"  -> Error saving colorbar: {e}")


def process_dlc_file(file_path, output_dir):
    """
    Process a single CSV file to generate a trajectory plot.
    """
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_name}")

    try:
        # Load Data
        df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
        scorer = df.columns.get_level_values(0)[0]
        
        # Extract Coordinates
        try:
            coords = df[scorer][BODYPART_TO_PLOT]
            x_raw = coords['x']
            y_raw = coords['y']
            likelihood = coords['likelihood']
        except KeyError:
            print(f"  -> Error: Body part '{BODYPART_TO_PLOT}' not found in {file_name}")
            return

        # Filter by Likelihood
        valid_points_mask = likelihood >= LIKELIHOOD_THRESHOLD
        x = x_raw[valid_points_mask].to_numpy()
        y = y_raw[valid_points_mask].to_numpy()
        
        # Check data sufficiency
        if len(x) < 2:
            print("  -> Warning: Not enough valid data points. Skipping.")
            return

        # Calculate Speed
        raw_speed = calculate_speed(x, y)
        
        # Smooth Speed
        if SMOOTHING_WINDOW_SIZE > 1:
            speed_series = pd.Series(raw_speed)
            smoothed_speed = speed_series.rolling(
                window=SMOOTHING_WINDOW_SIZE, 
                min_periods=1,
                center=True
            ).mean().to_numpy()
        else:
            smoothed_speed = raw_speed

        # Prepare Line Segments for Plotting
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        
        # Normalize speed for color mapping
        norm = plt.Normalize(0, SPEED_MAX)
        lc = LineCollection(segments, cmap=COLOR_MAP, norm=norm)
        lc.set_array(smoothed_speed[:-1]) # Color based on speed
        lc.set_linewidth(LINE_WIDTH)

        ax.add_collection(lc)
        ax.set_xlim(0, VIDEO_WIDTH)
        ax.set_ylim(0, VIDEO_HEIGHT)
        ax.invert_yaxis() # Invert Y to match video coordinates
        ax.set_aspect('equal', adjustable='box')
        
        # Clean up Axes (No ticks, simple border)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3.0)

        # Save Plot
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_trajectory_{BODYPART_TO_PLOT}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        print(f"  -> Plot saved: {output_filename}")

    except Exception as e:
        print(f"  -> Error processing {file_name}: {e}")
        plt.close()


def main():
    print("=" * 60)
    print("Trajectory Visualization Script")
    print(f"Speed Max Limit: {SPEED_MAX}")
    print("=" * 60)

    # Validate Input
    if not CSV_FILE_PATHS:
        print("Error: No input files specified in 'CSV_FILE_PATHS'. Please edit the script.")
        return

    # Setup Output Directory
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER_NAME)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Saving results to: {output_dir}")

    # Process Files
    print("\n--- Generating Trajectories ---")
    for file_path in CSV_FILE_PATHS:
        if os.path.exists(file_path):
            process_dlc_file(file_path, output_dir)
        else:
            print(f"Warning: File not found: {file_path}")

    # Generate Colorbar
    print("\n--- Generating Legend ---")
    create_standard_colorbar(output_dir)

    print("\n" + "=" * 60)
    print("All tasks completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
