"""
Spatial Heatmap Visualization for Open Field Test

Description:
    This script generates spatial density heatmaps from DeepLabCut CSV files.
    It uses 2D histogram binning with Gaussian smoothing to visualize where the animal
    spent the most time. It also generates a separate colorbar legend.

    - Filters low-confidence tracking points.
    - Applies log-scaling and Gaussian smoothing for better visualization.
    - Generates a "Low" to "High" density colorbar.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy.ndimage import gaussian_filter
from plotnine import *
import warnings

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
OUTPUT_FOLDER_NAME = "heatmap_results"

# 3. Data Processing Parameters
BODYPART_TO_PLOT = 'HEAD'      # Body part to track
LIKELIHOOD_THRESHOLD = 0.60    # Confidence threshold
VIDEO_WIDTH = 500              # Arena width in pixels
VIDEO_HEIGHT = 500             # Arena height in pixels

# 4. Heatmap Algorithm Parameters
BINS = 120                     # Resolution of the grid (higher = finer details)
DENSITY_SMOOTH = 1.5           # Sigma for Gaussian filter (0 = no smoothing)
USE_LOG_SCALE = True           # Apply log1p transform to highlight low-density areas
PERCENTILE_CLIP = 99           # Clip top X% intensities to prevent hotspots from dominating
INTERPOLATION = 'gaussian'     # Interpolation method for imshow

# 5. Visualization Settings
DPI = 300                      # Output resolution
FIG_SIZE = (8, 8)              # Figure size in inches

# 6. Color Scheme (Blue -> Red)
COLORS = [
    '#0F467F', '#317CB6', '#6EAED1', '#B7D8E7', '#E8F2F4',
    '#FAE4D6', '#F8B193', '#DD6F58', '#B52330', '#6F011E'
]
CMAP_NODES = 256

# Initialize the global colormap object
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_heatmap_cmap", COLORS, N=CMAP_NODES)


# ==========================================
# --- FUNCTION DEFINITIONS ---
# ==========================================

def create_heatmap_colorbar(output_dir):
    """
    Generate a standalone colorbar legend labeled 'Low' and 'High'.
    """
    print("Generating heatmap colorbar...")
    
    # Create dummy data for the gradient
    dummy_data = pd.DataFrame({
        'x': np.repeat(0, 100),
        'y': np.linspace(0, 100, 100),
        'density': np.linspace(0, 100, 100)
    })
    
    custom_breaks = [0, 100]
    custom_labels = ["Low", "High"]
    
    # Create plot using plotnine
    p = (ggplot(dummy_data, aes(x='x', y='y', color='density'))
         + geom_point(alpha=0) # Invisible points
         + scale_color_gradientn(
             colors=COLORS,
             name="",
             limits=[0, 100],
             breaks=custom_breaks,
             labels=custom_labels
         )
         + theme_void()
         + theme(
             legend_position='right',
             legend_title=element_blank(),
             legend_text=element_text(size=12, color='black', margin={'l': 5}),
             legend_ticks=element_line(color='black'),
             legend_key_width=20,
             legend_key_height=120,
             figure_size=(0.7, 3),
             plot_margin=0,
             panel_spacing=0
         ))
         
    output_path = os.path.join(output_dir, "colorbar_heatmap.pdf")
    try:
        p.save(output_path, dpi=300, verbose=False, bbox_inches='tight')
        print(f"  -> Colorbar saved: {output_path}")
    except Exception as e:
        print(f"  -> Error saving colorbar: {e}")


def create_heatmap_from_dlc(file_path, output_dir):
    """
    Process a single DLC CSV file and generate a heatmap.
    """
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_name}")

    try:
        # 1. Load Data
        df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
        scorer = df.columns.get_level_values(0)[0]
        
        # Extract Coordinates
        try:
            coords = df[scorer][BODYPART_TO_PLOT]
            x_raw = coords['x']
            y_raw = coords['y']
            likelihood = coords['likelihood']
        except KeyError:
            print(f"  -> Error: Body part '{BODYPART_TO_PLOT}' not found.")
            return

        # 2. Filter Data
        valid_points_mask = likelihood >= LIKELIHOOD_THRESHOLD
        x_filtered = x_raw[valid_points_mask]
        y_filtered = y_raw[valid_points_mask]
        
        if len(x_filtered) < 10:
            print("  -> Warning: Not enough valid data points. Skipping.")
            return

        # 3. Generate 2D Histogram (Binning)
        heatmap, xedges, yedges = np.histogram2d(
            x=x_filtered, 
            y=y_filtered, 
            bins=BINS,
            range=[[0, VIDEO_WIDTH], [0, VIDEO_HEIGHT]]
        )
        
        # 4. Apply Transformations
        # Log Scale
        if USE_LOG_SCALE:
            heatmap = np.log1p(heatmap) # log(x+1)
            
        # Gaussian Smoothing
        if DENSITY_SMOOTH > 0:
            heatmap = gaussian_filter(heatmap, sigma=DENSITY_SMOOTH)
        
        # Percentile Clipping (Calculation of Vmax)
        if heatmap.max() > 0:
            # Calculate percentile only on non-zero values to avoid skewing
            vmax = np.percentile(heatmap[heatmap > 0], PERCENTILE_CLIP)
        else:
            vmax = 1
        
        # 5. Plotting
        plt.figure(figsize=FIG_SIZE, dpi=DPI)
        ax = plt.gca()

        # Display Heatmap
        # Note: Transpose (heatmap.T) is required because histogram2d returns (x,y) 
        # but imshow expects (rows, cols) which is (y,x).
        plt.imshow(heatmap.T, 
                  origin='lower',
                  cmap=CUSTOM_CMAP,
                  vmin=0,
                  vmax=vmax,
                  interpolation=INTERPOLATION,
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        # Style Adjustments
        ax.set_xlim(0, VIDEO_WIDTH)
        ax.set_ylim(0, VIDEO_HEIGHT)
        ax.invert_yaxis() # Match video coordinate system (0,0 at top-left)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off') # Hide axes
        
        # 6. Save Output
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_heatmap_{BODYPART_TO_PLOT}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        plt.savefig(output_path,
                   format='png',
                   bbox_inches='tight',
                   pad_inches=0.0,
                   transparent=True,
                   dpi=DPI)
        plt.close()
        
        print(f"  -> Heatmap saved: {output_filename}")

    except Exception as e:
        print(f"  -> Error processing {file_name}: {e}")
        plt.close()


def main():
    print("=" * 60)
    print("Spatial Heatmap Generation Script")
    print("=" * 60)

    # Validate Input
    if not CSV_FILE_PATHS:
        print("Error: No input files specified in 'CSV_FILE_PATHS'. Please edit the script.")
        return

    # Setup Output Directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER_NAME)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Saving results to: {output_dir}")

    # Process Files
    print("\n--- Generating Heatmaps ---")
    for file_path in CSV_FILE_PATHS:
        if os.path.exists(file_path):
            create_heatmap_from_dlc(file_path, output_dir)
        else:
            print(f"Warning: File not found: {file_path}")

    # Generate Colorbar
    print("\n--- Generating Legend ---")
    create_heatmap_colorbar(output_dir)

    print("\n" + "=" * 60)
    print("All tasks completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
