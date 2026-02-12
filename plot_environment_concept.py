"""
Environment visualization for concept environment
Plots obstacles from environment_concept.json
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon

def plot_environment(json_path):
    """Load and plot environment from JSON file"""

    # Load environment data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each obstacle polygon
    for poly_data in data['polygons']:
        vertices = poly_data['vertices']
        polygon = MPLPolygon(vertices,
                            facecolor='gray',
                            edgecolor='black',
                            linewidth=1.5,
                            alpha=0.7)
        ax.add_patch(polygon)

    # Set bounds
    bounds = data['coord_bounds']
    ax.set_xlim(bounds[0] - 10, bounds[1] + 10)
    ax.set_ylim(bounds[2] - 10, bounds[3] + 10)

    # Configure plot
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [units]', fontsize=12)
    ax.set_ylabel('Y [units]', fontsize=12)
    ax.set_title('Environment: Concept Case', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = json_path.replace('.json', '_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Environment plot saved to: {output_path}")

if __name__ == '__main__':
    json_path = r'd:\Data_visualization_code\result\environment_concept.json'
    plot_environment(json_path)
