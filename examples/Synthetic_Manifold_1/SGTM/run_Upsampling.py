"""
Example script demonstrating the use of UpsamplingModule for
upsampling latent graphs and propagating to data space in SGTM.

This script shows how to:
1. Load a trained SGTM network
2. Upsample the latent graph with intermediate nodes
3. Propagate upsampled points to data space
4. Compute curvature on the upsampled manifold
5. Visualize the results
"""

import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add SGTM module path
sys.path.append('../../../SGTM/Python')

from AGTMModule import *
from UpsamplingModule import (
    upsample_latent_graph,
    upsample_single_edge,
    upsample_and_reconstruct_filament,
    compute_all_curvatures,
    get_upsampled_edges_for_plotting,
    plot_upsampled_filament
)

# ============================================================================
# LOAD DATA AND TRAINED MODEL
# ============================================================================

print("Loading data and trained SGTM model...")

# Load original data
NoisyD = pd.read_csv('../Synthetic_Manifold_1.csv', header=None).to_numpy()

# Load trained SGTM network
net, GMDist, NoisyMan = pickle.load(open("../Output/SGTM_output.pkl", 'rb'))

print(f"Loaded {len(net)} trained networks")

# ============================================================================
# EXAMPLE 1: Basic Upsampling
# ============================================================================

print("\n--- Example 1: Basic Upsampling ---")

# Select the first network for demonstration
network_idx = 0
trained_net = net[network_idx]

# Upsample with 10 intermediate nodes per edge
upsampled_points = upsample_latent_graph(trained_net, num_subdivisions=10)

print(f"Original nodes: {trained_net.gmmnet.V_.shape[0]}")
print(f"Total upsampled points: {upsampled_points.shape[0]}")
print(f"Data space dimensions: {upsampled_points.shape[1]}")

# ============================================================================
# EXAMPLE 2: Upsampling with Augmented Graph
# ============================================================================

print("\n--- Example 2: Upsampling with Graph Return ---")

# Get both upsampled points and the augmented graph
upsampled, aug_graph = upsample_latent_graph(
    trained_net, 
    num_subdivisions=5, 
    return_graph=True
)

print(f"Augmented graph nodes: {len(aug_graph.nodes)}")
print(f"Augmented graph edges: {len(aug_graph.edges)}")

# ============================================================================
# EXAMPLE 3: Single Edge Upsampling
# ============================================================================

print("\n--- Example 3: Single Edge Upsampling ---")

# Get list of edges
edges = list(trained_net.rbfnet.graph.edges())
if len(edges) > 0:
    first_edge = edges[0]
    print(f"Upsampling edge: {first_edge}")
    
    # Upsample single edge with 20 subdivisions
    edge_points = upsample_single_edge(trained_net, first_edge, num_subdivisions=20)
    print(f"Points on edge: {edge_points.shape[0]}")

# ============================================================================
# EXAMPLE 4: Curvature Computation
# ============================================================================

print("\n--- Example 4: Curvature Computation ---")

# Compute curvature for all edges
curvatures, edge_list = compute_all_curvatures(trained_net, num_subdivisions=10)

print(f"Number of edges: {len(curvatures)}")
print(f"Mean curvature: {np.mean(curvatures):.6f}")
print(f"Max curvature: {np.max(curvatures):.6f}")
print(f"Min curvature: {np.min(curvatures):.6f}")

# ============================================================================
# EXAMPLE 5: Full Filament Reconstruction
# ============================================================================

print("\n--- Example 5: Full Filament Reconstruction ---")

# Reconstruct the complete filament with smoothing
filament_points, connectivity = upsample_and_reconstruct_filament(
    trained_net,
    num_subdivisions=10,
    smooth=True,
    smoothing_window=3
)

print(f"Total filament points: {filament_points.shape[0]}")
print(f"Number of connections: {len(connectivity)}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n--- Generating Visualization ---")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Original data with original graph
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.scatter(NoisyD[:, 0], NoisyD[:, 1], NoisyD[:, 2], s=0.1, alpha=0.3, c='gray')
V_original = trained_net.gmmnet.V_
ax1.scatter(V_original[:, 0], V_original[:, 1], V_original[:, 2], 
            c='red', s=50, marker='D', label='Original nodes')
ax1.set_title('Original SGTM Nodes')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Plot 2: Upsampled graph
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.scatter(NoisyD[:, 0], NoisyD[:, 1], NoisyD[:, 2], s=0.1, alpha=0.3, c='gray')

# Plot upsampled edges
edge_curves = get_upsampled_edges_for_plotting(trained_net, num_subdivisions=10)
for curve in edge_curves:
    ax2.plot(curve[:, 0], curve[:, 1], curve[:, 2], 
             'b-', linewidth=1.5, marker='.', markersize=2)

ax2.scatter(V_original[:, 0], V_original[:, 1], V_original[:, 2], 
            c='red', s=50, marker='D', label='Original nodes', zorder=10)
ax2.set_title('Upsampled SGTM (10 subdivisions/edge)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

# Plot 3: Curvature visualization
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.scatter(NoisyD[:, 0], NoisyD[:, 1], NoisyD[:, 2], s=0.1, alpha=0.2, c='gray')

# Color edges by curvature
norm_curv = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min() + 1e-10)
cmap = plt.cm.hot

for i, curve in enumerate(edge_curves):
    color = cmap(norm_curv[i])
    ax3.plot(curve[:, 0], curve[:, 1], curve[:, 2], 
             color=color, linewidth=2)

ax3.set_title('Edges Colored by Curvature')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(curvatures.min(), curvatures.max()))
sm.set_array([])
plt.colorbar(sm, ax=ax3, label='Curvature', shrink=0.6)

# Plot 4: Dense reconstruction
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.scatter(NoisyD[:, 0], NoisyD[:, 1], NoisyD[:, 2], s=0.1, alpha=0.2, c='gray')

# Plot dense reconstruction from upsample_and_reconstruct_filament
ax4.scatter(filament_points[:, 0], filament_points[:, 1], filament_points[:, 2],
            c='blue', s=5, alpha=0.8, label='Dense reconstruction')
ax4.scatter(V_original[:, 0], V_original[:, 1], V_original[:, 2], 
            c='red', s=50, marker='D', label='Original nodes', zorder=10)
ax4.set_title('Dense Filament Reconstruction (Smoothed)')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.legend()

plt.tight_layout()
plt.savefig('../Images/7.-Upsampling_output.png', dpi=150)
print("Saved figure to ../Images/7.-Upsampling_output.png")

plt.show()

print("\n--- Script Complete ---")
