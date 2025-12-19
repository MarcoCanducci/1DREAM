"""
UpsamplingModule.py - Latent Graph Upsampling and Data Space Propagation for SGTM

This module provides functions to upsample the latent graph used in SGTM (Supervised 
Generative Topographic Mapping) and propagate the upsampled points to the data space 
using the trained RBF network.

The upsampling is performed by adding intermediate nodes along the edges of the latent 
graph, and the propagation uses geodesic distances (shortest paths) on the augmented 
graph to compute RBF activations.

Copyright (C) 2024
Based on MATLAB implementation by Marco Canducci

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Optional, Union
from copy import deepcopy


def upsample_latent_graph(
    net,
    num_subdivisions: int = 5,
    edge_indices: Optional[List[int]] = None,
    return_graph: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, nx.Graph]]:
    """
    Upsample the latent graph by adding intermediate nodes on edges and propagate
    them to the data space using the trained RBF network.

    This function subdivides selected (or all) edges of the latent graph by adding
    equally-spaced intermediate nodes along each edge. The new nodes are then
    mapped to the data space using the RBF network weights and geodesic distances.

    Parameters
    ----------
    net : net object (from AGTMModule)
        The trained SGTM network containing:
        - net.rbfnet: The RBF network with centers (c), width (c_width), 
                      weights (W), and activation matrix (Phi)
        - net.gmmnet: The GMM network with data space positions (V_)
        - net.graph: The latent graph structure

    num_subdivisions : int, optional (default=5)
        Number of intermediate nodes to add on each edge.
        Higher values produce smoother curves but increase computation.

    edge_indices : list of int, optional
        Indices of edges to upsample. If None, all edges are upsampled.
        Useful for selective upsampling of specific filament segments.

    return_graph : bool, optional (default=False)
        If True, also return the augmented NetworkX graph with upsampled nodes.

    Returns
    -------
    upsampled_points : np.ndarray
        Array of shape (n_total_points, data_dim) containing:
        - Original node positions in data space
        - New interpolated positions for all subdivided edges
        
    augmented_graph : nx.Graph (only if return_graph=True)
        The augmented latent graph with additional nodes on edges.

    Examples
    --------
    >>> # Basic upsampling with 5 subdivisions per edge
    >>> upsampled = upsample_latent_graph(trained_net, num_subdivisions=5)
    
    >>> # Get both upsampled points and the augmented graph
    >>> upsampled, aug_graph = upsample_latent_graph(
    ...     trained_net, num_subdivisions=10, return_graph=True
    ... )
    
    >>> # Upsample only specific edges (e.g., first 3 edges)
    >>> upsampled = upsample_latent_graph(trained_net, edge_indices=[0, 1, 2])

    Notes
    -----
    The upsampling process works as follows:
    1. For each edge (u, v), we add `num_subdivisions` nodes between u and v
    2. Edge weights for the new edges are set to create a smooth transition
    3. Geodesic distances from RBF centers to new nodes are computed
    4. RBF activations are calculated: Phi = exp(-dist^2 / (2 * c_width^2))
    5. Data space positions are computed as: Y = (Phi @ W)^T * sigma + mu

    See Also
    --------
    rbf_map_continuous : Low-level function for mapping arbitrary nodes
    upsample_single_edge : Upsample a single edge with custom parameters
    """
    
    # Extract network components
    rbfnet = net.rbfnet
    gmmnet = net.gmmnet
    graph = rbfnet.graph
    
    # Get original graph structure
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    n_original_nodes = len(graph.nodes)
    
    # Get edge list
    edges = list(graph.edges())
    n_edges = len(edges)
    
    # Filter edges if specific indices provided
    if edge_indices is not None:
        edges = [edges[i] for i in edge_indices]
    
    # Original node positions in data space
    V_original = gmmnet.V_
    
    # Initialize list to store all upsampled points
    all_upsampled_points = [V_original.copy()]
    
    # Create augmented graph for distance computation
    augmented_graph = nx.DiGraph(adj_matrix)
    
    # Track new node index
    new_node_idx = n_original_nodes
    
    # Store mapping of new nodes to their corresponding edges
    new_nodes_per_edge = {}
    
    for edge_idx, (u, v) in enumerate(edges):
        # Create new node indices for this edge
        new_nodes = list(range(new_node_idx, new_node_idx + num_subdivisions))
        new_nodes_per_edge[(u, v)] = new_nodes
        new_node_idx += num_subdivisions
        
        # Create edge weight distribution (linear interpolation)
        t = 1.0 / (num_subdivisions + 1)
        weights = np.ones(num_subdivisions + 1) * t
        weights[-1] = 1 - num_subdivisions * t  # Ensure weights sum to original edge weight
        
        # Add new edges to augmented graph
        # Edge structure: u -> new_nodes[0] -> new_nodes[1] -> ... -> v
        edge_list = [(u, new_nodes[0])]
        for i in range(len(new_nodes) - 1):
            edge_list.append((new_nodes[i], new_nodes[i + 1]))
        edge_list.append((new_nodes[-1], v))
        
        # Set unit weights for original edges
        for n in graph.nodes():
            for neighbor in graph.neighbors(n):
                if not augmented_graph.has_edge(n, neighbor):
                    augmented_graph.add_edge(n, neighbor, weight=1.0)
                else:
                    augmented_graph[n][neighbor]['weight'] = 1.0
        
        # Add new edges with appropriate weights
        for i, (src, dst) in enumerate(edge_list):
            augmented_graph.add_edge(src, dst, weight=weights[i])
            augmented_graph.add_edge(dst, src, weight=weights[i])  # Bidirectional
    
    # Compute upsampled points for all new nodes
    all_new_nodes = []
    for (u, v), new_nodes in new_nodes_per_edge.items():
        all_new_nodes.extend(new_nodes)
    
    if len(all_new_nodes) > 0:
        # Map new nodes to data space
        upsampled_data_points = rbf_map_continuous(
            augmented_graph, 
            all_new_nodes, 
            rbfnet,
            gmmnet
        )
        all_upsampled_points.append(upsampled_data_points)
    
    # Combine all points
    upsampled_points = np.vstack(all_upsampled_points)
    
    if return_graph:
        return upsampled_points, augmented_graph
    return upsampled_points


def rbf_map_continuous(
    graph: nx.Graph,
    node_indices: List[int],
    rbfnet,
    gmmnet,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Map arbitrary nodes on the latent graph to the data space using RBF interpolation.

    This is the core function for propagating latent space positions to data space.
    It computes geodesic distances from RBF centers to the target nodes and applies
    the Gaussian RBF activation function.

    Parameters
    ----------
    graph : nx.Graph
        The (potentially augmented) latent graph with edge weights.
        
    node_indices : list of int
        Indices of nodes to map to data space.
        
    rbfnet : rbfnet object
        The trained RBF network containing:
        - c: RBF center indices
        - c_width: RBF kernel width
        - W: Weight matrix for mapping to data space
        
    gmmnet : gmmnet object
        The GMM network containing:
        - V_: Data space positions (used for standardization recovery)
        - W: Weight matrix
        
    mu : np.ndarray, optional
        Mean for de-standardization. If None, uses zero mean.
        
    sigma : np.ndarray, optional
        Standard deviation for de-standardization. If None, uses unit std.

    Returns
    -------
    data_points : np.ndarray
        Array of shape (len(node_indices), data_dim) containing
        the data space positions of the input nodes.

    Notes
    -----
    The mapping follows these steps:
    1. Compute shortest path distances from RBF centers to target nodes
    2. Apply Gaussian RBF: Phi = exp(-dist^2 / (2 * c_width^2))
    3. Compute data space positions: Y = Phi @ W
    4. De-standardize: Y_final = Y * sigma + mu

    This is equivalent to the MATLAB function `rbfmap_cont` in 
    Compute_CurvatureOn_rbf_FiniteDifference.m
    """
    
    # Extract RBF parameters
    eps_centers = rbfnet.c.flatten().astype(int)
    c_width = rbfnet.c_width
    W = gmmnet.W
    
    # Get data dimensionality from weight matrix
    if W.ndim == 1:
        data_dim = len(W)
    else:
        data_dim = W.shape[1] if W.shape[0] < W.shape[1] else W.shape[0]
    
    # Set default standardization parameters
    if mu is None:
        mu = np.zeros(data_dim)
    if sigma is None:
        sigma = np.ones(data_dim)
    
    # Compute geodesic distances from RBF centers to target nodes
    dist = compute_geodesic_distances(graph, eps_centers, node_indices)
    
    # Compute RBF activations (Gaussian kernel)
    # Using factor of 3 as in the original implementation
    factor = 3
    arg_exp = -(dist ** 2) / (2 * (factor * c_width) ** 2)
    Phi_new = np.exp(arg_exp)
    
    # Map to data space: Y = Phi^T @ W
    data_points = np.dot(Phi_new.T, W)
    
    # De-standardize
    data_points = data_points * sigma + mu
    
    return data_points


def compute_geodesic_distances(
    graph: nx.Graph,
    source_nodes: np.ndarray,
    target_nodes: List[int]
) -> np.ndarray:
    """
    Compute geodesic (shortest path) distances between source and target nodes.

    Parameters
    ----------
    graph : nx.Graph
        The graph on which to compute distances.
        
    source_nodes : np.ndarray
        Array of source node indices.
        
    target_nodes : list of int
        List of target node indices.

    Returns
    -------
    distances : np.ndarray
        Array of shape (len(source_nodes), len(target_nodes)) containing
        the shortest path distances.
    """
    
    # Compute all shortest path lengths
    all_lengths = dict(nx.shortest_path_length(graph, weight='weight'))
    
    # Build distance matrix
    n_sources = len(source_nodes)
    n_targets = len(target_nodes)
    dist = np.zeros((n_sources, n_targets))
    
    for i, s in enumerate(source_nodes):
        source_lengths = all_lengths.get(int(s), {})
        for j, t in enumerate(target_nodes):
            dist[i, j] = source_lengths.get(int(t), np.inf)
    
    return dist


def upsample_single_edge(
    net,
    edge: Tuple[int, int],
    num_subdivisions: int = 5,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Upsample a single edge of the latent graph.

    This function is useful for detailed analysis of specific filament segments,
    such as curvature computation or local structure investigation.

    Parameters
    ----------
    net : net object
        The trained SGTM network.
        
    edge : tuple of (int, int)
        The edge to upsample, specified as (source_node, target_node).
        
    num_subdivisions : int, optional (default=5)
        Number of intermediate nodes to add.
        
    weights : np.ndarray, optional
        Custom edge weights for the subdivisions. If None, uses uniform weights.
        Should have length (num_subdivisions + 1).

    Returns
    -------
    edge_points : np.ndarray
        Array of shape (num_subdivisions + 2, data_dim) containing:
        - First row: source node position in data space
        - Middle rows: interpolated positions
        - Last row: target node position in data space
    """
    
    rbfnet = net.rbfnet
    gmmnet = net.gmmnet
    graph = rbfnet.graph
    
    u, v = edge
    n_original = len(graph.nodes)
    
    # Create augmented graph
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    aug_graph = nx.DiGraph(adj_matrix)
    
    # Set unit weights for all original edges
    for n in graph.nodes():
        for neighbor in graph.neighbors(n):
            aug_graph[n][neighbor]['weight'] = 1.0
    
    # Create new node indices
    new_nodes = list(range(n_original, n_original + num_subdivisions))
    
    # Set up weights
    if weights is None:
        t = 1.0 / (num_subdivisions + 1)
        weights = np.ones(num_subdivisions + 1) * t
        weights[-1] = 1 - num_subdivisions * t
    
    # Add new edges
    edge_list = [(u, new_nodes[0])]
    for i in range(len(new_nodes) - 1):
        edge_list.append((new_nodes[i], new_nodes[i + 1]))
    edge_list.append((new_nodes[-1], v))
    
    for i, (src, dst) in enumerate(edge_list):
        aug_graph.add_edge(src, dst, weight=weights[i])
        aug_graph.add_edge(dst, src, weight=weights[i])
    
    # Get endpoint positions
    V_original = gmmnet.V_
    start_pos = V_original[u:u+1, :]
    end_pos = V_original[v:v+1, :]
    
    # Map new nodes to data space
    if len(new_nodes) > 0:
        new_positions = rbf_map_continuous(aug_graph, new_nodes, rbfnet, gmmnet)
        edge_points = np.vstack([start_pos, new_positions, end_pos])
    else:
        edge_points = np.vstack([start_pos, end_pos])
    
    return edge_points


def upsample_and_reconstruct_filament(
    net,
    num_subdivisions: int = 5,
    smooth: bool = False,
    smoothing_window: int = 3
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Upsample the entire latent graph and return a densely sampled filament.

    This function provides a complete reconstruction of the manifold by
    upsampling all edges and optionally applying smoothing.

    Parameters
    ----------
    net : net object
        The trained SGTM network.
        
    num_subdivisions : int, optional (default=5)
        Number of intermediate nodes to add per edge.
        
    smooth : bool, optional (default=False)
        Whether to apply moving average smoothing to the result.
        
    smoothing_window : int, optional (default=3)
        Window size for moving average smoothing (only used if smooth=True).

    Returns
    -------
    upsampled_filament : np.ndarray
        Dense array of data space positions representing the filament.
        
    edge_connectivity : list of tuples
        Connectivity information for the upsampled points,
        useful for plotting or further analysis.
    """
    
    rbfnet = net.rbfnet
    gmmnet = net.gmmnet
    graph = rbfnet.graph
    
    edges = list(graph.edges())
    V_original = gmmnet.V_
    
    all_edge_points = []
    edge_connectivity = []
    current_idx = 0
    
    for u, v in edges:
        edge_points = upsample_single_edge(net, (u, v), num_subdivisions)
        
        # Track connectivity (excluding first point which is the source node)
        n_points = len(edge_points)
        for i in range(n_points - 1):
            edge_connectivity.append((current_idx + i, current_idx + i + 1))
        
        all_edge_points.append(edge_points)
        current_idx += n_points
    
    # Combine all edge points
    upsampled_filament = np.vstack(all_edge_points)
    
    # Optional smoothing
    if smooth and smoothing_window > 1:
        from scipy.ndimage import uniform_filter1d
        for dim in range(upsampled_filament.shape[1]):
            upsampled_filament[:, dim] = uniform_filter1d(
                upsampled_filament[:, dim], 
                size=smoothing_window, 
                mode='nearest'
            )
    
    return upsampled_filament, edge_connectivity


def compute_curvature_on_edge(
    net,
    edge: Tuple[int, int],
    num_subdivisions: int = 5,
    h: float = 1e-2
) -> float:
    """
    Compute the curvature on a single edge using finite differences.

    This is a Python implementation of the curvature computation from
    Compute_CurvatureOn_rbf_FiniteDifference.m

    Parameters
    ----------
    net : net object
        The trained SGTM network.
        
    edge : tuple of (int, int)
        The edge on which to compute curvature.
        
    num_subdivisions : int, optional (default=5)
        Number of subdivisions for curvature estimation.
        
    h : float, optional (default=1e-2)
        Step size for finite difference approximation.

    Returns
    -------
    curvature : float
        The estimated curvature value for the edge.

    Notes
    -----
    Curvature is estimated using the second derivative of the parameterized
    curve. For a curve y(t) in data space, the curvature κ is related to:
    κ ≈ ||d²y/dt²|| / ||dy/dt||²
    
    The finite difference approximation uses:
    d²y/dt² ≈ (y(t+h) - 2*y(t) + y(t-h)) / h²
    """
    
    # Get upsampled edge points
    edge_points = upsample_single_edge(net, edge, num_subdivisions)
    
    # Compute curvature using finite differences
    # Use central difference for second derivative
    n_points = len(edge_points)
    
    if n_points < 3:
        return 0.0
    
    curvatures = []
    for i in range(1, n_points - 1):
        # Second derivative approximation
        d2y = (edge_points[i+1] - 2 * edge_points[i] + edge_points[i-1])
        
        # First derivative approximation (for normalization)
        dy = (edge_points[i+1] - edge_points[i-1]) / 2
        
        # Curvature magnitude
        curvature = np.linalg.norm(d2y) / (np.linalg.norm(dy) ** 2 + 1e-10)
        curvatures.append(curvature)
    
    return np.mean(curvatures)


def compute_all_curvatures(
    net,
    num_subdivisions: int = 5,
    h: float = 1e-2
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute curvature for all edges in the latent graph.

    Parameters
    ----------
    net : net object
        The trained SGTM network.
        
    num_subdivisions : int, optional (default=5)
        Number of subdivisions per edge.
        
    h : float, optional (default=1e-2)
        Step size for finite differences.

    Returns
    -------
    curvatures : np.ndarray
        Array of curvature values, one per edge.
        
    edges : list of tuples
        List of edges corresponding to the curvature values.
    """
    
    graph = net.rbfnet.graph
    edges = list(graph.edges())
    
    curvatures = np.zeros(len(edges))
    for i, edge in enumerate(edges):
        curvatures[i] = compute_curvature_on_edge(net, edge, num_subdivisions, h)
    
    return curvatures, edges


# Utility functions for visualization

def get_upsampled_edges_for_plotting(
    net,
    num_subdivisions: int = 5
) -> List[np.ndarray]:
    """
    Get upsampled edge curves suitable for 3D plotting.

    Parameters
    ----------
    net : net object
        The trained SGTM network.
        
    num_subdivisions : int, optional (default=5)
        Number of subdivisions per edge.

    Returns
    -------
    edge_curves : list of np.ndarray
        List of arrays, each containing the data space coordinates
        for one upsampled edge. Useful for matplotlib plot3D.
    """
    
    graph = net.rbfnet.graph
    edges = list(graph.edges())
    
    edge_curves = []
    for edge in edges:
        curve = upsample_single_edge(net, edge, num_subdivisions)
        edge_curves.append(curve)
    
    return edge_curves


def plot_upsampled_filament(
    net,
    num_subdivisions: int = 5,
    ax=None,
    color: str = 'blue',
    linewidth: float = 1.5,
    marker: str = 'o',
    markersize: float = 3,
    show_original_nodes: bool = True,
    original_node_color: str = 'red',
    original_node_size: float = 50
):
    """
    Plot the upsampled filament in 3D.

    Parameters
    ----------
    net : net object
        The trained SGTM network.
        
    num_subdivisions : int, optional (default=5)
        Number of subdivisions per edge.
        
    ax : matplotlib 3D axis, optional
        Axis to plot on. If None, creates a new figure.
        
    color : str, optional (default='blue')
        Color for the upsampled curves.
        
    linewidth : float, optional (default=1.5)
        Line width for plotting.
        
    marker : str, optional (default='o')
        Marker style for interpolated points.
        
    markersize : float, optional (default=3)
        Size of markers.
        
    show_original_nodes : bool, optional (default=True)
        Whether to highlight the original graph nodes.
        
    original_node_color : str, optional (default='red')
        Color for original node markers.
        
    original_node_size : float, optional (default=50)
        Size of original node markers.

    Returns
    -------
    ax : matplotlib axis
        The axis with the plot.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Get upsampled edge curves
    edge_curves = get_upsampled_edges_for_plotting(net, num_subdivisions)
    
    # Plot each edge
    for curve in edge_curves:
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                color=color, linewidth=linewidth, 
                marker=marker, markersize=markersize)
    
    # Highlight original nodes
    if show_original_nodes:
        V_original = net.gmmnet.V_
        ax.scatter(V_original[:, 0], V_original[:, 1], V_original[:, 2],
                   c=original_node_color, s=original_node_size, 
                   marker='D', zorder=10, label='Original nodes')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    return ax
