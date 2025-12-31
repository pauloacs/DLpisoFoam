"""
Domain boundary detection, obstacle clustering, and distance calculations.
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _cluster_square_cylinders_3d(
    obst_points: np.ndarray,
    grid_res: float,
    eps_multiplier: float = 1.5):
    """
    Cluster obstacle points in 3D based on proximity in all directions.
    Creates axis-aligned bounding boxes for each cluster (squared cylinders).
    
    Args:
        obst_points: Obstacle boundary points (N, 3) array
        grid_res: Grid resolution
        eps_multiplier: Multiplier for DBSCAN epsilon (neighborhood radius)
        
    Returns:
        List of tuples (x_min, x_max, y_min, y_max, z_min, z_max) for each cluster
    """
    if obst_points.size == 0 or obst_points.shape[0] < 4:
        return []
    
    # Use 3D DBSCAN clustering based on proximity in all directions
    eps = grid_res * eps_multiplier
    min_samples = max(4, int(np.cbrt(obst_points.shape[0]) * 0.1))  # Adaptive min_samples
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(obst_points)
    labels = clustering.labels_
    
    # Get unique clusters (excluding noise labeled as -1)
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    
    boxes = []
    for label in unique_labels:
        cluster_pts = obst_points[labels == label]
        
        if cluster_pts.shape[0] < 4:
            continue
            
        # Create axis-aligned bounding box for this cluster
        x_min, x_max = cluster_pts[:, 0].min(), cluster_pts[:, 0].max()
        y_min, y_max = cluster_pts[:, 1].min(), cluster_pts[:, 1].max()
        z_min, z_max = cluster_pts[:, 2].min(), cluster_pts[:, 2].max()
        
        boxes.append((x_min, x_max, y_min, y_max, z_min, z_max))
    
    return boxes


def _visualize_clusters_3d(
    obst_points: np.ndarray,
    boxes: list,
    fig_path: str = None):
    """
    Visualize the 3D clustered obstacles as bounding boxes.
    
    Args:
        obst_points: Original obstacle points
        boxes: List of bounding boxes (x_min, x_max, y_min, y_max, z_min, z_max)
        fig_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(obst_points[:, 0], obst_points[:, 1], obst_points[:, 2], 
               c='gray', s=1, alpha=0.3, label='Obstacle Points')
    
    # Plot bounding boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(boxes)))
    
    for i, (x0, x1, y0, y1, z0, z1) in enumerate(boxes):
        # Create vertices of the box
        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]   # top
        ]
        
        # Define the 12 edges of the box
        edges = [
            [vertices[0], vertices[1]], [vertices[1], vertices[2]], 
            [vertices[2], vertices[3]], [vertices[3], vertices[0]],
            [vertices[4], vertices[5]], [vertices[5], vertices[6]], 
            [vertices[6], vertices[7]], [vertices[7], vertices[4]],
            [vertices[0], vertices[4]], [vertices[1], vertices[5]], 
            [vertices[2], vertices[6]], [vertices[3], vertices[7]]
        ]
        
        # Plot edges
        for edge in edges:
            edge = np.array(edge)
            ax.plot3D(*edge.T, color=colors[i], linewidth=2, alpha=0.8)
        
        # Add semi-transparent faces
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]]   # top
        ]
        
        face_collection = Poly3DCollection(faces, alpha=0.2, facecolor=colors[i], edgecolor='none')
        ax.add_collection3d(face_collection)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'3D Obstacle Clustering ({len(boxes)} clusters)', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _distance_to_boxes(xyz: np.ndarray, boxes: list):
    """
    Compute minimum outside distance from each point to list of axis-aligned boxes.
    Inside a box => distance 0.
    
    Args:
        xyz: Query points (N, 3)
        boxes: List of (x_min, x_max, y_min, y_max, z_min, z_max) tuples
        
    Returns:
        ndarray: Minimum distance to any box for each point
    """
    if not boxes:
        return np.full(xyz.shape[0], np.inf)
    x = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    d_all = []
    for (x0, x1, y0, y1, z0, z1) in boxes:
        dx = np.where(x < x0, x0 - x, np.where(x > x1, x - x1, 0))
        dy = np.where(y < y0, y0 - y, np.where(y > y1, y - y1, 0))
        dz = np.where(z < z0, z0 - z, np.where(z > z1, z - z1, 0))
        d_all.append(np.sqrt(dx*dx + dy*dy + dz*dz))
    return np.min(np.vstack(d_all), axis=0)


def domain_dist(
    boundaries,
    xyz0,
    grid_res,
    find_limited_index=True,
    eps_multiplier=5,
    visualize=True):
    """
    Generalized domain + signed distance for multiple 3D squared cylinders.
    
    Args:
        boundaries: Dictionary with boundary arrays
        xyz0: Query points (N, 3)
        grid_res: Grid resolution
        find_limited_index: Whether to trim -100.0 sentinels
        eps_multiplier: DBSCAN epsilon multiplier for clustering
        visualize: Whether to visualize the clusters
    
    Returns:
        domain_bool: Boolean mask for flow domain
        sdf: Signed distance field
    """
    z_top = boundaries['z_top_boundary']
    y_top = boundaries['y_top_boundary']
    y_bot = boundaries['y_bot_boundary']
    z_bot = boundaries['z_bot_boundary']
    obst = boundaries['obst_boundary']

    # Always compute the domain limits from the boundaries
    max_x, max_y, min_x, min_y = (
        np.max(z_top[:,0]) + grid_res,
        np.max(y_top[:,1]) + grid_res,
        np.min(z_top[:,0]) - grid_res,
        np.min(y_bot[:,1]) - grid_res
    )
    max_z, min_z = np.max(z_top[:,2]), np.min(z_bot[:,2])

    # Since most times the outer domain is a parallelogram
    # using this simplified approach
    is_inside_domain = (
        (xyz0[:, 0] <= max_x) &
        (xyz0[:, 0] >= min_x) &
        (xyz0[:, 1] <= max_y) &
        (xyz0[:, 1] >= min_y) &
        (xyz0[:, 2] <= max_z) &
        (xyz0[:, 2] >= min_z)
    )

    # Cluster obstacles in 3D
    boxes = _cluster_square_cylinders_3d(obst, grid_res, eps_multiplier)
    
    # Visualize if requested
    if visualize and len(boxes) > 0:
        _visualize_clusters_3d(obst, boxes, fig_path='obstacle_clusters_3d.png')

    # Point-in-box test (vectorized)
    x = xyz0[:, 0]; y = xyz0[:, 1]; z = xyz0[:, 2]
    inside_obst = np.zeros(xyz0.shape[0], dtype=bool)
    for (x0, x1, y0, y1, z0, z1) in boxes:
        inside_obst |= (
            (x >= x0) & (x <= x1) &
            (y >= y0) & (y <= y1) &
            (z >= z0) & (z <= z1)
        )

    flow_bool = is_inside_domain & ~inside_obst

    # Distance to boxes (vectorized)
    if len(boxes) == 0:
        obst_dist = np.full(xyz0.shape[0], np.inf)
    else:
        obst_dist = _distance_to_boxes(xyz0, boxes)

    # Boundary distances via KDTree (subsampled for speed)
    step = max(1, int(np.sqrt(obst.shape[0]) / 100)) if obst.shape[0] > 0 else 1
    
    def build_tree(arr):
        if arr.size == 0:
            return None
        return cKDTree(arr[::step])

    trees = [build_tree(a) for a in [z_top, z_bot, y_top, y_bot]]

    def nearest(tree):
        if tree is None:
            return np.full(xyz0.shape[0], np.inf)
        return tree.query(xyz0, k=1)[0]

    dist_surfaces = [nearest(t) for t in trees]
    sdf = np.minimum.reduce([obst_dist] + dist_surfaces) * flow_bool

    return flow_bool, sdf
