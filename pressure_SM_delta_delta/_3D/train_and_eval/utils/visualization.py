"""
All plotting and visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import pyvista as pv
import imageio


def createGIF(first_sim, last_sim, first_t, last_t):
    """
    Create a GIF with all the frames.
    
    Args:
        n_sims: Number of simulations
        n_ts: Number of time steps
    """
    for sim in range(first_sim, last_sim+1):
        
        filenamesp = []

        for time in range(first_t, last_t):
            filenamesp.append(f'plots/sim{sim}/deltap_pred_t{time}_slices.png')  # hardcoded to get the frames in order

        with imageio.get_writer(f'plots/p_movie_sim{sim}.gif', mode='I', fps=2) as writer:
            for filename in filenamesp:
                image = imageio.imread(filename)
                writer.append_data(image)


def get_facecolors(data_slice, norm, cmap):
    """
    Map data to colors, ensuring NaNs are transparent.
    
    Args:
        data_slice: 2D slice of data
        norm: Normalization object
        cmap: Colormap
        
    Returns:
        ndarray: RGBA colors
    """
    normed_data = norm(data_slice)  # Normalize data
    rgba_colors = cmap(normed_data)  # Map to colormap
    rgba_colors[np.isnan(data_slice)] = [0, 0, 0, 0]  # Make NaNs fully transparent
    return rgba_colors


def plot_random_blocks_3d_render(res_concat, y_array, x_array, sim, time, save_plots):
    """
    Plot 9 randomly sampled blocks for reference (3D rendering version).
    NOT WORKING... volumetric rendering issue.
    
    Args:
        res_concat: Array containing predicted flow fields for each block
        y_array: Array containing ground truth flow fields for each block
        x_array: Array containing input flow fields for each block
        sim: Simulation number
        time: Time step number
        save_plots: Whether to save the plots
    """
    if save_plots:
        # plot blocks
        N = res_concat.shape[0]  # Number of blocks

        # Select 9 random indices
        random_indices = np.random.choice(N, size=9, replace=False)

        # Create the figure and axes for a 3x6 grid (3x3 for each side)
        fig, axes = plt.subplots(3, 6, figsize=(18, 12))

        # Add big titles for the left and right 3x3 grids with larger font size
        fig.text(0.25, 0.92, "SM Predictions", ha="center", fontsize=18, fontweight='bold')
        fig.text(0.75, 0.92, "CFD Predictions (Ground Truth)", ha="center", fontsize=18, fontweight='bold')

        for idx, i in enumerate(random_indices):
            row = idx // 3
            col = idx % 3
            
            # Plot SM predictions (left 3x3 grid)
            ax_sm = axes[row, col]
            p_sm = res_concat[i, :, :, :, 0]
            grid_sm = pv.ImageData(dimensions=p_sm.shape)
            grid_sm['Pressure'] = p_sm.flatten(order='F')
            slices_sm = grid_sm.slice_orthogonal()
            pl_sm = pv.Plotter(off_screen=True)
            pl_sm.add_mesh(slices_sm, cmap='viridis', show_edges=False)
            pl_sm.add_title(f"Block {i} - Slices", font_size=18)
            screenshot_sm = pl_sm.screenshot()
            ax_sm.imshow(screenshot_sm)
            ax_sm.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_sm.axis("off")
            
            # Plot CFD predictions (right 3x3 grid)
            ax_cfd = axes[row, col + 3]
            p_cfd = y_array[i, :, :, :, 0]
            grid_cfd = pv.ImageData(dimensions=p_sm.shape)
            grid_cfd['Pressure'] = p_cfd.flatten(order='F')
            slices_cfd = grid_cfd.slice_orthogonal()
            pl_cfd = pv.Plotter(off_screen=True)
            pl_cfd.add_mesh(slices_cfd, cmap='viridis', show_edges=False)
            pl_cfd.add_title(f"Block {i} - Slices", font_size=18)
            screenshot_cfd = pl_cfd.screenshot()
            ax_cfd.imshow(screenshot_cfd)
            ax_cfd.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_cfd.axis("off")

        # Adjust layout to make space for titles
        plt.tight_layout(rect=[0, 0, 1, 0.88])

        # Save the plot as an image file
        output_path = f"plots/sim{sim}/SM_vs_CFD_predictions_t{time}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_random_blocks(res_concat, y_array, x_array, sim, time, save_plots):
    """
    Plot 9 randomly sampled blocks for reference using slices.
    
    Args:
        res_concat: Array containing predicted flow fields for each block
        y_array: Array containing ground truth flow fields for each block
        x_array: Array containing input flow fields for each block
        sim: Simulation number
        time: Time step number
        save_plots: Whether to save the plots
    """
    if save_plots:
        # plot blocks
        N = res_concat.shape[0]  # Number of blocks

        # Select 9 random indices
        random_indices = np.random.choice(N, size=9, replace=False)

        # Create the figure and axes for a 3x6 grid (3x3 for each side)
        fig, axes = plt.subplots(3, 6, figsize=(18, 12))

        # Add big titles for the left and right 3x3 grids with larger font size
        fig.text(0.25, 0.92, "SM Predictions", ha="center", fontsize=18, fontweight='bold')
        fig.text(0.75, 0.92, "CFD Predictions (Ground Truth)", ha="center", fontsize=18, fontweight='bold')

        for idx, i in enumerate(random_indices):
            row = idx // 3
            col = idx % 3
            
            # Plot SM predictions (left 3x3 grid)
            ax_sm = axes[row, col]
            p_sm = res_concat[i, :, :, :, 0]
            grid_sm = pv.ImageData(dimensions=p_sm.shape)
            grid_sm['Pressure'] = p_sm.flatten(order='F')
            slices_sm = grid_sm.slice_orthogonal()
            pl_sm = pv.Plotter(off_screen=True)
            pl_sm.add_mesh(slices_sm, cmap='viridis', show_edges=False)
            pl_sm.add_title(f"Block {i} - Slices", font_size=18)
            screenshot_sm = pl_sm.screenshot()
            ax_sm.imshow(screenshot_sm)
            ax_sm.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_sm.axis("off")
            
            # Plot CFD predictions (right 3x3 grid)
            ax_cfd = axes[row, col + 3]
            p_cfd = y_array[i, :, :, :, 0]
            grid_cfd = pv.ImageData(dimensions=p_sm.shape)
            grid_cfd['Pressure'] = p_cfd.flatten(order='F')
            slices_cfd = grid_cfd.slice_orthogonal()
            pl_cfd = pv.Plotter(off_screen=True)
            pl_cfd.add_mesh(slices_cfd, cmap='viridis', show_edges=False)
            pl_cfd.add_title(f"Block {i} - Slices", font_size=18)
            screenshot_cfd = pl_cfd.screenshot()
            ax_cfd.imshow(screenshot_cfd)
            ax_cfd.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_cfd.axis("off")

        # Adjust layout to make space for titles
        plt.tight_layout(rect=[0, 0, 1, 0.88])

        # Save the plot as an image file
        output_path = f"plots/sim{sim}/SM_vs_CFD_predictions_t{time}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_delta_p_comparison(cfd_results, field_deltap, no_flow_bool, slices_indices=[5, 50, 95], fig_path=None):
    """
    Plot 3D comparison of delta_p predicted, CFD results, and error.
    
    Args:
        cfd_results: CFD results array
        field_deltap: Predicted delta_p field
        no_flow_bool: Boolean mask for no-flow regions
        slices_indices: Indices of slices to plot (as a percentage value)
        fig_path: Optional path to save the figure
    """
    # Mask the error field
    error = np.abs((field_deltap - cfd_results) / (np.max(cfd_results) - np.min(cfd_results)) * 100)
    masked_error = np.where(no_flow_bool, np.nan, error)

    # Masking the predicted delta_p field
    masked_deltap = np.where(no_flow_bool, np.nan, field_deltap)

    # Mask the CFD results
    masked_cfd = np.where(no_flow_bool, np.nan, cfd_results)

    # Create figure and axes
    fig = plt.figure(figsize=(20, 8))
    ax_deltap = fig.add_subplot(131, projection='3d')
    ax_cfd = fig.add_subplot(132, projection='3d')
    ax_error = fig.add_subplot(133, projection='3d')

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(masked_cfd.shape[2]), np.arange(masked_cfd.shape[1]))

    # Set colormap and normalization
    cmap = cm.viridis
    vmin = min(np.nanmin(masked_cfd), np.nanmin(masked_deltap))
    vmax = max(np.nanmax(masked_cfd), np.nanmax(masked_deltap))
    norm = Normalize(vmin=vmin, vmax=vmax)
    error_norm = Normalize(vmin=0, vmax=25)

    slices_list = [int((idx / 100) * masked_deltap.shape[0]) for idx in slices_indices]

    # --- Plot delta_p predicted ---
    for idx in slices_list:
        Z = np.full_like(X, idx)
        alpha = 0.9 if idx == 50 else 0.7  # Adjust transparency
        facecolors = get_facecolors(masked_deltap[idx, :, :], norm, cmap)
        ax_deltap.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, alpha=alpha, linewidth=0, edgecolor='none')

    ax_deltap.set_title("delta_p predicted", fontsize=14, fontweight='bold', pad=5)

    # --- Plot CFD results ---
    for idx in slices_list:
        Z = np.full_like(X, idx)
        alpha = 0.9 if idx == 50 else 0.7
        facecolors = get_facecolors(masked_cfd[idx, :, :], norm, cmap)
        ax_cfd.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, alpha=alpha, linewidth=0, edgecolor='none')

    ax_cfd.set_title("CFD results", fontsize=14, fontweight='bold', pad=5)

    # --- Plot error field ---
    for idx in slices_list:
        Z = np.full_like(X, idx)
        alpha = 0.9 if idx == 50 else 0.7
        facecolors = get_facecolors(masked_error[idx, :, :], error_norm, cmap)
        ax_error.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, alpha=alpha, linewidth=0, edgecolor='none')

    ax_error.set_title("Error (%)", fontsize=14, fontweight='bold', pad=5)

    # Adjust plot appearance
    for ax in [ax_deltap, ax_cfd, ax_error]:
        ax.set_box_aspect([4, 1, 1])
        ax.view_init(elev=30, azim=-90)
        ax.grid(False)
        ax.set_axis_off()

    # Add colorbar for error field
    mappable_error = cm.ScalarMappable(cmap=cmap, norm=error_norm)
    mappable_error.set_array(masked_error)
    cbar = fig.colorbar(mappable_error, ax=ax_error, shrink=0.6, orientation='vertical')
    cbar.set_label("Error (%)", fontsize=12)

    # Reduce extra spacing
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()

    # Show or save the plot
    if fig_path:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()

def plot_delta_p_comparison_volumetric(cfd_results, field_deltap, no_flow_bool, threshold_percentile=10, fig_path=None):
    """
    Plot 3D volumetric comparison of delta_p predicted, CFD results, and error using transparency.
    
    Args:
        cfd_results: CFD results array, shape (Z, Y, X)
        field_deltap: Predicted delta_p field, shape (Z, Y, X)
        no_flow_bool: Boolean mask for no-flow regions
        threshold_percentile: Percentile threshold for transparency (0-100). 
                            Values below this percentile will be more transparent.
        fig_path: Optional path to save the figure
    """
    # Mask the error field
    error = np.abs((field_deltap - cfd_results) / (np.max(cfd_results) - np.min(cfd_results)) * 100)
    masked_error = np.where(no_flow_bool, np.nan, error)

    # Masking the predicted delta_p field
    masked_deltap = np.where(no_flow_bool, np.nan, field_deltap)

    # Mask the CFD results
    masked_cfd = np.where(no_flow_bool, np.nan, cfd_results)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(24, 8))
    ax_deltap = fig.add_subplot(131, projection='3d')
    ax_cfd = fig.add_subplot(132, projection='3d')
    ax_error = fig.add_subplot(133, projection='3d')

    # Set colormap and normalization
    cmap = cm.viridis
    vmin = min(np.nanmin(masked_cfd), np.nanmin(masked_deltap))
    vmax = max(np.nanmax(masked_cfd), np.nanmax(masked_deltap))
    norm = Normalize(vmin=vmin, vmax=vmax)
    error_norm = Normalize(vmin=0, vmax=25)

    # Helper function to create volumetric plot with transparency
    def plot_volumetric(ax, data, norm, cmap, title):
        # Get all voxel coordinates where data is not NaN
        z_coords, y_coords, x_coords = np.where(~np.isnan(data))
        values = data[z_coords, y_coords, x_coords]
        
        # Normalize values for coloring
        colors = cmap(norm(values))
        
        # Calculate alpha based on value magnitude
        threshold = np.nanpercentile(np.abs(values), threshold_percentile)
        alphas = np.clip((np.abs(values) - threshold) / (np.nanmax(np.abs(values)) - threshold), 0.1, 0.9)
        colors[:, 3] = alphas  # Set alpha channel
        
        # Plot voxels
        ax.scatter(x_coords, y_coords, z_coords, c=colors, marker='s', 
                  s=20, edgecolors='none', depthshade=True)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([data.shape[2], data.shape[1], data.shape[0]])
        ax.view_init(elev=20, azim=-60)
        ax.grid(True, alpha=0.3)
    
    # Plot delta_p predicted
    plot_volumetric(ax_deltap, masked_deltap, norm, cmap, "delta_p predicted")
    
    # Plot CFD results
    plot_volumetric(ax_cfd, masked_cfd, norm, cmap, "CFD results")
    
    # Plot error field
    plot_volumetric(ax_error, masked_error, error_norm, cmap, "Error (%)")
    
    # Add colorbar for error field
    mappable_error = cm.ScalarMappable(cmap=cmap, norm=error_norm)
    mappable_error.set_array(masked_error)
    cbar = fig.colorbar(mappable_error, ax=ax_error, shrink=0.6, orientation='vertical')
    cbar.set_label("Error (%)", fontsize=12)
    
    plt.tight_layout()
    
    # Show or save the plot
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_cfd_results_3d_helper(cfd_results, no_flow_bool, slices_indices=[25, 50, 75], fig_path=None, alpha_boundary=0.4):
    """
    Plot multiple slices of the CFD results field in 3D.
    
    Args:
        cfd_results: CFD results array, shape (Z, Y, X)
        no_flow_bool: Boolean mask for no-flow regions
        slices_indices: Indices of slices to plot. If None, plot every 10th slice
        fig_path: Path to save the figure. If None, the plot is displayed
        alpha_boundary: Alpha transparency for boundary slices
    """
    # Mask the CFD results
    masked_cfd = np.where(no_flow_bool, np.nan, cfd_results)

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(masked_cfd.shape[2]), np.arange(masked_cfd.shape[1]))

    # Set colormap and normalization
    cmap = cm.RdYlBu
    vmin = np.nanmin(masked_cfd)
    vmax = np.nanmax(masked_cfd)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Create figure and axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    slices_list = [int((idx / 100) * masked_cfd.shape[0]) for idx in slices_indices]

    # Plot CFD results slices
    for idx in slices_list:
        Z = np.full_like(X, idx)
        alpha = 1 if (idx == slices_list[0] or idx == slices_list[-1]) else 0.5
        facecolors = cmap(norm(masked_cfd[idx, :, :]))
        facecolors[..., 3] = alpha
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, edgecolor='none', antialiased=False)

    # Plot boundary slices with transparency
    # X=0 and X=-1 (YZ planes)
    X0 = np.zeros_like(masked_cfd[:, :, 0])
    XN = np.full_like(masked_cfd[:, :, -1], masked_cfd.shape[2] - 1)
    facecolors_x0 = cmap(norm(masked_cfd[:, :, 0]))
    facecolors_x0[..., 3] = alpha_boundary
    ax.plot_surface(X0, np.arange(masked_cfd.shape[1])[:, None], np.arange(masked_cfd.shape[0])[None, :], 
                    facecolors=facecolors_x0.transpose(1, 0, 2), rstride=1, cstride=1, linewidth=0, edgecolor='none', antialiased=False)
    facecolors_xn = cmap(norm(masked_cfd[:, :, -1]))
    facecolors_xn[..., 3] = alpha_boundary
    ax.plot_surface(XN, np.arange(masked_cfd.shape[1])[:, None], np.arange(masked_cfd.shape[0])[None, :], 
                    facecolors=facecolors_xn.transpose(1, 0, 2), rstride=1, cstride=1, linewidth=0, edgecolor='none', antialiased=False)

    # Y=0 and Y=-1 (XZ planes)
    Y0 = np.zeros_like(masked_cfd[:, 0, :])
    YN = np.full_like(masked_cfd[:, -1, :], masked_cfd.shape[1] - 1)
    facecolors_y0 = cmap(norm(masked_cfd[:, 0, :]))
    facecolors_y0[..., 3] = alpha_boundary
    ax.plot_surface(np.arange(masked_cfd.shape[2])[None, :], Y0, np.arange(masked_cfd.shape[0])[:, None], 
                    facecolors=facecolors_y0, rstride=1, cstride=1, linewidth=0, edgecolor='none', antialiased=False)
    facecolors_yn = cmap(norm(masked_cfd[:, -1, :]))
    facecolors_yn[..., 3] = alpha_boundary
    ax.plot_surface(np.arange(masked_cfd.shape[2])[None, :], YN, np.arange(masked_cfd.shape[0])[:, None], 
                    facecolors=facecolors_yn, rstride=1, cstride=1, linewidth=0, edgecolor='none', antialiased=False)

    ax.set_box_aspect([masked_cfd.shape[2], masked_cfd.shape[1], masked_cfd.shape[0]])
    ax.view_init(elev=20, azim=200)
    ax.grid(False)
    ax.set_axis_off()
    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)
    else:
        plt.show()


def plot_inputs_slices(ux, uy, uz, sdf, deltap, slices_indices=[5, 50, 95], fig_path=None):
    """
    Plot slices of the input fields: ux, uy, uz, SDF, and delta_p.
    
    Args:
        ux: Velocity field in the x-direction
        uy: Velocity field in the y-direction
        uz: Velocity field in the z-direction
        sdf: Signed distance field
        deltap: Delta pressure field
        slices_indices: Indices of slices to plot
        fig_path: Optional path to save the figure
    """

    slices_list = [int((idx / 100) * ux.shape[0]) for idx in slices_indices]

    # Create the figure and axes with a wide aspect ratio and reduced height
    fig_height = 2.5 * len(slices_list)
    fig_width = 18  # Wide figure
    fig, axs = plt.subplots(len(slices_list), 5, figsize=(fig_width, fig_height), constrained_layout=True)

    # Set the column titles
    for col, title in enumerate(["ux", "uy", "uz", "SDF", "output - deltaP"]):
        axs[0, col].set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Plot the slices for each field
    for i, idx in enumerate(slices_list):
        for j, field in enumerate([ux, uy, uz, sdf, deltap]):
            im = axs[i, j].imshow(field[idx, :, :], cmap='viridis', origin='lower', aspect='auto')
            axs[i, j].set_title(f"Slice {idx}/100", fontsize=11, fontweight='bold', loc='left')
            axs[i, j].axis("off")
            # Optionally add colorbar only for the first row
            if i == 0:
                plt.colorbar(im, ax=axs[i, j], fraction=0.03, pad=0.02)

    # Adjust layout for better spacing
    plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)

    # Show or save the plot
    if fig_path:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()


def plot_delta_p_comparison_slices(cfd_results, field_deltap, no_flow_bool, slices_indices=[5, 50, 95], fig_path=None):
    """
    Plot comparison of delta_p predicted, CFD results, and error using 2D slices.
    
    Args:
        cfd_results: CFD results array
        field_deltap: Predicted delta_p field
        no_flow_bool: Boolean mask for no-flow regions
        slices_indices: Indices of slices to plot
        fig_path: Optional path to save the figure
    """
    # Mask the error field
    error = np.abs((field_deltap - cfd_results) / (np.max(cfd_results) - np.min(cfd_results)) * 100)
    masked_error = np.where(no_flow_bool, np.nan, error)

    # Masking the predicted delta_p field
    masked_deltap = np.where(no_flow_bool, np.nan, field_deltap)

    # Mask the CFD results
    masked_cfd = np.where(no_flow_bool, np.nan, cfd_results)

    # Set colormap and normalization
    cmap = cm.viridis
    vmin = min(np.nanmin(masked_cfd), np.nanmin(masked_deltap))
    vmax = max(np.nanmax(masked_cfd), np.nanmax(masked_deltap))
    norm = Normalize(vmin=vmin, vmax=vmax)
    error_norm = Normalize(vmin=0, vmax=25)

    slices_list = [int((idx / 100) * masked_deltap.shape[0]) for idx in slices_indices]

    # Create figure and axes with improved aspect ratio
    fig_height = 2.5 * len(slices_list)
    fig_width = 45  # Wide figure for better aspect
    fig, axs = plt.subplots(len(slices_list), 3, figsize=(fig_width, fig_height))
    total_slices = masked_deltap.shape[0]

    # Plot slices
    for i, idx in enumerate(slices_list):
        # Plot delta_p predicted
        axs[i, 0].imshow(masked_deltap[idx, :, :], cmap=cmap, norm=norm, origin='lower', aspect='auto')
        axs[i, 0].set_title(f"delta_p predicted (Slice {idx}/{total_slices})", fontsize=30, fontweight='bold')
        axs[i, 0].axis("off")

        # Plot CFD results
        axs[i, 1].imshow(masked_cfd[idx, :, :], cmap=cmap, norm=norm, origin='lower', aspect='auto')
        axs[i, 1].set_title(f"CFD results (Slice {idx}/{total_slices})", fontsize=30, fontweight='bold')
        axs[i, 1].axis("off")

        # Plot error field
        axs[i, 2].imshow(masked_error[idx, :, :], cmap=cmap, norm=error_norm, origin='lower', aspect='auto')
        axs[i, 2].set_title(f"Error (%) (Slice {idx}/{total_slices})", fontsize=30, fontweight='bold')
        axs[i, 2].axis("off")

    # Adjust layout for better spacing
    plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)

    # Show or save the plot
    if fig_path:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()
