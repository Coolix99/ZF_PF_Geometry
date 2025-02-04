import os
import shutil
import pyvista as pv
import napari
import numpy as np
import logging
from datetime import datetime
from simple_file_checksum import get_checksum
from zf_pf_geometry.metadata_manager import get_JSON, write_JSON
from zf_pf_geometry.utils import load_tif_image

def key_callback(key,logger,plotter,surface_file_path,data_name):
    logger.info(f"Key pressed: {key}")
    if key in {'d', 'r'}:
        logger.info(f"Deleting surface: {surface_file_path}")
        try:
            shutil.rmtree(os.path.dirname(surface_file_path))
        except Exception as e:
            logger.error(f"Failed to delete {os.path.dirname(surface_file_path)}: {e}")
    if key == 'r':
        logger.info(f"Additional cleanup for {data_name}.")
    plotter.close()
    return False

def plot_surface_pyvista(mesh, logger, surface_file_path,data_name):
    """
    Plots a surface mesh using PyVista and handles deletion via key events.

    Args:
        surface_file_path (str): Path to the surface file.
        data_name (str): Name of the data.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        None
    """
    
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='thickness' if 'thickness' in mesh.point_data else None)

    plotter.add_key_event('d', lambda: key_callback('d',logger,plotter,surface_file_path,data_name))
    plotter.add_key_event('r', lambda: key_callback('r',logger,plotter,surface_file_path,data_name))
    plotter.add_key_event('q', plotter.close)

    try:
        plotter.show()
    except Exception as e:
        logger.error(f"Error displaying plot for {data_name}: {e}")

def plot_surface_napari(mesh, mask, logger, scale,surface_file_path,data_name):
    """
    Plots a mask and mesh in Napari.

    Args:
        mesh (pv.PolyData): The 3D surface mesh.
        mask (np.ndarray): The corresponding mask image.
        logger (logging.Logger): Logger instance for logging messages.
        scale (list or tuple): Scaling factor for visualization.

    Returns:
        None
    """

    logger.info(f"Loaded mask with shape {mask.shape}.")
    viewer = napari.Viewer(ndisplay=3)
    
    # Add the mask
    viewer.add_labels(mask, scale=scale)

    # Convert PyVista mesh to numpy format for Napari (z, y, x)
    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Convert from PyVista face format
    logger.info(f"Converted mesh with {len(vertices)} vertices and {len(faces)} faces.")

    # Add the mesh as a surface layer in Napari
    viewer.add_surface((vertices, faces), colormap="gray")


    # Bind key events
    @viewer.bind_key('d')
    def delete_only(viewer):
        key_callback('d',logger,viewer,surface_file_path,data_name)

    @viewer.bind_key('r')
    def delete_and_cleanup(viewer):
        key_callback('r',logger,viewer,surface_file_path,data_name)

    @viewer.bind_key('q')
    def close_viewer(viewer):
        viewer.close()

    napari.run()

def plot_all_surfaces(surface_dir, state="surface", skip_shown=True, mask_dir=None):
    """
    Processes and plots surfaces in the given directory based on their state and metadata.
    If a mask directory is provided, surfaces are shown in Napari instead of PyVista.

    Args:
        surface_dir (str): Path to the directory containing surface data.
        state (str): Desired state of the surface ('surface', 'coord', or 'thickness').
        skip_shown (bool): Whether to skip surfaces that have already been shown if their checksum hasn't changed.
        mask_dir (str or None): Path to the directory containing mask files.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    surface_folder_list = [
        item for item in os.listdir(surface_dir) if os.path.isdir(os.path.join(surface_dir, item))
    ]
    logger.info(f"Found {len(surface_folder_list)} surface folders for processing.")

    for data_name in surface_folder_list:
        folder_path = os.path.join(surface_dir, data_name)

        # Load metadata
        metadata = get_JSON(folder_path)
        if not metadata or state not in metadata:
            logger.warning(f"Skipping {data_name}: Missing metadata or state '{state}' not found.")
            continue

        # Load surface file
        surface_file_key = f"{state.capitalize()} file name"
        if surface_file_key not in metadata[state]:
            logger.warning(f"Skipping {data_name}: Surface file key '{surface_file_key}' missing in metadata.")
            continue

        surface_file_path = os.path.join(folder_path, metadata[state][surface_file_key])

        if not os.path.exists(surface_file_path):
            logger.warning(f"Skipping {data_name}: Surface file '{surface_file_path}' not found.")
            continue
        mesh = pv.read(surface_file_path)
        logger.info(f"Loaded surface mesh with {mesh.n_points} points and {mesh.n_cells} cells.")

        # Compute the checksum of the surface file
        current_checksum = get_checksum(surface_file_path, algorithm="SHA1")

        # Check if the surface was already shown and unchanged
        state_shown = metadata.get("state_shown", {})
        if skip_shown and state in state_shown:
            shown_state = state_shown[state]
            if shown_state.get("checksum") == current_checksum:
                logger.info(
                    f"Skipping {data_name}: Already shown on {shown_state['last_shown']} with an unchanged checksum."
                )
                continue

        # Determine whether to plot in PyVista or Napari
        if mask_dir:
            mask_image = load_tif_image(os.path.join(mask_dir, data_name))

            logger.info(f"Plotting mask for {data_name} in Napari.")
            plot_surface_napari(mesh,mask_image, logger,metadata[state]['scale'],surface_file_path,data_name)
        else:
            logger.info(f"Plotting surface for {data_name} in PyVista.")
            plot_surface_pyvista(mesh, logger,surface_file_path,data_name)

        # Update metadata with state_shown information
        metadata.setdefault("state_shown", {})[state] = {
            "last_shown": datetime.now().isoformat(),
            "checksum": current_checksum,
        }
        write_JSON(folder_path, "state_shown", metadata["state_shown"])
        logger.info(f"Updated metadata for {data_name} with last shown time and checksum.")
