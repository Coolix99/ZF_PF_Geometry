import os
import shutil
import pyvista as pv
import logging
from datetime import datetime
from simple_file_checksum import get_checksum
from zf_pf_geometry.metadata_manager import get_JSON, write_JSON

def plot_all_surfaces(surface_dir, state="surface", skip_shown=True):
    """
    Processes and plots surfaces in the given directory based on their state and metadata.
    Records when a surface was shown under a `state_shown` key and optionally skips already shown surfaces
    if their checksum hasn't changed.

    Args:
        surface_dir (str): Path to the directory containing surface data.
        state (str): Desired state of the surface ('surface', 'coord', or 'thickness').
        skip_shown (bool): Whether to skip surfaces that have already been shown if their checksum hasn't changed.

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

        surface_file_name = metadata[state][surface_file_key]
        surface_file_path = os.path.join(folder_path, surface_file_name)

        if not os.path.exists(surface_file_path):
            logger.warning(f"Skipping {data_name}: Surface file '{surface_file_path}' not found.")
            continue

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

        logger.info(f"Plotting surface for {data_name} from file: {surface_file_path}")

        # Load the surface mesh
        mesh = pv.read(surface_file_path)
        logger.info(f"Loaded surface mesh with {mesh.n_points} points and {mesh.n_cells} cells.")

        # Create the plotter
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars='thickness' if 'thickness' in mesh.point_data else None)

        # Define a callback function for interactive deletion
        def key_callback(key):
            logger.info(f"Key pressed: {key}")
            if key in {'d', 'r'}:
                logger.info(f"Deleting surface: {surface_file_path}")
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    logger.error(f"Failed to delete {folder_path}: {e}")
            if key == 'r':
                logger.info(f"Additional cleanup for {data_name}.")
                # Implement additional cleanup logic if needed
            plotter.close()
            return False

        # Bind the key callbacks
        plotter.add_key_event('d', lambda: key_callback('d'))  # Delete only the surface
        plotter.add_key_event('r', lambda: key_callback('r'))  # Delete surface and associated data
        plotter.add_key_event('q', plotter.close)             # Quit without deleting

        # Show the plot
        try:
            plotter.show()
        except Exception as e:
            logger.error(f"Error displaying plot for {data_name}: {e}")
            continue

        # Update metadata with state_shown information
        if "state_shown" not in metadata:
            metadata["state_shown"] = {}
        metadata["state_shown"][state] = {
            "last_shown": datetime.now().isoformat(),
            "checksum": current_checksum,
        }
        write_JSON(folder_path, "state_shown", metadata["state_shown"])
        logger.info(f"Updated metadata for {data_name} with last shown time and checksum.")
