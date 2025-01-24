import os
import git
from simple_file_checksum import get_checksum
import pandas as pd
import numpy as np
import pyvista as pv
from tqdm import tqdm
import logging

from zf_pf_geometry.metadata_manager import should_process, write_JSON
from zf_pf_geometry.orientation import orientation_session
from zf_pf_geometry.center_line import center_line_session
from zf_pf_geometry.surface import construct_Surface
from zf_pf_geometry.coord import create_coord_system, calculateCurvatureTensor
from zf_pf_geometry.thickness import calculate_Thickness
from zf_pf_geometry.utils import make_path
from zf_pf_geometry.image_operations import get_Image


def setup_folders(base_dirs, data_name):
    """
    Constructs folder paths for input and output and ensures the output folder exists.

    Args:
        base_dirs (dict): Dictionary with base paths for surface, mask, etc.
        data_name (str): Name of the dataset being processed.

    Returns:
        dict: Paths for each key in base_dirs, plus the output folder.
    """
    paths = {key: os.path.join(base_dir, data_name) for key, base_dir in base_dirs.items()}
    make_path(paths["output"])
    return paths

def update_metadata(metadata, file_paths, input_checksum=None):
    """
    Updates metadata with checksums, Git information, and file paths.

    Args:
        metadata (dict): Metadata to update.
        file_paths (dict): File paths with their keys (e.g., surface, orientation).
        input_checksum (str): Optional input data checksum.

    Returns:
        dict: Updated metadata.
    """
    repo = git.Repo('.', search_parent_directories=True)
    metadata["git hash"] = repo.head.object.hexsha
    metadata["git origin url"] = repo.remotes.origin.url

    if input_checksum:
        metadata["input_data_checksum"] = input_checksum

    for key, path in file_paths.items():
        if os.path.exists(path):
            metadata[f"{key} file name"] = os.path.basename(path)
            metadata[f"{key} checksum"] = get_checksum(path, algorithm="SHA1")
    return metadata

def load_tif_image(folder):
    """
    Loads a single `.tif` image from the specified folder.

    Args:
        folder (str): Path to the folder containing `.tif` files.

    Returns:
        np.ndarray: Loaded image.
    """
    img_list = [item for item in os.listdir(folder) if item.endswith('.tif')]
    if len(img_list) != 1:
        raise ValueError(f"Expected 1 .tif file, found {len(img_list)} in {folder}")
    return get_Image(os.path.join(folder, img_list[0]))

def do_orientation(input_dirs, key_0, output_dir):
    """
    Processes orientation from input directories and saves the results to the output directory.
    Args:
        input_dirs (list of str): List of input directories containing image data.
        key_0 (str): Key to access specific data in the input metadata.
        output_dir (str): Directory where the output data will be saved.
    """
    logger = logging.getLogger(__name__)
    output_key = 'orientation'
    base_dirs = {"input_0": input_dirs[0], "output": output_dir}

    img_0_folder_list = [item for item in os.listdir(base_dirs["input_0"]) if os.path.isdir(os.path.join(base_dirs["input_0"], item))]
    
    # Add a progress bar
    logger.info(f"Processing {len(img_0_folder_list)} items...")
    for data_name in tqdm(img_0_folder_list, desc="Processing orientation", unit="dataset"):
        logger.info(f"Processing {data_name}")

        # Setup folders
        paths = setup_folders(base_dirs, data_name)

        # Check if processing is needed
        res = should_process([paths["input_0"]], [key_0], paths["output"], output_key, verbose=False)
        if not res:
            logger.info(f"Skipping {data_name}: No processing needed.")
            continue
        input_data, input_checksum = res

        # Gather image folders
        img_folder_list = [paths["input_0"]]
        for i in range(1, len(input_dirs)):
            img_folder_list.append(os.path.join(input_dirs[i], data_name))

        # Load images
        images = []
        for img_folder in img_folder_list:
            try:
                images.append(load_tif_image(img_folder))
            except ValueError as e:
                logger.error(f"Error loading images from {img_folder}: {e}")
                continue

        # Process orientation
        df, fin_side = orientation_session(images, input_data[key_0]['scale'])

        # Save results
        file_name = 'orientation.csv'
        df_csv_file = os.path.join(paths["output"], file_name)
        df.to_csv(df_csv_file, index=False)

        # Update metadata and save
        file_paths = {"df": df_csv_file}
        res_MetaData = update_metadata(input_data[key_0], file_paths, input_checksum)
        if fin_side is not None:
            res_MetaData['fin_side'] = fin_side
        write_JSON(paths["output"], output_key, res_MetaData)

def do_center_line(orientation_dir, mask_dir, mask_key, output_dir):
    """
    Processes center lines based on orientation data and mask images, saving results to the output directory.
    
    Args:
        orientation_dir (str): Directory containing orientation data folders.
        mask_dir (str): Directory containing mask image folders.
        mask_key (str): Key to access specific mask data in the metadata.
        output_dir (str): Directory where the processed center line data will be saved.
    """
    logger = logging.getLogger(__name__)
    output_key = 'center line'

    # Gather orientation folders
    orientation_folder_list = [
        item for item in os.listdir(orientation_dir) if os.path.isdir(os.path.join(orientation_dir, item))
    ]
    logger.info(f"Found {len(orientation_folder_list)} orientation folders for processing.")

    # Add a progress bar
    for data_name in tqdm(orientation_folder_list, desc="Processing center lines", unit="dataset"):
        logger.info(f"Processing {data_name}")

        # Set up folder paths
        orientation_folder = os.path.join(orientation_dir, data_name)
        mask_folder = os.path.join(mask_dir, data_name)
        output_folder = os.path.join(output_dir, data_name)
        make_path(output_folder)

        # Check if processing is needed
        res = should_process(
            [orientation_folder, mask_folder], ['orientation', mask_key], output_folder, output_key, verbose=True
        )
        if not res:
            logger.info(f"Skipping {data_name}: No processing needed.")
            continue
        input_data, input_checksum = res

        # Load orientation data
        orientation_file = os.path.join(orientation_folder, input_data['orientation']['df file name'])
        orientation_df = pd.read_csv(orientation_file)

        # Load mask image
        img_list = [item for item in os.listdir(mask_folder) if item.endswith('.tif')]
        if len(img_list) == 1:
            mask_image = get_Image(os.path.join(mask_folder, img_list[0]))
        else:
            logger.error(f"Expected one mask image in {mask_folder}, found {len(img_list)}. Skipping {data_name}.")
            continue

        # Process center line
        logger.info(f"Computing center line for {data_name}.")
        center_line_path_3d = center_line_session(
            mask_image, orientation_df, input_data['orientation']['scale']
        )

        # Save center line data
        CenterLine_file_name = f"{data_name}_CenterLine.npy"
        CenterLine_file = os.path.join(output_folder, CenterLine_file_name)
        np.save(CenterLine_file, center_line_path_3d)

        # Update metadata and save
        logger.info(f"Saving metadata for {data_name}.")
        res_MetaData = input_data['orientation']
        res_MetaData['CenterLine file name'] = CenterLine_file_name
        res_MetaData['CenterLine checksum'] = get_checksum(CenterLine_file, algorithm="SHA1")
        res_MetaData['input_data_checksum'] = input_checksum

        # Get git information
        try:
            repo = git.Repo('.', search_parent_directories=True)
            res_MetaData['git hash'] = repo.head.object.hexsha
            res_MetaData['git origin url'] = repo.remotes.origin.url
        except Exception as e:
            logger.warning(f"Git information could not be retrieved: {e}")

        write_JSON(output_folder, output_key, res_MetaData)

    logger.info("Center line processing completed.")
    return

def do_surface(orientation_dir, center_line_dir, mask_dir, mask_key, output_dir):
    """
    Processes surface data by combining orientation, center line, and mask data,
    and saves the resulting surface and rip data to the output directory.

    Args:
        orientation_dir (str): Directory containing orientation data.
        center_line_dir (str): Directory containing center line data.
        mask_dir (str): Directory containing mask image data.
        mask_key (str): Key to access specific mask data in the metadata.
        output_dir (str): Directory where the processed surface data will be saved.
    """
    logger = logging.getLogger(__name__)
    output_key = "surface"
    base_dirs = {
        "orientation": orientation_dir,
        "center_line": center_line_dir,
        "mask": mask_dir,
        "output": output_dir
    }

    # Gather center line folders
    center_line_folder_list = [
        item for item in os.listdir(center_line_dir) if os.path.isdir(os.path.join(center_line_dir, item))
    ]
    logger.info(f"Found {len(center_line_folder_list)} center line folders for processing.")

    for data_name in tqdm(center_line_folder_list, desc="Processing surfaces", unit="dataset"):
        logger.info(f"Processing dataset: {data_name}")

        # Setup folders
        paths = setup_folders(base_dirs, data_name)

        # Check if processing is needed
        res = should_process(
            [paths["center_line"], paths["orientation"], paths["mask"]],
            ["center line", "orientation", mask_key],
            paths["output"],
            output_key
        )
        if not res:
            logger.info(f"Skipping {data_name}: No processing needed.")
            continue

        input_data, input_checksum = res

        try:
            # Load orientation, center line, and mask data
            orientation_file = os.path.join(paths["orientation"], input_data["orientation"]["df file name"])
            orientation_df = pd.read_csv(orientation_file)
            logger.info(f"Loaded orientation data from {orientation_file}.")

            center_line_file = os.path.join(paths["center_line"], input_data["center line"]["CenterLine file name"])
            center_line_path_3d = np.load(center_line_file)
            logger.info(f"Loaded center line data from {center_line_file}.")

            mask_image = load_tif_image(paths["mask"])
            logger.info(f"Loaded mask image from {paths['mask']}.")

            # Process surface
            logger.info(f"Constructing surface for {data_name}.")
            smooth_surface, rip_df = construct_Surface(
                mask_image, orientation_df, center_line_path_3d, input_data["orientation"]["scale"]
            )

            # Save rip data
            rip_file = os.path.join(paths["output"], f"{data_name}_rip.csv")
            rip_df.to_csv(rip_file, index=False)
            logger.info(f"Saved rip data to {rip_file}.")

            # Save surface data
            surface_file = os.path.join(paths["output"], f"{data_name}_surface.vtk")
            smooth_surface.save(surface_file)
            logger.info(f"Saved surface data to {surface_file}.")

            # Update and write metadata
            file_paths = {"Surface": surface_file, "Rip": rip_file}
            res_MetaData = update_metadata(input_data["center line"], file_paths, input_checksum)
            write_JSON(paths["output"], output_key, res_MetaData)
            logger.info(f"Updated and saved metadata for {data_name}.")

        except Exception as e:
            logger.error(f"Error processing {data_name}: {e}")
            continue

    logger.info("Surface processing completed.")

def do_coord(orientation_dir, surface_dir, output_dir):
    """
    Processes surface and orientation data to create a coordinate system and saves the results.
    
    Args:
        orientation_dir (str): Path to the directory containing orientation data.
        surface_dir (str): Path to the directory containing surface data.
        output_dir (str): Path to save output data.
    """
    logger = logging.getLogger(__name__)
    output_key = 'coord'
    base_dirs = {"surface": surface_dir, "orientation": orientation_dir, "output": output_dir}

    # Gather surface folders
    surface_folder_list = [
        item for item in os.listdir(surface_dir) if os.path.isdir(os.path.join(surface_dir, item))
    ]
    logger.info(f"Found {len(surface_folder_list)} surface folders for processing.")

    # Add progress bar
    for data_name in tqdm(surface_folder_list, desc="Processing coordinates", unit="dataset"):
        logger.info(f"Processing dataset: {data_name}")

        # Setup folders
        paths = setup_folders(base_dirs, data_name)

        # Check if processing is needed
        res = should_process([paths["surface"], paths["orientation"]],
                             ['surface', 'orientation'], paths["output"], output_key)
        if not res:
            logger.info(f"Skipping {data_name}: No processing needed.")
            continue

        input_data, input_checksum = res


        # Load input files
        orientation_file = os.path.join(paths["orientation"], input_data["orientation"]["df file name"])
        orientation_df = pd.read_csv(orientation_file)
        logger.info(f"Loaded orientation data from {orientation_file}.")

        surface_file = os.path.join(paths["surface"], input_data["surface"]["Surface file name"])
        mesh = pv.read(surface_file)
        logger.info(f"Loaded surface data from {surface_file}.")

        rip_file = os.path.join(paths["surface"], input_data["surface"]["Rip file name"])
        rip_df = pd.read_csv(rip_file)
        logger.info(f"Loaded rip data from {rip_file}.")

        # Process mesh to create coordinate system
        logger.info(f"Creating coordinate system for {data_name}.")
        mesh = create_coord_system(mesh, orientation_df, rip_df, input_data["surface"]["scale"])
        if mesh is None:
            logger.error(f"Failed to create coordinate system for {data_name}. Skipping.")
            continue

        # Calculate curvature tensor
        logger.info(f"Calculating curvature tensor for {data_name}.")
        res = calculateCurvatureTensor(mesh)
        if res is None:
            logger.error(f"Failed to calculate curvature tensor for {data_name}. Skipping.")
            continue

        # Save the resulting mesh
        surface_file_name = f"{data_name}_coord.vtk"
        surface_file_path = os.path.join(paths["output"], surface_file_name)
        mesh.save(surface_file_path)
        logger.info(f"Saved coordinate mesh to {surface_file_path}.")

        # Update metadata and save
        file_paths = {"Surface(Coord)": surface_file_path}
        res_MetaData = update_metadata(input_data["surface"], file_paths, input_checksum)
        write_JSON(paths["output"], output_key, res_MetaData)
        logger.info(f"Updated and saved metadata for {data_name}.")

       

    logger.info("Coordinate processing completed.")

def do_thickness(coord_dir, mask_dir, mask_key, output_dir):
    """
    Processes coordinate and mask data to calculate thickness and saves the results.
    
    Args:
        coord_dir (str): Path to the directory containing coordinate data.
        mask_dir (str): Path to the directory containing mask data.
        mask_key (str): Key for accessing mask metadata.
        output_dir (str): Path to save output data.
    """
    logger = logging.getLogger(__name__)
    output_key = 'thickness'
    base_dirs = {"coord": coord_dir, "mask": mask_dir, "output": output_dir}

    # Gather coordinate folders
    coord_folder_list = [
        item for item in os.listdir(coord_dir) if os.path.isdir(os.path.join(coord_dir, item))
    ]
    logger.info(f"Found {len(coord_folder_list)} coordinate folders for processing.")

    # Add progress bar
    for data_name in tqdm(coord_folder_list, desc="Processing thickness", unit="dataset"):
        logger.info(f"Processing dataset: {data_name}")

        # Setup folders
        paths = setup_folders(base_dirs, data_name)

        # Check if processing is needed
        res = should_process([paths["coord"], paths["mask"]],
                             ['coord', mask_key], paths["output"], output_key)
        if not res:
            logger.info(f"Skipping {data_name}: No processing needed.")
            continue

        input_data, input_checksum = res

        
        # Load input files
        surface_file = os.path.join(paths["coord"], input_data["coord"]["Surface(Coord) file name"])
        mesh = pv.read(surface_file)
        logger.info(f"Loaded surface data from {surface_file}.")

        mask_image = load_tif_image(paths["mask"])
        logger.info(f"Loaded mask image from {paths['mask']}.")

        # Process thickness
        logger.info(f"Calculating thickness for {data_name}.")
        mesh = calculate_Thickness(mask_image, mesh, input_data["coord"]["scale"])

        # Save the resulting mesh
        surface_file_name = f"{data_name}_thickness.vtk"
        surface_file_path = os.path.join(paths["output"], surface_file_name)
        mesh.save(surface_file_path)
        logger.info(f"Saved thickness mesh to {surface_file_path}.")

        # Update metadata and save
        file_paths = {"Surface(Thickness)": surface_file_path}
        res_MetaData = update_metadata(input_data["coord"], file_paths, input_checksum)
        write_JSON(paths["output"], output_key, res_MetaData)
        logger.info(f"Updated and saved metadata for {data_name}.")

        

    logger.info("Thickness processing completed.")

