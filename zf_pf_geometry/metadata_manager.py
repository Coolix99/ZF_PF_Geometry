import os
import json
import hashlib
import logging
from typing import List, Tuple, Union

# Initialize the logger
logger = logging.getLogger(__name__)

def get_JSON(dir: str, name: str = None) -> Union[dict, None]:
    """
    Reads a JSON file from the specified directory.
    Args:
        dir (str): Directory containing the JSON file.
        name (str, optional): Name of the JSON file (default is 'MetaData.json').

    Returns:
        dict or None: Parsed JSON data as a dictionary, or None if file is not found.
    """
    name = 'MetaData.json' if name is None else f'MetaData_{name}.json'
    json_file_path = os.path.join(dir, name)

    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        logger.warning(f"{json_file_path} doesn't exist.")
        return None

def write_JSON(dir: str, key: str, value: dict, name: str = None):
    """
    Writes a JSON file to the specified directory, updating the given key with a new value.

    Args:
        dir (str): Directory to save the JSON file.
        key (str): Key to update in the JSON data.
        value (dict): Value to associate with the key.
        name (str, optional): Name of the JSON file (default is 'MetaData.json').
    """
    name = 'MetaData.json' if name is None else f'MetaData_{name}.json'
    json_file_path = os.path.join(dir, name)

    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        logger.info(f"JSON file not found at {json_file_path}. Creating a new one.")
        data = {}  # Create an empty dictionary if the file doesn't exist

    data[key] = value
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    logger.info(f"Updated JSON file at {json_file_path} with key: {key}")

def calculate_dict_checksum(data: dict) -> str:
    """
    Calculate a checksum (SHA256) of a dictionary.

    Args:
        data (dict): The dictionary to calculate the checksum for.

    Returns:
        str: The SHA256 checksum as a hexadecimal string.
    """
    serialized_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
    checksum = hashlib.sha256(serialized_data.encode('utf-8')).hexdigest()
    return checksum

def should_process(input_dirs: List[str], input_keys: List[str], output_dir: str, output_key: str, verbose: bool = False) -> Union[Tuple[dict, str], bool]:
    """
    Determines whether processing is required based on the state of input and output metadata.

    Args:
        input_dirs (List[str]): List of input directories.
        input_keys (List[str]): Corresponding keys in the input metadata.
        output_dir (str): Directory containing the output metadata.
        output_key (str): Key to check in the output metadata.
        verbose (bool, optional): Enables additional logging.

    Returns:
        Tuple[dict, str] or bool: Input metadata and checksum if processing is needed, or False otherwise.
    """
    if len(input_dirs) != len(input_keys):
        logger.error("The number of input directories and keys should be the same.")
        raise ValueError("The number of input directories and keys should be the same.")

    input_data = {}
    for dir, key in zip(input_dirs, input_keys):
        data = get_JSON(dir)
        if data is None:
            logger.warning(f"Skipping processing because the JSON file in {dir} doesn't exist.")
            return False
        if key not in data:
            logger.warning(f"Skipping processing because the key {key} doesn't exist in the JSON file in {dir}.")
            return False
        input_data[key] = data[key]

    input_data_checksum = calculate_dict_checksum(input_data)
    output_data = get_JSON(output_dir)

    if output_data is None:
        logger.info(f"Processing because the output JSON file doesn't exist.")
        return input_data, input_data_checksum

    if output_key not in output_data:
        logger.info(f"Processing because the output key {output_key} doesn't exist in the output JSON file.")
        return input_data, input_data_checksum

    if "input_data_checksum" in output_data[output_key]:
        if input_data_checksum == output_data[output_key]["input_data_checksum"]:
            logger.info("Skipping processing because the input data has not changed.")
            return False
        logger.info("Processing because the input data has changed.")
        return input_data, input_data_checksum

    logger.info("Processing because the input data checksum is missing in the output JSON file.")
    return input_data, input_data_checksum
