import os
import json
from typing import List
import hashlib

def get_JSON(dir,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(dir, name)

    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"{json_file_path} doesn't exist")
        return None
    
    return data

def write_JSON(dir,key,value,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(dir, name)

    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}  # Create an empty dictionary if the file doesn't exist

    data[key] = value
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    json_file.close()


def calculate_dict_checksum(data: dict) -> str:
    """
    Calculate a checksum (SHA256) of a dictionary.

    Args:
        data (dict): The dictionary to calculate the checksum for.

    Returns:
        str: The SHA256 checksum as a hexadecimal string.
    """
    # Serialize the dictionary into a sorted JSON string to ensure consistency
    serialized_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
    # Compute the SHA256 hash of the serialized string
    checksum = hashlib.sha256(serialized_data.encode('utf-8')).hexdigest()
    return checksum

def should_process(input_dirs:List[str], input_keys: List[str], output_dir: str, verbose: bool = False) -> dict|bool:
    if len(input_dirs) != len(input_keys):
        raise ValueError("The number of input directories and keys should be the same.")
    
    input_data = {}
    for dir,key in zip(input_dirs,input_keys):
        data = get_JSON(dir)
        if data is None:
            if verbose:
                print(f"Skipping because the JSON file in{dir} doesn't exist.")
            return False
        if key not in data:
            if verbose:
                print(f"Skipping because the key {key} doesn't exist in the JSON file in {dir}.")
            return False
        input_data[key]=data[key]

    output_data = get_JSON(output_dir)
    if output_data is None:
        if verbose:
            print(f"Processing because the output JSON file doesn't exist.")
        return input_data
    
        
    input_data_checksum = calculate_dict_checksum(input_data)
    if "input_data_checksum" in output_data:
        if input_data_checksum == output_data["input_data_checksum"]:
            if verbose:
                print("Skipping because the input data has not changed.")
            return False
        if verbose:
            print("Processing because the input data has changed.")
        return input_data
    if verbose:
        print("Processing because the input data checksum is missing in the output JSON file.")
    return input_data

    
    

        
    