import os
import json
import tempfile
import pytest
from zf_pf_geometry.metadata_manager import get_JSON,write_JSON,should_process, calculate_dict_checksum



def test_get_JSON_existing_file():
    """Test that get_JSON correctly reads an existing JSON file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare test JSON file
        test_data = {"key": "value"}
        json_file_path = os.path.join(temp_dir, "MetaData.json")
        with open(json_file_path, "w") as json_file:
            json.dump(test_data, json_file)

        # Call the function
        result = get_JSON(temp_dir)

        # Assertions
        assert result is not None, "The function should return data for an existing file."
        assert result == test_data, "The returned data should match the JSON file content."

def test_get_JSON_nonexistent_file():
    """Test that get_JSON returns None when the file does not exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function with no file present
        result = get_JSON(temp_dir)

        # Assertions
        assert result is None, "The function should return None when the file does not exist."

def test_get_JSON_custom_name():
    """Test that get_JSON handles custom file names correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare test JSON file with a custom name
        test_data = {"custom_key": "custom_value"}
        custom_name = "MetaData_custom.json"
        json_file_path = os.path.join(temp_dir, custom_name)
        with open(json_file_path, "w") as json_file:
            json.dump(test_data, json_file)

        # Call the function with a custom name
        result = get_JSON(temp_dir, name="custom")

        # Assertions
        assert result is not None, "The function should return data for an existing file with a custom name."
        assert result == test_data, "The returned data should match the content of the custom JSON file."

def test_write_JSON_creates_new_file():
    """Test that write_JSON creates a new file with the correct data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call write_JSON to create a new file
        write_JSON(temp_dir, key="test_key", value="test_value")

        # Verify the file was created and contains the correct data
        json_file_path = os.path.join(temp_dir, "MetaData.json")
        assert os.path.exists(json_file_path), "The JSON file should be created."

        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
        
        assert data == {"test_key": "test_value"}, "The JSON file should contain the correct data."

def test_write_JSON_updates_existing_file():
    """Test that write_JSON updates an existing file with the new key-value pair."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an initial JSON file
        initial_data = {"initial_key": "initial_value"}
        json_file_path = os.path.join(temp_dir, "MetaData.json")
        with open(json_file_path, "w") as json_file:
            json.dump(initial_data, json_file)

        # Call write_JSON to update the file
        write_JSON(temp_dir, key="new_key", value="new_value")

        # Verify the file contains both the old and new data
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        expected_data = {"initial_key": "initial_value", "new_key": "new_value"}
        assert data == expected_data, "The JSON file should contain both old and new data."

def test_write_JSON_with_custom_name():
    """Test that write_JSON handles custom file names correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call write_JSON with a custom file name
        write_JSON(temp_dir, key="custom_key", value="custom_value", name="custom")

        # Verify the custom file was created with the correct data
        json_file_path = os.path.join(temp_dir, "MetaData_custom.json")
        assert os.path.exists(json_file_path), "The custom JSON file should be created."

        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        assert data == {"custom_key": "custom_value"}, "The custom JSON file should contain the correct data."

def test_get_and_write_JSON_compatibility():
    """Test compatibility between write_JSON and get_JSON."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write some data using write_JSON
        write_JSON(temp_dir, key="compatible_key", value="compatible_value")

        # Read the data back using get_JSON
        result = get_JSON(temp_dir)

        # Verify that the retrieved data matches the written data
        assert result == {"compatible_key": "compatible_value"}, (
            "The data retrieved by get_JSON should match the data written by write_JSON."
        )

def create_json_file(directory: str, filename: str, data: dict) -> None:
    """Helper function to create a JSON file."""
    file_path = os.path.join(directory, filename)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)

def test_should_process_missing_json():
    """Test should_process when an input JSON file is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dirs = [temp_dir]
        input_keys = ["key1"]
        output_dir = os.path.join(temp_dir, "output")
        output_key = "output_key"

        # No JSON file exists in the input directory
        result = should_process(input_dirs, input_keys, output_dir, output_key, verbose=True)
        assert result is False, "The function should return False if an input JSON file is missing."

def test_should_process_missing_key():
    """Test should_process when a required key is missing in the input JSON."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an input JSON file without the required key
        input_data = {"other_key": "value"}
        create_json_file(temp_dir, "MetaData.json", input_data)

        input_dirs = [temp_dir]
        input_keys = ["key1"]
        output_dir = os.path.join(temp_dir, "output")
        output_key = "output_key"

        result = should_process(input_dirs, input_keys, output_dir, output_key, verbose=True)
        assert result is False, "The function should return False if a required key is missing."

def test_should_process_no_output_json():
    """Test should_process when the output JSON file is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create valid input JSON files
        input_data = {"key1": "value1"}
        create_json_file(temp_dir, "MetaData.json", input_data)

        input_dirs = [temp_dir]
        input_keys = ["key1"]
        output_dir = os.path.join(temp_dir, "output")
        output_key = "output_key"

        # Ensure no output JSON file exists
        result = should_process(input_dirs, input_keys, output_dir, output_key, verbose=True)
        assert result == ({"key1": "value1"}, calculate_dict_checksum(input_data)), (
            "The function should return input data and its checksum if the output JSON is missing."
        )

def test_should_process_input_data_changed():
    """Test should_process when input data has changed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create valid input JSON files
        input_data = {"key1": "value1"}
        create_json_file(temp_dir, "MetaData.json", input_data)

        input_dirs = [temp_dir]
        input_keys = ["key1"]

        # Create output JSON with old checksum
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        old_input_data = {"key1": "old_value"}
        old_checksum = calculate_dict_checksum(old_input_data)
        output_data = {"output_key": {"input_data_checksum": old_checksum}}
        create_json_file(output_dir, "MetaData.json", output_data)

        # Check if processing is required
        result = should_process(input_dirs, input_keys, output_dir, "output_key", verbose=True)
        assert result == ({"key1": "value1"}, calculate_dict_checksum(input_data)), (
            "The function should return input data and its checksum if the input data has changed."
        )

def test_should_process_no_changes():
    """Test should_process when there are no changes in input data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create valid input JSON files
        input_data = {"key1": "value1"}
        create_json_file(temp_dir, "MetaData.json", input_data)

        input_dirs = [temp_dir]
        input_keys = ["key1"]

        # Create output JSON with matching checksum
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        checksum = calculate_dict_checksum(input_data)
        output_data = {"output_key": {"input_data_checksum": checksum}}
        create_json_file(output_dir, "MetaData.json", output_data)

        # Check if processing is required
        result = should_process(input_dirs, input_keys, output_dir, "output_key", verbose=True)
        assert result is False, "The function should return False if the input data has not changed."


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
