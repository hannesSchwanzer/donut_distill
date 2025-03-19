from typing import Any, Dict
import yaml
import donut_distill.config.config as CONFIG


def load_config(path: str):
    """
    Load configuration settings from a YAML file and update the CONFIG object.

    Args:
        path (str): Path to the YAML configuration file.

    Raises:
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(path, "r") as config_file:
        config_data: Dict[str, Any] = yaml.safe_load(config_file)

    # Update CONFIG attributes if they exist, using uppercase keys
    for key, value in config_data.items():
        if hasattr(CONFIG, key.upper()):
            setattr(CONFIG, key.upper(), value)
        else:
            print("Coultn't find key:", key.upper())
