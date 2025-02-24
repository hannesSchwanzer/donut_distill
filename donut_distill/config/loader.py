from typing import Any, Dict
import yaml
import donut_distill.config.config as CONFIG

def load_config(path: str):
    with open(path, 'r') as config_file:
        config_data: Dict[str, Any] = yaml.safe_load(config_file)

    for key, value in config_data.items():
        if hasattr(CONFIG, key.upper()):
            setattr(CONFIG, key.upper(), value)
