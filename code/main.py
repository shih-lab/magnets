import json
import os
import sys
import subprocess
from glob import glob

import yaml

def load_config(directory):
    config_path = os.path.join(directory, "config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_all_scripts(directory):
    config = load_config(directory)
    scripts = sorted(glob(os.path.join(directory, "[0-9]*.py")))

    for script in scripts:
        script_name = os.path.splitext(os.path.basename(script))[0]
        script_config = config.get(script_name, {})
        env = os.environ.copy()
        env["SCRIPT_CONFIG"] = json.dumps(script_config)
        print(f"running: {script}")
        try:
            subprocess.run([sys.executable, script], check=True, env=env)
        except subprocess.CalledProcessError:
            raise Exception(f"failed to execute: {script}")

if __name__ == "__main__":
    run_all_scripts(os.path.dirname(os.path.abspath(__file__)))