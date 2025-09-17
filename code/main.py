import os
import sys
import subprocess
from glob import glob

def run_all_notebooks(directory):
    notebooks = sorted(glob(os.path.join(directory, "*.ipynb")))

    for notebook in notebooks:
        print(f"running: {notebook}")
        try:
            subprocess.run([
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",  # overwrite the notebook with output
                notebook
            ], check=True)
        except subprocess.CalledProcessError:
            raise Exception(f"failed to execute: {notebook}")

if __name__ == "__main__":
    run_all_notebooks('/code')