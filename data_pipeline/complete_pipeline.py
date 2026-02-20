"""
Orchestrates the entire NanoMath data pipeline, running each data processing script sequentially.
"""

import os
import subprocess
import sys

def run_step(script_path):
    """Executes a single python script via subprocess."""
    print(f"\n========== RUNNING {os.path.basename(script_path)} ==========")
    try:
        # Run the script using the current Python executable
        # check=True will raise an error if the script fails
        subprocess.run([sys.executable, script_path], check=True)
        print(f"========== FINISHED {os.path.basename(script_path)} ==========\n")
    except subprocess.CalledProcessError:
        print(f"\n!!!!!!!!!! ERROR in {os.path.basename(script_path)} !!!!!!!!!!")
        print("Stopping pipeline.")
        sys.exit(1) # Exit with an error code

if __name__ == "__main__":
    # The list of scripts to run in order
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        os.path.join(script_dir, "download_data.py"),
        os.path.join(script_dir, "generate_math_problems.py"),
        os.path.join(script_dir, "tokenizer.py"),
        os.path.join(script_dir, "pre_tokenize.py")
    ]

    for script_path in scripts:
        run_step(script_path)

    print("Pipeline complete! 🚀")