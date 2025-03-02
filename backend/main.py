import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(smiles_input):
    """
    Run the complete pipeline with the provided SMILES input
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Starting pipeline with SMILES input: {smiles_input}")
        
        # Save SMILES input to a temporary file or environment variable
        os.environ['SMILES_INPUT'] = smiles_input
        
        # Step 1: Run Chemprop2.py
        logger.info("Running Chemprop2.py...")
        result = subprocess.run(
            ["python", os.path.join(current_dir, "Chemprop2.py")],
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Chemprop2.py output: {result.stdout}")
        
        # Step 2: Run Molgpt.py
        logger.info("Running Molgpt.py...")
        result = subprocess.run(
            ["python", os.path.join(current_dir, "Molgpt (1).py")],
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Molgpt.py output: {result.stdout}")
        
        # Step 3: Run Overview.py
        logger.info("Running Overview.py...")
        result = subprocess.run(
            ["python", os.path.join(current_dir, "Overview (1).py")],
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Overview.py output: {result.stdout}")
        
        logger.info("Pipeline execution complete!")
        return "Pipeline executed successfully!"
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Pipeline failed at step: {e.cmd[1]}. Error: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error in pipeline: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No SMILES input provided")
        sys.exit(1)
    
    smiles_input = sys.argv[1]
    try:
        result = run_pipeline(smiles_input)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1) 