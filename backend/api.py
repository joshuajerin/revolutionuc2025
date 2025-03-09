from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import subprocess
import sys
from typing import Dict
import asyncio
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the conda environment path where chemprop is installed
CONDA_ENV_PATH = os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "chemprop")

# Set up Python path based on operating system
if sys.platform == "win32":
    PYTHON_PATH = os.path.join(CONDA_ENV_PATH, "python.exe")
else:
    PYTHON_PATH = os.path.join(CONDA_ENV_PATH, "bin", "python")

logger.debug(f"Conda environment path: {CONDA_ENV_PATH}")
logger.debug(f"Python interpreter path: {PYTHON_PATH}")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the new_main directory for static file serving
app.mount("/static", StaticFiles(directory="new_main"), name="static")

class SimulationRequest(BaseModel):
    input_text: str

@app.post("/api/simulate")
async def run_simulation(request: SimulationRequest):
    try:
        logger.info(f"Received simulation request with input: {request.input_text}")
        
        # Validate input is not empty
        if not request.input_text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")

        # Get the absolute path to integrated_main.py
        main_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        if not os.path.exists(main_py_path):
            logger.error(f"integrated_main.py not found at path: {main_py_path}")
            raise HTTPException(status_code=500, detail=f"integrated_main.py not found at path: {main_py_path}")

        # Check if the chemprop environment exists
        if not os.path.exists(CONDA_ENV_PATH):
            error_msg = f"Chemprop environment not found at: {CONDA_ENV_PATH}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        logger.debug(f"Using Python from chemprop environment: {PYTHON_PATH}")
        logger.debug(f"Executing integrated_main.py at path: {main_py_path}")
        
        # Set up environment variables for the subprocess
        env = {
            **os.environ,
            "PATH": f"{os.path.join(CONDA_ENV_PATH, 'bin')}{os.pathsep}{os.environ.get('PATH', '')}",
            "CONDA_PREFIX": CONDA_ENV_PATH,
            "PYTHONPATH": CONDA_ENV_PATH,
            "AWS_DEFAULT_REGION": "us-east-1",
            "PYTHONUNBUFFERED": "1",
            # Pass AWS credentials from environment
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", "")
        }
        
        logger.info("=== Starting Simulation Process ===")
        logger.info("AWS Region: us-east-1")
        logger.info(f"AWS Access Key ends with: ...{env.get('AWS_ACCESS_KEY_ID', '')[-4:] if env.get('AWS_ACCESS_KEY_ID') else 'not set'}")
        logger.info(f"Session Token present: {bool(env.get('AWS_SESSION_TOKEN'))}")
        
        # Create a subprocess to run the simulation with the SMILES string as an argument
        process = await asyncio.create_subprocess_exec(
            PYTHON_PATH,
            main_py_path,
            request.input_text.strip(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(main_py_path),
            env=env
        )
        
        # Wait for the process to complete
        stdout, stderr = await process.communicate()
        
        stdout_text = stdout.decode()
        stderr_text = stderr.decode()
        
        logger.debug(f"Process returned with code: {process.returncode}")
        logger.debug(f"stdout: {stdout_text}")
        if stderr_text:
            logger.error(f"stderr: {stderr_text}")
        
        if process.returncode != 0:
            error_msg = f"Simulation failed with return code {process.returncode}. Error: {stderr_text}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Return the simulation results
        return {
            "status": "success",
            "output": stdout_text,
            "error": stderr_text if stderr_text else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.exception(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/molecule_2d")
async def get_molecule_2d():
    """Serve the 2D molecule image file"""
    try:
        file_path = os.path.join("new_main", "molecule.png")
        if not os.path.exists(file_path):
            logger.error(f"2D molecule image not found at path: {file_path}")
            raise HTTPException(status_code=404, detail="2D molecule image not found")
        
        headers = {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        return FileResponse(
            file_path,
            media_type="image/png",
            headers=headers,
            filename="molecule.png"
        )
    except Exception as e:
        logger.error(f"Error serving 2D molecule image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/molecule_3d")
async def get_molecule_3d():
    """Serve the 3D molecule visualization file"""
    file_path = os.path.join("new_main", "molecule_3d.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="3D model not found")
    
    # Read the HTML content
    with open(file_path, 'r') as f:
        html_content = f.read()
    
    # Return as HTML response
    return HTMLResponse(content=html_content) 