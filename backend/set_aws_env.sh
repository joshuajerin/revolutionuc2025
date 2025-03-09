#!/bin/bash

# AWS Credentials
export AWS_ACCESS_KEY_ID="ASIA5ZIIF7NBCEYNETLH"
export AWS_SECRET_ACCESS_KEY="vvq/q7WzANvrjQ6+EHPMJp6HkUlTTzo+zfK0hE0Z"
export AWS_DEFAULT_REGION="us-east-1"

# Run the API server
cd "$(dirname "$0")"
PYTHONPATH=/Users/joshuajerin/miniconda3/envs/synthemol python run.py 