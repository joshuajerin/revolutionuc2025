import boto3
import json
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def get_aws_client():
    try:
        logger.info("=== Starting AWS Client Initialization ===")
        
        # Create the Bedrock client
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        
        logger.info("=== AWS Client Initialization Complete ===")
        return bedrock
        
    except Exception as e:
        logger.error(f"AWS Client Error: {str(e)}")
        raise

def generate_molecule_overview(properties_dict, smiles):
    try:
        logger.info("=== Starting Molecule Overview Generation ===")
        
        # Create the prompt
        prompt = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{
                "role": "user",
                "content": f"""You are a pharmaceutical expert. Analyze this molecule and provide a comprehensive overview.

SMILES: {smiles}

Properties:
- Molecular Weight: {properties_dict['molecular_weight']:.2f} g/mol
- LogP (lipophilicity): {properties_dict['logp']:.2f}
- Hydrogen Bond Acceptors: {properties_dict['h_acceptors']}
- Hydrogen Bond Donors: {properties_dict['h_donors']}
- Rotatable Bonds: {properties_dict['num_rotatable_bonds']}

Please provide:
1. A brief description of the molecule's structure
2. Its potential pharmaceutical applications based on its properties
3. Any notable characteristics that might affect its drug-likeness
4. Potential concerns or advantages based on its properties

Format the response in clear sections with bullet points where appropriate."""
            }],
            "temperature": 0.7,
            "top_p": 0.9
        }

        # Get AWS client
        bedrock = get_aws_client()
        
        logger.info("Making request to AWS Bedrock")
        response = bedrock.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(prompt)
        )
        
        # Parse response
        response_body = json.loads(response.get('body').read())
        logger.debug(f"Raw response: {json.dumps(response_body, indent=2)}")
        
        overview = response_body.get('completion', '')
        if not overview:
            raise Exception("Empty response from AWS Bedrock")
        
        logger.info("Successfully generated overview")
        return overview
        
    except Exception as e:
        logger.error(f"Error in AWS overview generation: {str(e)}")
        raise

  def save_overview(overview, output_path):
      try:
          logger.info(f"Saving overview to {output_path}")
         with open(output_path, "w") as f:
         # First read the existing properties file
         properties_file = "molecule_properties.txt"
         existing_content = ""
         try:
             with open(properties_file, 'r') as f:
                 existing_content = f.read()
         except Exception as e:
             logger.error(f"Could not read properties file: {str(e)}")
         
         # Combine properties and overview
         combined_content = existing_content + "\n\n=== AI Analysis ===\n" + overview
         
         # Save combined content back to properties file
         with open(properties_file, "w") as f:
             f.write(combined_content)
         
         # Also save to original overview file
         with open(output_path, "w") as f:
              f.write(overview)
          logger.info("Successfully saved overview")
      except Exception as e:
          logger.error(f"Error saving overview: {str(e)}")
          raise 