import boto3
import json

# Initialize the Bedrock runtime client
client = boto3.client('bedrock-runtime')

# Use the correct model ID from list-foundation-models
model_id = "us.anthropic.claude-3-haiku-20240307-v1:0"

# Define the input payload (corrected format)
payload = {
    "anthropic_version": "bedrock-2023-05-31",  # Required field for Claude models
    "messages": [
        {"role": "user", "content": "Write a short poem about AI and the future."}
    ],
    "max_tokens": 200  # Adjust as needed
}

# Convert payload to JSON
payload_json = json.dumps(payload)

# Invoke the model using the correct model ID
response = client.invoke_model(
    modelId=model_id,
    body=payload_json,
    accept="application/json",
    contentType="application/json"
)

# Read and parse response
response_body = json.loads(response["body"].read())

# Print the response
print(response_body)
