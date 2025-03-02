import subprocess

# Step 1: Run Chemprop2.py to generate antibiotic recommendations
print("Running Chemprop2.py...")
subprocess.run(["python", "Chemprop2.py"])

# Step 2: Run molgpt.py to generate new SMILES
print("Running molgpt.py...")
subprocess.run(["python", "molgpt.py"])

# Step 3: Run overview.py to display the image of the generated SMILES
print("Running overview.py...")
subprocess.run(["python", "overview.py"])

print("Pipeline execution complete!")