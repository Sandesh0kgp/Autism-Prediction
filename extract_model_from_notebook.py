"""
Simple script to run the notebook cells and extract the trained model.
This ensures we use the EXACT same preprocessing and model from your notebook.
"""

import subprocess
import sys

print("=" * 70)
print("EXTRACTING MODEL FROM NOTEBOOK")
print("=" * 70)
print()
print("This script will:")
print("1. Convert your Jupyter notebook to a Python script")
print("2. Execute it to train the model")
print("3. Save best_model.pkl and encoders.pkl")
print()
print("=" * 70)
print()

# Convert notebook to Python script
print("Step 1: Converting notebook to Python script...")
try:
    result = subprocess.run([
        "jupyter", "nbconvert", "--to", "script",
        "Autism_Preidiction_using_machine_Learning.ipynb",
        "--output", "temp_notebook_script"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Error converting notebook:")
        print(result.stderr)
        print("\n💡 Make sure jupyter is installed: pip install jupyter")
        sys.exit(1)
    
    print("✅ Notebook converted successfully")
    
except FileNotFoundError:
    print("❌ jupyter command not found")
    print("💡 Install it with: pip install jupyter")
    sys.exit(1)

# Execute the script
print("\nStep 2: Executing notebook code to train model...")
print("(This may take a few minutes...)")
print()

try:
    result = subprocess.run([
        sys.executable, "temp_notebook_script.py"
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print("❌ Error executing notebook:")
        print(result.stderr)
        sys.exit(1)
    
    print("✅ Model training completed")
    
except subprocess.TimeoutExpired:
    print("⚠️  Training took too long (>5 minutes)")
    print("The model might still be training...")

# Check if files were created
import os

print("\nStep 3: Checking for model files...")

if os.path.exists("best_model.pkl"):
    print("✅ best_model.pkl created")
else:
    print("❌ best_model.pkl not found")

if os.path.exists("encoders.pkl"):
    print("✅ encoders.pkl created")
else:
    print("❌ encoders.pkl not found")

# Clean up
if os.path.exists("temp_notebook_script.py"):
    os.remove("temp_notebook_script.py")
    print("\n🧹 Cleaned up temporary files")

print()
print("=" * 70)
print("✅ DONE! You can now run: streamlit run app.py")
print("=" * 70)
