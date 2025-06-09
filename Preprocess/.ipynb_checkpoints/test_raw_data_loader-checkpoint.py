import os
import sys
import pandas as pd
from utils import RawDataLoader, SMILESTokenizerBuilder

def test_raw_data_loader():
    print("Starting RawDataLoader Tests...")
    print("=" * 50)
    
    # Test 1: Basic Initialization
    print("\nTest 1: Basic Initialization")
    print("-" * 30)
    try:
        data_loader = RawDataLoader()
        print("✓ Successfully initialized RawDataLoader")
    except Exception as e:
        print(f"✗ Failed to initialize RawDataLoader: {str(e)}")
        return
    
    # Test 2: Loading qm9.csv
    print("\nTest 2: Loading qm9.csv")
    print("-" * 30)
    try:
        data = data_loader.load_data("qm9.csv", smiles_column="smiles")
        print(f"✓ Successfully loaded qm9.csv")
        print(f"Number of rows: {len(data)}")
        print("\nFirst few rows:")
        print(data.head())
    except Exception as e:
        print(f"✗ Failed to load qm9.csv: {str(e)}")
        return
    
    # Test 3: Creating Tokenizer
    print("\nTest 3: Creating Tokenizer")
    print("-" * 30)
    try:
        tokenizer = data_loader.call_tokenizer(SMILESTokenizerBuilder)
        print("✓ Successfully created tokenizer")
    except Exception as e:
        print(f"✗ Failed to create tokenizer: {str(e)}")
        return
    
    # Test 4: Basic Tokenization
    print("\nTest 4: Basic Tokenization")
    print("-" * 30)
    test_smiles = "CC(=O)O"
    try:
        tokens = tokenizer.tokenize_with_vocab(test_smiles)
        print(f"Input SMILES: {test_smiles}")
        print(f"Tokenized: {tokens}")
    except Exception as e:
        print(f"✗ Failed to tokenize SMILES: {str(e)}")
        return
    
    # Test 5: Encode/Decode Operations
    print("\nTest 5: Encode/Decode Operations")
    print("-" * 30)
    try:
        encoded = tokenizer.encode(test_smiles)
        decoded = tokenizer.decode(encoded)
        print(f"Original SMILES: {test_smiles}")
        print(f"Encoded indices: {encoded}")
        print(f"Decoded SMILES: {decoded}")
        print(f"Reconstruction successful: {test_smiles == decoded}")
    except Exception as e:
        print(f"✗ Failed to encode/decode SMILES: {str(e)}")
        return
    
    # Test 6: Process Multiple SMILES
    print("\nTest 6: Process Multiple SMILES")
    print("-" * 30)
    try:
        # Get first 5 SMILES from the data
        sample_smiles = data['smiles'].head().tolist()
        print("Processing first 5 SMILES from qm9.csv:")
        for i, smiles in enumerate(sample_smiles, 1):
            tokens = tokenizer.tokenize_with_vocab(smiles)
            print(f"\n{i}. SMILES: {smiles}")
            print(f"   Tokens: {tokens}")
    except Exception as e:
        print(f"✗ Failed to process multiple SMILES: {str(e)}")
        return
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    # Add the parent directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    test_raw_data_loader() 