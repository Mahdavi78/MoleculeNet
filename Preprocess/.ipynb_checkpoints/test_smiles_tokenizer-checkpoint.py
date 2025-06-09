import os
import sys

# Print current working directory and Python path for debugging
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print("Added to Python path:", parent_dir)

try:
    from Preprocess.smiles_tokenizer import SMILESTokenizerBuilder
    print("Successfully imported SMILESTokenizerBuilder")
except ImportError as e:
    print("Error importing SMILESTokenizerBuilder:", e)
    sys.exit(1)

def print_separator(title):
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

def test_basic_tokenization():
    print_separator("1. Basic Tokenization Tests")
    
    # Test case 1: Simple SMILES with basic vocabulary
    vocab = ['C', 'CC', 'O', 'N', 'Cl', 'Br']
    tokenizer = SMILESTokenizerBuilder(vocab)
    
    # Test SMILES string (acetic acid)
    smiles = "CC(=O)O"
    print(f"Input SMILES: {smiles}")
    print(f"Tokenized: {tokenizer.tokenize_with_vocab(smiles)}")
    print(f"Character level: {tokenizer.tokenize_char_level(smiles)}")

def test_vocab_mapping():
    print_separator("2. Testing Vocabulary Mapping")
    
    vocab = ['C', 'CC', 'O', 'N', 'Cl', 'Br']
    tokenizer = SMILESTokenizerBuilder(vocab)
    
    print("Before building mappings:")
    print(f"vocab_to_idx: {tokenizer.vocab_to_idx}")
    print(f"idx_to_vocab: {tokenizer.idx_to_vocab}")
    
    tokenizer.build_vocab_mappings()
    
    print("\nAfter building mappings:")
    print(f"vocab_to_idx: {tokenizer.vocab_to_idx}")
    print(f"idx_to_vocab: {tokenizer.idx_to_vocab}")

def test_encode_decode():
    print_separator("3. Testing Encode/Decode Operations")
    
    vocab = ['C', 'CC', 'O', 'N', 'Cl', 'Br']
    tokenizer = SMILESTokenizerBuilder(vocab)
    tokenizer.build_vocab_mappings()
    
    smiles = "CC(=O)O"
    print(f"Original SMILES: {smiles}")
    
    # Encode
    indices = tokenizer.encode(smiles)
    print(f"Encoded indices: {indices}")
    
    # Decode
    reconstructed = tokenizer.decode(indices)
    print(f"Reconstructed SMILES: {reconstructed}")
    print(f"Reconstruction successful: {smiles == reconstructed}")

def test_error_handling():
    print_separator("4. Testing Error Handling")
    
    new_tokenizer = SMILESTokenizerBuilder(['C', 'O'])
    
    try:
        indices = new_tokenizer.encode("CO")
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Now build mappings and try again
    new_tokenizer.build_vocab_mappings()
    indices = new_tokenizer.encode("CO")
    print(f"Successfully encoded after building mappings: {indices}")

def test_complex_smiles():
    print_separator("5. Testing with More Complex SMILES")
    
    complex_vocab = ['C', 'CC', 'CCC', 'O', 'N', 'Cl', 'Br', 'c1ccccc1', 'C(=O)', 'C(=O)O']
    complex_tokenizer = SMILESTokenizerBuilder(complex_vocab)
    complex_tokenizer.build_vocab_mappings()
    
    # Test with aspirin SMILES
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    print(f"Aspirin SMILES: {aspirin}")
    print(f"Tokenized: {complex_tokenizer.tokenize_with_vocab(aspirin)}")
    print(f"Encoded: {complex_tokenizer.encode(aspirin)}")
    print(f"Decoded: {complex_tokenizer.decode(complex_tokenizer.encode(aspirin))}")

if __name__ == "__main__":
    print("Starting SMILESTokenizerBuilder Tests...")
    
    test_basic_tokenization()
    test_vocab_mapping()
    test_encode_decode()
    test_error_handling()
    test_complex_smiles()
    
    print("\nAll tests completed!") 