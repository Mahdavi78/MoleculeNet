"""
Utility classes for molecular data processing and tokenization.

Classes:
    SMILESTokenizerBuilder: Handles tokenization of SMILES strings with support for both
                           vocabulary-based and character-level tokenization.
    RawDataLoader: Loads molecular data from files and manages tokenization process.
"""

from typing import List, Dict, Union, Optional, Tuple, Type
import re
from collections import Counter
import pandas as pd
import os

"""Definition of SMILESTokenizerBuilder

Handles tokenization of SMILES strings with support for both
                           vocabulary-based and character-level tokenization.
"""
class SMILESTokenizerBuilder:
    def __init__(self, vocab_list: List[str] = None):
        """
        Initialize the SMILES tokenizer with an optional vocabulary list.
        
        Args:
            vocab_list (List[str], optional): List of SMILES tokens to use for vocabulary-based tokenization.
                                             If None, will only use character-level tokenization.
        """
        self.vocab_list = vocab_list if vocab_list is not None else []
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.token_counts = Counter()
        
    def build_vocab_mappings(self) -> None:
        """
        Build the vocabulary mappings (vocab_to_idx and idx_to_vocab).
        Must be called before using encode/decode methods.
        """
        if not self.vocab_list:
            raise ValueError("Cannot build vocabulary mappings: vocabulary list is empty")
            
        # Add special tokens for unknown characters
        self.vocab_list = self.vocab_list + ['<unk>']
        self.vocab_to_idx = {token: idx for idx, token in enumerate(self.vocab_list)}
        self.idx_to_vocab = {idx: token for token, idx in self.vocab_to_idx.items()}
        
    def tokenize_with_vocab(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES string using the vocabulary list first.
        If a token is not in vocabulary, it will be split into characters.
        
        Args:
            smiles (str): Input SMILES string
            
        Returns:
            List[str]: List of tokens
        """
        if not self.vocab_list:
            return self.tokenize_char_level(smiles)
            
        tokens = []
        i = 0
        while i < len(smiles):
            # Try to match the longest possible token from vocabulary
            matched = False
            for token in sorted(self.vocab_list, key=len, reverse=True):
                if token == '<unk>':  # Skip the unknown token
                    continue
                if smiles[i:].startswith(token):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            
            # If no match found, use character-level tokenization
            if not matched:
                tokens.append(smiles[i])
                i += 1
                
        return tokens
    
    def tokenize_char_level(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES string at character level.
        
        Args:
            smiles (str): Input SMILES string
            
        Returns:
            List[str]: List of character tokens
        """
        return list(smiles)
    
    def encode(self, smiles: str) -> List[int]:
        """
        Convert SMILES string to sequence of token indices.
        
        Args:
            smiles (str): Input SMILES string
            
        Returns:
            List[int]: List of token indices
            
        Raises:
            ValueError: If vocabulary mappings haven't been built
        """
        if self.vocab_to_idx is None or self.idx_to_vocab is None:
            raise ValueError("Vocabulary mappings not built. Call build_vocab_mappings() first.")
            
        tokens = self.tokenize_with_vocab(smiles)
        return [self.vocab_to_idx.get(token, self.vocab_to_idx['<unk>']) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """
        Convert sequence of token indices back to SMILES string.
        
        Args:
            indices (List[int]): List of token indices
            
        Returns:
            str: Reconstructed SMILES string
            
        Raises:
            ValueError: If vocabulary mappings haven't been built
        """
        if self.vocab_to_idx is None or self.idx_to_vocab is None:
            raise ValueError("Vocabulary mappings not built. Call build_vocab_mappings() first.")
            
        return ''.join([self.idx_to_vocab.get(idx, '') for idx in indices])
    
    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            int: Number of tokens in vocabulary
        """
        return len(self.vocab_list)


#RawDataLoader: Loads molecular data from files and manages tokenization process.
class RawDataLoader:
    """
    A class for loading and processing raw molecular data files.
    Handles data loading and creating tokenizer of SMILES strings.
    """
    
    def __init__(self, data_name: str, tokenizer_class: Optional[Type[SMILESTokenizerBuilder]] = None):
        """
        Initialize the RawDataLoader.
        
        Args:
            data_name (str): Name of the data file (e.g., "qm9.csv", "data.tsv")
            tokenizer_class (Type[SMILESTokenizerBuilder], optional): Class to use for tokenization
        """
        # Get the directory where utils.py is located
        self.utils_dir = os.path.dirname(os.path.abspath(__file__))
        # Go one level up from utils directory
        self.parent_dir = os.path.dirname(self.utils_dir)
        # Set data directory to "data inspection" folder
        self.data_dir = os.path.join(self.parent_dir, "data inspection")
        # Store the data file name
        self.data_name = data_name
        
        self.tokenizer_class = tokenizer_class
        self.tokenizer = None
        self.data = None
        
    def load_data(self, smiles_column: str = 'SMILES') -> pd.DataFrame:
        """
        Load data from the specified file in the data inspection folder.
        
        Args:
            smiles_column (str): Name of the column containing SMILES strings
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist in the data inspection folder
            ValueError: If the specified column doesn't exist in the file
        """
        file_path = os.path.join(self.data_dir, self.data_name)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file '{self.data_name}' not found in '{self.data_dir}'. "
                f"Please ensure the file exists in the data inspection folder."
            )
        
        # Read the data file based on extension
        if self.data_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif self.data_name.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(
                f"Unsupported file format for '{self.data_name}'. "
                f"Please use CSV or TSV files."
            )
        
        # Check if the SMILES column exists
        if smiles_column not in df.columns:
            raise ValueError(
                f"Column '{smiles_column}' not found in the data file. "
                f"Available columns: {', '.join(df.columns)}"
            )
            
        self.data = df
        return df
        
    def call_tokenizer(self, 
                      tokenizer_class: Type[SMILESTokenizerBuilder],
                      vocab_list: Optional[List[str]] = None,
                      smiles_column: str = 'SMILES') -> SMILESTokenizerBuilder:
        """
        Create and configure a tokenizer instance based on the loaded data.
        
        Args:
            tokenizer_class (Type[SMILESTokenizerBuilder]): Class to use for tokenization
            vocab_list (List[str], optional): List of tokens for vocabulary-based tokenization
            smiles_column (str): Name of the column containing SMILES strings
            
        Returns:
            SMILESTokenizerBuilder: Configured tokenizer instance
            
        Raises:
            ValueError: If no data has been loaded or if tokenizer_class is not provided
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if tokenizer_class is None:
            raise ValueError("tokenizer_class must be provided")
            
        # Create tokenizer instance
        self.tokenizer = tokenizer_class(vocab_list)
        
        # If vocab_list is None, return the tokenizer as is
        if vocab_list is None:
            print("No vocabulary list provided. Returning the tokenizer as is.")
            return self.tokenizer
            
        # Process SMILES strings to build vocabulary
        all_tokens = set()
        for smiles in self.data[smiles_column]:
            if pd.isna(smiles):
                continue
            tokens = self.tokenizer.tokenize_with_vocab(smiles)
            all_tokens.update(tokens)
        
        # Update tokenizer's vocabulary
        self.tokenizer.vocab_list = sorted(list(all_tokens))
        print("tokenizer is instantiated and vocab_list is updated")
        
        return self.tokenizer 