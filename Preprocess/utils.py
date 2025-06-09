"""
Utility classes for molecular data processing and tokenization.

Classes:
    SMILESTokenizerBuilder: Handles tokenization of SMILES strings with support for both
                           vocabulary-based and character-level tokenization.
    RawDataLoader: Loads molecular data from files and manages tokenization process.
"""

from typing import List, Dict, Union, Optional, Tuple, Type, Set
import re
from collections import Counter
import pandas as pd
import os
import torch

"""Definition of SMILESTokenizerBuilder

Handles tokenization of SMILES strings with support for both
                           vocabulary-based and character-level tokenization.
"""
class SMILESTokenizerBuilder:
    """
    A class for tokenizing SMILES strings with support for both vocabulary-based
    and character-level tokenization.
    """
    
    def __init__(self, vocab_list: Optional[List[str]] = None):
        """
        Initialize the tokenizer with an optional vocabulary list.
        
        Args:
            vocab_list (List[str], optional): List of tokens for vocabulary-based tokenization
        """
        # Add special tokens to vocabulary
        self.special_tokens = ['<SOS>','<PAD>','<EOS>']
        # Convert vocabulary to uppercase if provided
        if vocab_list:
            # Convert all tokens to uppercase
            vocab_list = [token.upper() for token in vocab_list]
            # Add special tokens at the beginning
            self.vocab_list = self.special_tokens+ vocab_list 
        else:
            self.vocab_list = None
            
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.token_counts = Counter()
        
    def build_vocab_mappings(self) -> None:
        """
        Build vocabulary mappings with specific index ordering:
        - PAD: 0
        - SOS: 1
        - Alphabetic tokens (e.g., 'C', 'Cl', 'CH3'): 10-29
        - Non-numeric special tokens (e.g., '(', ')', '='): 30-39
        - Numeric tokens (e.g., '1', '2', '12'): 40-49
        - EOS: 60
        """
        if self.vocab_list is None:
            raise ValueError("Vocabulary list not set. Please set vocab_list first.")
            
        # Initialize mappings
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        
        # First assign special tokens
        self.vocab_to_idx['<PAD>'] = 0
        self.idx_to_vocab[0] = '<PAD>'
        
        self.vocab_to_idx['<SOS>'] = 1
        self.idx_to_vocab[1] = '<SOS>'
        
        self.vocab_to_idx['<EOS>'] = 60
        self.idx_to_vocab[60] = '<EOS>'
        
        # Separate remaining tokens into categories
        remaining_tokens = [t for t in self.vocab_list if t not in ['<PAD>', '<SOS>', '<EOS>']]
        
        # Categorize tokens
        alphabetic_tokens = []
        special_tokens = []
        numeric_tokens = []
        
        for token in remaining_tokens:
            if token.isalpha():  # Contains only letters
                alphabetic_tokens.append(token)
            elif token.isdigit():  # Contains only digits
                numeric_tokens.append(token)
            else:  # Contains special characters
                special_tokens.append(token)
        
        # Sort each category
        alphabetic_tokens.sort()
        special_tokens.sort()
        numeric_tokens.sort()
        
        # Assign indices to alphabetic tokens (starting from 10)
        for i, token in enumerate(alphabetic_tokens):
            idx = 10 + i
            self.vocab_to_idx[token] = idx
            self.idx_to_vocab[idx] = token
            
        # Assign indices to special tokens (starting from 30)
        for i, token in enumerate(special_tokens):
            idx = 30 + i
            self.vocab_to_idx[token] = idx
            self.idx_to_vocab[idx] = token
            
        # Assign indices to numeric tokens (starting from 40)
        for i, token in enumerate(numeric_tokens):
            idx = 40 + i
            self.vocab_to_idx[token] = idx
            self.idx_to_vocab[idx] = token

    def sort_tokens(self, tokens: List[str]) -> List[str]:
        """
        Sort tokens in specific order:
        1. 'SOS'
        2. Alphabets (A-Z)
        3. Non-numeric special characters
        4. Numbers (0-9)
        5. 'EOS'
        6. 'PAD'
        """
        # First ensure all special tokens are in the list
        all_tokens = set(tokens)  # Convert to set to remove duplicates
        
        # Separate tokens into categories
        special_tokens = []
        if '<SOS>' in all_tokens:
            special_tokens.append('<SOS>')
        if '<EOS>' in all_tokens:
            special_tokens.append('<EOS>')
        if '<PAD>' in all_tokens:
            special_tokens.append('<PAD>')
            
        # Get remaining tokens (excluding special tokens)
        remaining = [t for t in all_tokens if t not in ['<PAD>', '<SOS>', '<EOS>']]
        
        # Separate into alphabets, special chars, and numbers
        alphabets = sorted([t for t in remaining if t.isalpha()])
        special_chars = sorted([t for t in remaining if not t.isalnum()])
        numbers = sorted([t for t in remaining if t.isdigit()])
        
        # Combine in specified order: SOS, alphabets, special chars, numbers, EOS, PAD
        return special_tokens[:1] + alphabets + special_chars + numbers + special_tokens[1:2] + special_tokens[2:]

    def tokenize_char_level(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES string at character level.
        All characters are converted to uppercase.
        
        Args:
            smiles (str): Input SMILES string
            
        Returns:
            List[str]: List of character tokens
        """
        # Convert to uppercase and split into characters
        tokens = list(smiles.upper())
        return self.sort_tokens(tokens)
    
    def tokenize_with_vocab(self, Data_frame: Union[str, pd.DataFrame], smiles_column: str = 'smiles', vocab: List[str] = None) -> List[str]:
        """
        Process SMILES strings to build vocabulary. Adds new tokens to vocab_list when found.
        If DataFrame is provided, it will process all SMILES strings in the specified column.
        Counts frequency of each token encountered.
        
        Args:
            Data_frame (Union[str, pd.DataFrame]): Input SMILES string or DataFrame with SMILES column
            smiles_column (str): Name of the column containing SMILES strings (default: 'smiles')
            vocab (List[str], optional): Vocabulary to use for tokenization. If None, uses instance vocab_list.
            
        Returns:
            List[str]: Updated vocabulary list including new tokens found
            
        Raises:
            ValueError: If no vocabulary is available or if smiles column not found in DataFrame
        """
        if vocab is not None:
            print("Using vocabulary provided in tokenizer instance")
            self.vocab_list = vocab
        elif self.vocab_list is None:
            raise ValueError("No vocabulary available. Please provide a vocabulary or set vocab_list.")
            
        # Sort vocabulary by length (longest first) to ensure proper tokenization
        self.vocab_list = sorted(self.vocab_list, key=len, reverse=True)
        
        # Initialize token counts if not already done
        if not hasattr(self, 'token_counts'):
            self.token_counts = {}
        
        if isinstance(Data_frame, pd.DataFrame):
            # Process entire DataFrame
            if smiles_column not in Data_frame.columns:
                raise ValueError("Default'smiles' column not found in DataFrame. Please provide a column name")
            
            for smi in Data_frame[smiles_column]:
                # Convert SMILES to uppercase
                smi = smi.upper()
                i = 0
                while i < len(smi):
                    matched = False
                    for token in self.vocab_list:
                        if smi[i:i+len(token)] == token:
                            # Count token frequency
                            self.token_counts[token] = self.token_counts.get(token, 0) + 1
                            i += len(token)
                            matched = True
                            break
                    if not matched:
                        # If no match found, add the character as a token to vocab_list
                        new_token = smi[i]
                        # Count token frequency
                        self.token_counts[new_token] = self.token_counts.get(new_token, 0) + 1
                        if new_token not in self.vocab_list:
                            self.vocab_list.append(new_token)  # Single characters are always shorter
                        i += 1
        else:
            # Process single SMILES string
            smi = Data_frame.upper()
            i = 0
            while i < len(smi):
                matched = False
                for token in self.vocab_list:
                    if smi[i:i+len(token)] == token:
                        # Count token frequency
                        self.token_counts[token] = self.token_counts.get(token, 0) + 1
                        i += len(token)
                        matched = True
                        break
                if not matched:
                    # If no match found, add the character as a token to vocab_list
                    new_token = smi[i]
                    # Count token frequency
                    self.token_counts[new_token] = self.token_counts.get(new_token, 0) + 1
                    if new_token not in self.vocab_list:
                        self.vocab_list.append(new_token)  # Single characters are always shorter
                    i += 1
        
        # Sort the final vocabulary in the specified order
        self.vocab_list = self.sort_tokens(self.vocab_list)
        return self.vocab_list
    
    
    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary based on the range of indices.
        
        Returns:
            int: Size of the vocabulary (max index - min index + 1)
        """
        if self.vocab_to_idx is None:
            raise ValueError("Vocabulary mappings not built. Call build_vocab_mappings first.")
        return max(self.vocab_to_idx.values()) - min(self.vocab_to_idx.values()) + 1

    def encode_to_tensor(self, pd: pd.DataFrame, seq_len: int, smiles_col: str = 'smiles', vocab_to_idx: Dict[str, int] = None) -> torch.Tensor:
        """
        Convert SMILES strings from DataFrame to padded tensor of indices.
        
        Args:
            pd (pd.DataFrame): DataFrame containing SMILES strings
            seq_len (int): Desired sequence length (0 to seq_len-1)
            smiles_col (str, optional): Name of column containing SMILES strings. Defaults to 'smiles'
            vocab_to_idx (Dict[str, int], optional): Vocabulary mapping. If None, uses instance's vocab_to_idx
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len) containing token indices
            
        Raises:
            ValueError: If vocab_to_idx is not provided and instance's vocab_to_idx is None
            ValueError: If smiles_col is not found in DataFrame columns
        """
        # Check vocabulary mapping
        if vocab_to_idx is None:
            if self.vocab_to_idx is None:
                raise ValueError("No vocabulary mapping provided and instance's vocab_to_idx is None")
            vocab_to_idx = self.vocab_to_idx
            
        # Check if smiles column exists
        if smiles_col not in pd.columns:
            raise ValueError(f"Column '{smiles_col}' not found in DataFrame")
            
        # Initialize list to store all sequences
        all_sequences = []
        
        # Process each SMILES string
        for smiles in pd[smiles_col]:
            # Convert to uppercase and split into characters
            tokens = list(smiles.upper())
            
            # Add special tokens
            tokens = ['<SOS>'] + tokens + ['<EOS>']
            
            # Convert to indices
            indices = [vocab_to_idx[token] for token in tokens]
            
            # Pad or truncate to desired length
            if len(indices) < seq_len:
                indices.extend([vocab_to_idx['<PAD>']] * (seq_len - len(indices)))
            else:
                indices = indices[:seq_len]
                
            all_sequences.append(indices)
            
        # Convert to tensor
        return torch.tensor(all_sequences, dtype=torch.long)

#RawDataLoader: Loads molecular data from files and manages tokenization process.
class RawDataLoader:
    """
    A class for loading and processing raw molecular data files.
    Handles data loading and creating tokenizer of SMILES strings.
    """
    
    def __init__(self, data_name: str, tokenizer_class: Optional[Type[SMILESTokenizerBuilder]] = None, smiles_column: str = 'smiles'):
        """
        Initialize the RawDataLoader and automatically load the data.
        
        Args:
            data_name (str): Name of the data file (e.g., "qm9.csv", "data.tsv")
            tokenizer_class (Type[SMILESTokenizerBuilder], optional): Class to use for tokenization
            smiles_column (str): Name of the column containing SMILES strings
        """
        # Get the directory where utils.py is located
        self.utils_dir = os.path.dirname(os.path.abspath(__file__))
        # Go one level up from utils directory
        self.parent_dir = os.path.dirname(self.utils_dir)
        # Set data directory to "data inspection" folder
        self.data_dir = os.path.join(self.parent_dir, "data inspection")
        # Store the data file name
        self.data_name = data_name
        # Store the full path to the data file
        self.data_path = os.path.join(self.data_dir, self.data_name)
        
        self.tokenizer_class = tokenizer_class
        self.tokenizer = None
        
        # Automatically load the data
        self.data = self.load_data(self.data_path, smiles_column)
    
    @staticmethod
    def load_data(file_path: str, smiles_column: str = 'SMILES') -> pd.DataFrame:
        """
        Load data from the specified file based on its extension.
        
        Args:
            file_path (str): Full path to the data file
            smiles_column (str): Name of the column containing SMILES strings
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the specified column doesn't exist in the file or if file format is not supported
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found at '{file_path}'"
            )
        
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            return RawDataLoader._load_csv(file_path, smiles_column)
        elif file_path.endswith('.tsv'):
            return RawDataLoader._load_tsv(file_path, smiles_column)
        else:
            raise ValueError(
                f"Unsupported file format for '{file_path}'. "
                f"Please use CSV or TSV files."
            )
    
    @staticmethod
    def _load_csv(file_path: str, smiles_column: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            smiles_column (str): Name of the column containing SMILES strings
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            ValueError: If the specified column doesn't exist in the file
        """
        df = pd.read_csv(file_path)
        if smiles_column not in df.columns:
            raise ValueError(
                f"Column '{smiles_column}' not found in the CSV file. "
                f"Available columns: {', '.join(df.columns)}"
            )
        return df
    
    @staticmethod
    def _load_tsv(file_path: str, smiles_column: str) -> pd.DataFrame:
        """
        Load data from a TSV file.
        
        Args:
            file_path (str): Path to the TSV file
            smiles_column (str): Name of the column containing SMILES strings
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            ValueError: If the specified column doesn't exist in the file
        """
        df = pd.read_csv(file_path, sep='\t')
        if smiles_column not in df.columns:
            raise ValueError(
                f"Column '{smiles_column}' not found in the TSV file. "
                f"Available columns: {', '.join(df.columns)}"
            )
        return df
        