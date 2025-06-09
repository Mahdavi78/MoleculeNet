from typing import List, Dict, Union
import re

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
        
    def build_vocab_mappings(self) -> None:
        """
        Build the vocabulary mappings (vocab_to_idx and idx_to_vocab).
        Must be called before using encode/decode methods.
        """
        if not self.vocab_list:
            raise ValueError("Cannot build vocabulary mappings: vocabulary list is empty")
            
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
        return [self.vocab_to_idx.get(token, len(self.vocab_list)) for token in tokens]
    
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