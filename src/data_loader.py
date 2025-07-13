"""
Contains the GPTDatasetV1 class and a utility function to create a PyTorch
DataLoader for training a GPT-style model.
"""

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        """
        Args:
            txt (str): The full raw text to process.
            tokenizer: The tokenizer instance (e.g. from tiktoken).
            max_length (int): The maximum length of each input sequence (context size).
            stride (int): The step size to move the sliding window across the text.git status
        """
        self.input_ids = []
        self.target_ids = []

        # Step 1: Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Step 2: Use a sliding window to create chunks of text
        for i in range(0,len(token_ids) - max_length, stride):
            # The input chunk is a sequence of tokens of size max_length
            input_chunk = token_ids[i:i + max_length]
            # The target chunk is the same sequence, shifted by one token to the right
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # Convert chunks to PyTorch tensors and store them
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self) -> int:
        """Returns the total number of chunks in the dataset."""
        return len(self.input_ids)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single input-target pair from the dataset at a given index.
        
        Args:
            index (int): The index of sample to retrieve.
        
        Returns:
            A tuple containing the input tensor and the target tensor.
        """
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(txt: str, batch_size: int = 4, max_length: int = 256, stride: int = 128,
                      shuffle: bool = True, drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Creates a PyTorch DataLoader from a raw text string.

    Args:
        txt (str): The raw text to looad.
        batch_size (int): The number of sequences per batch.
        max_length (int): The context size for the model.
        stride (int): The step size for the sliding window.
        shuffle (bool): Whether to shuffle the data chunks.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of subprocesses to use for data loading.
    
    Returns:
        DataLoader: A PyTorch DataLoader instance ready for training.
    """
    # Initialize the GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the dataset
    dataset = GPTDatasetV1(txt=txt, tokenizer=tokenizer, max_length=max_length, stride=stride)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader