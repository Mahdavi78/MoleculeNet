import torch
import matplotlib.pyplot as plt

def train_x_stats(train_loader, val_loader, show_plot=True):
    """
    Analyze sequence length statistics from train and validation data loaders.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        show_plot (bool): Whether to display the histogram plot
        
    Returns:
        dict: Dictionary containing statistics
    """
    # Get all data from both loaders' datasets
    train_data = train_loader.dataset.tensors[0]  # Get X from train dataset
    val_data = val_loader.dataset.tensors[0]      # Get X from val dataset

    # Combine the data
    all_data = torch.cat([train_data, val_data])

    # Find first occurrence of padding token (0) for each sequence
    padding_mask = (all_data == 0)
    first_padding_idx = padding_mask.int().argmax(dim=1)

    # For sequences with no padding, use the full length
    no_padding_mask = ~padding_mask.any(dim=1)
    sequence_lengths = torch.where(
        no_padding_mask,
        torch.tensor(all_data.size(1), device=all_data.device),
        first_padding_idx
    )

    # Calculate statistics
    stats = {
        'min_length': sequence_lengths.min().item(),
        'max_length': sequence_lengths.max().item(),
        'mean_length': sequence_lengths.float().mean().item(),
        'median_length': sequence_lengths.float().median().item(),
        'std_length': sequence_lengths.float().std().item(),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'train_mean_length': sequence_lengths[:len(train_data)].float().mean().item(),
        'val_mean_length': sequence_lengths[len(train_data):].float().mean().item()
    }

    # Print statistics
    print("Sequence Length Statistics:")
    print(f"Min length: {stats['min_length']}")
    print(f"Max length: {stats['max_length']}")
    print(f"Mean length: {stats['mean_length']:.2f}")
    print(f"Median length: {stats['median_length']:.2f}")
    print(f"Std length: {stats['std_length']:.2f}")

    if show_plot:
        # Create histogram
        plt.figure(figsize=(12, 6))
        plt.hist(sequence_lengths.cpu().numpy(), bins=50, edgecolor='black')
        plt.title('Distribution of SMILES Sequence Lengths (Train + Val)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)

        # Add vertical lines for mean and median
        plt.axvline(stats['mean_length'], color='red', linestyle='dashed', linewidth=1, 
                   label=f'Mean: {stats["mean_length"]:.2f}')
        plt.axvline(stats['median_length'], color='green', linestyle='dashed', linewidth=1, 
                   label=f'Median: {stats["median_length"]:.2f}')
        plt.legend()
        plt.show()

    # Print length distribution in ranges
    print("\nLength Distribution:")
    length_ranges = [(1,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,45), (46,50)]
    for start, end in length_ranges:
        count = ((sequence_lengths >= start) & (sequence_lengths <= end)).sum().item()
        percentage = (count / len(sequence_lengths)) * 100
        print(f"Length {start}-{end}: {count} sequences ({percentage:.2f}%)")

    # Print separate statistics for train and val
    print("\nTrain vs Val Statistics:")
    print(f"Train set size: {stats['train_size']}")
    print(f"Val set size: {stats['val_size']}")
    print(f"Train mean length: {stats['train_mean_length']:.2f}")
    print(f"Val mean length: {stats['val_mean_length']:.2f}")

    return stats 