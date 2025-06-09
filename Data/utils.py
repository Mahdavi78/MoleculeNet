import torch
import matplotlib.pyplot as plt

def train_x_stats(loader, show_plot=True):
    """
    Analyze sequence length statistics from a data loader.
    
    Args:
        loader: DataLoader containing the data
        show_plot (bool): Whether to display the histogram plot
        
    Returns:
        dict: Dictionary containing statistics
    """
    # Get data from loader's dataset
    data = loader.dataset.tensors[0]  # Get X from dataset

    # Find first occurrence of padding token (0) for each sequence
    padding_mask = (data == 0)
    first_padding_idx = padding_mask.int().argmax(dim=1)

    # For sequences with no padding, use the full length
    no_padding_mask = ~padding_mask.any(dim=1)
    sequence_lengths = torch.where(
        no_padding_mask,
        torch.tensor(data.size(1), device=data.device),
        first_padding_idx
    )

    # Calculate statistics
    stats = {
        'min_length': sequence_lengths.min().item(),
        'max_length': sequence_lengths.max().item(),
        'mean_length': sequence_lengths.float().mean().item(),
        'median_length': sequence_lengths.float().median().item(),
        'std_length': sequence_lengths.float().std().item(),
        'total_size': len(data)
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
        plt.title('Distribution of SMILES Sequence Lengths')
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

    return stats 