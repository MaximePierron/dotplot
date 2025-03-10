import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import librosa
import argparse

def create_frequency_dotplot(audio_file, n_fft=2048, hop_length=512, similarity_threshold=0.7, 
                            similarity_measure="cosine", min_freq=20, max_freq=4000, 
                            denoise=False, figsize=(12, 10)):
    """
    Create a dotplot visualization of audio with color representing frequency.
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Number of samples between successive frames
    similarity_threshold : float, default=0.7
        Threshold for similarity (higher = more strict matching)
    similarity_measure : str, default="cosine"
        Method to calculate similarity: "cosine", "correlation", or "binary"
    figsize : tuple, default=(12, 10)
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Load audio file
    print(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file)
    print(f"Audio loaded: {len(y)/sr:.2f} seconds, {sr}Hz sample rate")
    
    # Compute spectrogram
    print("Computing spectrogram...")
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Filter frequencies if specified
    if min_freq > 0 or max_freq < sr/2:
        print(f"Filtering frequencies to range {min_freq}-{max_freq} Hz")
        # Convert Hz to spectrogram bin indices
        min_bin = int(min_freq * n_fft / sr)
        max_bin = int(max_freq * n_fft / sr)
        # Only keep the specified frequency range
        S_filtered = S[min_bin:max_bin, :]
    else:
        S_filtered = S
    
    # Calculate number of frames
    n_frames = S_filtered.shape[1]
    n_freqs = S_filtered.shape[0]
    print(f"Spectrogram computed: {n_frames} frames, {n_freqs} frequency bins")
    
    # Normalize each frame for comparison
    S_norm = librosa.util.normalize(S_filtered, axis=0)
    
    # Initialize similarity matrix
    similarity = np.zeros((n_frames, n_frames))
    
    # Initialize frequency content matrix (will store dominant frequency for each match)
    freq_content = np.zeros((n_frames, n_frames))
    
    print("Computing similarity and frequency matrices...")
    # Compute similarity matrix
    if similarity_measure == "cosine":
        # Cosine similarity
        similarity = np.dot(S_norm.T, S_norm)
        
        # For each pair of similar frames, identify dominant frequency
        for i in range(n_frames):
            for j in range(i, n_frames):  # Only compute upper triangle
                if similarity[i, j] >= similarity_threshold:
                    # Find the dominant frequency contribution to this similarity
                    contribution = S_norm[:, i] * S_norm[:, j]
                    # Weighted average frequency (bin index)
                    freq_bin = np.argmax(contribution) 
                    # Convert bin to normalized value (0-1)
                    norm_freq = freq_bin / n_freqs
                    freq_content[i, j] = norm_freq
                    freq_content[j, i] = norm_freq  # Mirror
    
    elif similarity_measure == "binary":
        # Number of peaks to consider
        num_peaks = 10
        
        # Find peaks in each frame
        peaks = []
        peak_freqs = []  # Store the actual frequencies of the peaks
        
        for i in range(n_frames):
            frame = S[:, i]
            # Get indices of top frequencies
            peak_indices = np.argsort(frame)[-num_peaks:]
            peaks.append(set(peak_indices))
            peak_freqs.append(peak_indices)
        
        # Compare peak sets
        for i in range(n_frames):
            for j in range(i, n_frames):  # Only compute upper triangle
                # Find matching peaks
                matching_peaks = peaks[i].intersection(peaks[j])
                matches = len(matching_peaks)
                similarity_val = matches / num_peaks
                
                # If similar enough, store similarity and frequency info
                if similarity_val >= similarity_threshold:
                    similarity[i, j] = similarity_val
                    similarity[j, i] = similarity_val  # Mirror
                    
                    # Find average frequency of matching peaks (or highest energy if no match)
                    if matches > 0:
                        # Get the frequency bins that matched
                        match_freqs = list(matching_peaks)
                        # Get the average frequency bin
                        avg_freq = np.mean(match_freqs) / n_freqs  # Normalize to 0-1
                    else:
                        avg_freq = 0.5  # Default to middle frequency
                    
                    freq_content[i, j] = avg_freq
                    freq_content[j, i] = avg_freq  # Mirror
    
    # Create frequency color map
    # Low frequencies: red, mid frequencies: green, high frequencies: blue
    cmap = plt.cm.jet
    
    # Create a masked version of the frequency content
    masked_freq = np.ma.masked_where(similarity < similarity_threshold, freq_content)
    
    # Apply denoising if requested
    if denoise:
        print("Applying denoising to remove isolated points...")
        # Create a binary version of the similarity matrix
        binary = (similarity >= similarity_threshold).astype(float)
        
        # Create a cleaned copy for processing
        cleaned = np.copy(binary)
        
        # For each point, check if it has enough neighbors
        for i in range(1, n_frames-1):
            for j in range(1, n_frames-1):
                if binary[i, j] > 0:
                    # Count neighbors in a 3x3 window
                    neighbors = np.sum(binary[i-1:i+2, j-1:j+2]) - binary[i, j]
                    # If not enough neighbors, remove the point
                    if neighbors < 2:
                        cleaned[i, j] = 0
        
        # Update the masked frequency content based on cleaned matrix
        masked_freq = np.ma.masked_where(cleaned < 0.5, freq_content)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    print("Generating visualization...")
    # Create a white background
    ax.set_facecolor('white')
    
    # Plot frequency content with the jet colormap
    im = ax.matshow(masked_freq, cmap=cmap, aspect='auto')
    
    # Add a colorbar for frequency
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Normalized Frequency (0=low, 1=high)')
    
    # Set labels
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Time (frames)")
    
    # Add time markers (approximate, based on hop_length and sample rate)
    seconds_per_frame = hop_length / sr
    max_seconds = int(n_frames * seconds_per_frame)
    
    # Add time ticks every 15 seconds
    if max_seconds > 15:
        seconds_ticks = np.arange(0, max_seconds, 15)
        frame_ticks = [int(s / seconds_per_frame) for s in seconds_ticks]
        
        ax.set_xticks(frame_ticks)
        ax.set_yticks(frame_ticks)
        ax.set_xticklabels([f"{s}s" for s in seconds_ticks])
        ax.set_yticklabels([f"{s}s" for s in seconds_ticks])
    
    plt.tight_layout()
    return fig, ax, similarity, masked_freq

def visualize_frequency_dotplot(audio_file, title=None, artist=None, n_fft=2048, hop_length=512,
                               similarity_threshold=0.7, similarity_measure="cosine", 
                               min_freq=20, max_freq=4000, denoise=False):
    """
    Visualize audio with frequency-colored dotplot.
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file
    title : str, optional
        Song title for metadata
    artist : str, optional
        Artist name for metadata
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Number of samples between successive frames
    similarity_threshold : float, default=0.7
        Threshold for similarity (higher = more strict matching)
    similarity_measure : str, default="cosine"
        Method to calculate similarity: "cosine", "correlation", or "binary"
    """
    # Create the dotplot
    fig, ax, similarity, freq_content = create_frequency_dotplot(
        audio_file, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        similarity_threshold=similarity_threshold,
        similarity_measure=similarity_measure,
        min_freq=min_freq,
        max_freq=max_freq,
        denoise=denoise
    )
    
    # Try to show the plot, but also save it to a file
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display the plot interactively: {e}")
    
    # Save the figure with metadata
    output_file = f"{'audio' if not title else title.replace(' ', '_')}_freq_dotplot.png"
    
    # Create metadata dictionary
    metadata = {
        'Title': title or 'Unknown', 
        'Artist': artist or 'Unknown',
        'Analysis': f"Audio frequency-colored similarity ({similarity_measure}, thresh={similarity_threshold})"
    }
    
    # Save the figure with metadata
    fig.savefig(output_file, dpi=150, metadata=metadata)
    print(f"Plot saved to {output_file} with metadata: {metadata}")

    return similarity, freq_content

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize audio with frequency-colored dotplot")
    parser.add_argument("filename", nargs="?", default=None, help="Path to audio file")
    parser.add_argument("--title", help="Song title")
    parser.add_argument("--artist", help="Artist name")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for STFT")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--measure", choices=["cosine", "binary"], default="cosine",
                        help="Similarity measure to use")
    parser.add_argument("--min-freq", type=int, default=20, help="Minimum frequency to analyze (Hz)")
    parser.add_argument("--max-freq", type=int, default=4000, help="Maximum frequency to analyze (Hz)")
    parser.add_argument("--denoise", action="store_true", help="Apply additional denoising to remove isolated points")
    
    args = parser.parse_args()
    
    if args.filename:
        visualize_frequency_dotplot(
            args.filename,
            title=args.title,
            artist=args.artist,
            n_fft=            args.n_fft,
            hop_length=args.hop_length,
            similarity_threshold=args.threshold,
            similarity_measure=args.measure,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            denoise=args.denoise
        )
    else:
        print("Please provide an audio file path.")
        print("Example usage: python audio_freq_dotplot.py song.mp3 --title 'My Song' --artist 'Artist'")
        print("\nOptions:")
        print("  --threshold 0.75      Higher threshold shows only stronger matches")
        print("  --measure binary      Use peak-based comparison")
        print("  --min-freq 200        Filter out low frequencies (rumble, bass)")
        print("  --max-freq 2000       Filter out high frequencies")
        print("  --n-fft 4096          Higher resolution frequency analysis (larger window)")
        print("  --hop-length 1024     Larger hop length (faster but less detailed)")
        print("  --denoise             Apply denoising to remove isolated points")