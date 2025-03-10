import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re

def create_lyrics_dotplot(lyrics, by_character=False, remove_punctuation=True, case_sensitive=False, figsize=(10, 10)):
    """
    Create a dotplot visualization of song lyrics, similar to DNA sequence comparison.
    
    Parameters:
    -----------
    lyrics : str
        The song lyrics to visualize
    by_character : bool, default=False
        If True, compare individual characters; if False, compare words
    remove_punctuation : bool, default=True
        If True, remove punctuation from the lyrics
    case_sensitive : bool, default=False
        If True, comparison is case-sensitive
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Clean the lyrics
    if remove_punctuation:
        lyrics = re.sub(r'[^\w\s]', '', lyrics)
    
    if not case_sensitive:
        lyrics = lyrics.lower()
    
    # Split lyrics into units (characters or words)
    if by_character:
        units = list(lyrics)
    else:
        units = lyrics.split()
    
    # Create a matrix for the dotplot
    n = len(units)
    matrix = np.zeros((n, n))
    
    # Fill the matrix with matches
    for i in range(n):
        for j in range(n):
            if units[i] == units[j]:
                matrix[i, j] = 1
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a custom colormap (white background, blue dots)
    cmap = ListedColormap(['white', 'blue'])
    
    # Plot the matrix
    ax.matshow(matrix, cmap=cmap, aspect='auto')
    
    # Hide the tick labels if there are too many (especially for character-level analysis)
    if n > 50:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig, ax

def visualize_song(lyrics, title=None, artist=None, by_character=False, remove_punctuation=True, case_sensitive=False):
    """
    Visualize song lyrics as a dotplot.
    
    Parameters:
    -----------
    lyrics : str
        The song lyrics to visualize
    title : str, optional
        Song title for the plot title
    artist : str, optional
        Artist name for the plot title
    by_character : bool, default=False
        If True, analyze by character; if False, analyze by word
    remove_punctuation : bool, default=True
        If True, remove punctuation from the lyrics
    case_sensitive : bool, default=False
        If True, comparison is case-sensitive
    """
    fig, ax = create_lyrics_dotplot(lyrics, by_character, remove_punctuation, case_sensitive)
    
    # Add song details to the title if provided
    plot_title = None
    if title or artist:
        plot_title = ""
        if title:
            plot_title += f"\"{title}\""
        if artist:
            plot_title += f" by {artist}" if title else f"{artist}"
    
    # Try to show the plot, but also save it to a file in case we're in a non-interactive environment
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display the plot interactively: {e}")
    
    # Save the figure to a file with metadata
    output_file = f"{'lyrics' if not title else title.replace(' ', '_')}_{'char' if by_character else 'word'}.png"
    
    # Create metadata dictionary
    metadata = {'Title': title or 'Unknown', 'Artist': artist or 'Unknown', 'Analysis': f"{'Character' if by_character else 'Word'}-level"}
    
    # Save the figure with metadata
    fig.savefig(output_file, dpi=150, metadata=metadata)
    print(f"Plot saved to {output_file} with metadata: {metadata}")

def analyze_lyrics_from_file(filename, title=None, artist=None, by_character=False, 
                        remove_punctuation=True, case_sensitive=False):
    """
    Load lyrics from a text file and visualize as a dotplot.
    
    Parameters:
    -----------
    filename : str
        Path to the text file containing lyrics
    title : str, optional
        Song title for the plot title
    artist : str, optional
        Artist name for the plot title
    by_character : bool, default=False
        If True, analyze by character; if False, analyze by word
    remove_punctuation : bool, default=True
        If True, remove punctuation from the lyrics
    case_sensitive : bool, default=False
        If True, comparison is case-sensitive
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lyrics = file.read()
        
        # If title not provided, use filename as title
        if title is None:
            import os
            title = os.path.basename(filename).split('.')[0]
            
        visualize_song(lyrics, title, artist, by_character, remove_punctuation, case_sensitive)
        print(f"Successfully analyzed lyrics from {filename}")
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

# Example usage:
if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Visualize song lyrics as a DNA-like dotplot")
    parser.add_argument("filename", nargs="?", default="demo", help="Path to text file containing lyrics")
    parser.add_argument("--title", help="Song title")
    parser.add_argument("--artist", help="Artist name")
    parser.add_argument("--by-character", action="store_true", help="Analyze by character instead of by word")
    parser.add_argument("--keep-punctuation", action="store_true", help="Keep punctuation in the analysis")
    parser.add_argument("--case-sensitive", action="store_true", help="Make the analysis case-sensitive")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If demo mode is selected, run with sample lyrics
    if args.filename == "demo":
        print("\nRunning demo with sample lyrics...")
        sample_lyrics = """
        Verse 1
        This is the first line of the song
        This is the second line with a rhyme
        Here comes another line that's new
        And the final line of verse one too
        
        Chorus
        This is the chorus that repeats
        With catchy phrases and upbeat beats
        This is the chorus that repeats
        With catchy phrases and upbeat beats
        
        Verse 2
        Now we're back with the second verse
        Different words but the same universe
        Here comes another brand new line
        And the last line brings us back in time
        
        Chorus
        This is the chorus that repeats
        With catchy phrases and upbeat beats
        This is the chorus that repeats
        With catchy phrases and upbeat beats
        
        Bridge
        Something different for the bridge
        A new melody and different lyrics
        
        Chorus
        This is the chorus that repeats
        With catchy phrases and upbeat beats
        This is the chorus that repeats
        With catchy phrases and upbeat beats
        """
        
        print("\nWord-level analysis:")
        visualize_song(sample_lyrics, title="Example Song", artist="Example Artist")
        
        print("\nCharacter-level analysis:")
        visualize_song(sample_lyrics, title="Example Song", artist="Example Artist", by_character=True)
    else:
        # Analyze the lyrics file
        analyze_lyrics_from_file(
            args.filename, 
            title=args.title, 
            artist=args.artist, 
            by_character=args.by_character,
            remove_punctuation=not args.keep_punctuation,
            case_sensitive=args.case_sensitive
        )
