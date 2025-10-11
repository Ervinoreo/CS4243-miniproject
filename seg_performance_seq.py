import os
import argparse
from pathlib import Path


def analyze_folder_structure(folder_path):
    """
    Analyze folder structure to check if len(name) equals number of files inside.
    
    Args:
        folder_path (str): Path to the folder to analyze
    
    Returns:
        dict: Statistics about the analysis
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None
    
    total_folders = 0
    matching_folders = 0
    non_matching_folders = 0
    
    # Iterate through all items in the provided folder
    for item in folder_path.iterdir():
        # Skip if not a directory or if it's the debug folder
        if not item.is_dir() or item.name == "debug":
            continue
        
        total_folders += 1
        folder_name = item.name
        
        # Extract name part from format <name>-0
        if folder_name.endswith("-0"):
            name_part = folder_name[:-2]  # Remove "-0" suffix
        else:
            # If folder doesn't follow the expected format, use the whole name
            name_part = folder_name
        
        # Count files in the folder
        file_count = len([f for f in item.iterdir() if f.is_file()])
        name_length = len(name_part)
        
        # Check if name length equals file count
        if name_length == file_count:
            matching_folders += 1
        else:
            non_matching_folders += 1
    
    # Calculate percentages
    matching_percentage = (matching_folders / total_folders * 100) if total_folders > 0 else 0
    non_matching_percentage = (non_matching_folders / total_folders * 100) if total_folders > 0 else 0
    
    # Print summary statistics
    print("=" * 50)
    print("SUMMARY:")
    print(f"Total number of folders: {total_folders}")
    print(f"Folders where len(name) = number of files: {matching_folders} ({matching_percentage:.1f}%)")
    print(f"Folders where len(name) ≠ number of files: {non_matching_folders} ({non_matching_percentage:.1f}%)")
    print("=" * 50)
    
    return {
        'total_folders': total_folders,
        'matching_folders': matching_folders,
        'non_matching_folders': non_matching_folders
    }


def main():
    """Main function to handle command line arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze folder structure to check if folder name length matches file count"
    )
    parser.add_argument(
        "folder_path",
        help="Path to the folder to analyze"
    )
    
    args = parser.parse_args()
    
    # Run the analysis
    result = analyze_folder_structure(args.folder_path)
    
    if result is None:
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
