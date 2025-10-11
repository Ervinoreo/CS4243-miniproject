import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def analyze_single_folder(item):
    """
    Analyze a single folder to check if name length equals file count.
    
    Args:
        item (Path): Path object representing the folder to analyze
    
    Returns:
        dict: Result of the analysis for this folder
    """
    folder_name = item.name
    
    # Extract name part from format <name>-0
    if folder_name.endswith("-0"):
        name_part = folder_name[:-2]  # Remove "-0" suffix
    else:
        # If folder doesn't follow the expected format, use the whole name
        name_part = folder_name
    
    # Count files in the folder
    try:
        file_count = len([f for f in item.iterdir() if f.is_file()])
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not access folder '{item}': {e}")
        return {'matches': False, 'error': True, 'folder_name': folder_name}
    
    name_length = len(name_part)
    matches = name_length == file_count
    
    return {
        'matches': matches,
        'error': False,
        'folder_name': folder_name,
        'name_length': name_length,
        'file_count': file_count
    }


def analyze_folder_structure(folder_path, max_workers=None, verbose=False):
    """
    Analyze folder structure to check if len(name) equals number of files inside.
    Uses parallel processing for improved performance.
    
    Args:
        folder_path (str): Path to the folder to analyze
        max_workers (int, optional): Maximum number of worker threads. 
                                   If None, uses min(32, (os.cpu_count() or 1) + 4)
        verbose (bool): If True, print detailed results for each folder
    
    Returns:
        dict: Statistics about the analysis
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None
    
    # Get all folders to analyze (excluding debug folder)
    folders_to_analyze = [
        item for item in folder_path.iterdir()
        if item.is_dir() and item.name != "debug"
    ]
    
    total_folders = len(folders_to_analyze)
    
    if total_folders == 0:
        print("No folders found to analyze.")
        return {'total_folders': 0, 'matching_folders': 0, 'non_matching_folders': 0}
    
    print(f"Analyzing {total_folders} folders using parallel processing...")
    
    matching_folders = 0
    non_matching_folders = 0
    errors = 0
    detailed_results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_folder = {
            executor.submit(analyze_single_folder, folder): folder 
            for folder in folders_to_analyze
        }
        
        # Process completed tasks
        for future in as_completed(future_to_folder):
            result = future.result()
            detailed_results.append(result)
            
            if result['error']:
                errors += 1
                non_matching_folders += 1  # Count errors as non-matching
            elif result['matches']:
                matching_folders += 1
            else:
                non_matching_folders += 1
    
    # Sort results by folder name for consistent output
    detailed_results.sort(key=lambda x: x['folder_name'])
    
    # Print detailed results if verbose mode is enabled
    if verbose:
        print("\nDETAILED RESULTS:")
        print("-" * 80)
        print(f"{'Folder Name':<20} {'Name Length':<12} {'File Count':<11} {'Match':<6} {'Status'}")
        print("-" * 80)
        for result in detailed_results:
            if result['error']:
                status = "ERROR"
                name_len = "N/A"
                file_count = "N/A"
                match = "NO"
            else:
                status = "OK"
                name_len = str(result['name_length'])
                file_count = str(result['file_count'])
                match = "YES" if result['matches'] else "NO"
            
            print(f"{result['folder_name']:<20} {name_len:<12} {file_count:<11} {match:<6} {status}")
        print("-" * 80)
    
    # Calculate percentages
    matching_percentage = (matching_folders / total_folders * 100) if total_folders > 0 else 0
    non_matching_percentage = (non_matching_folders / total_folders * 100) if total_folders > 0 else 0
    
    # Print summary statistics
    print("=" * 50)
    print("SUMMARY:")
    print(f"Total number of folders: {total_folders}")
    print(f"Folders where len(name) = number of files: {matching_folders} ({matching_percentage:.1f}%)")
    print(f"Folders where len(name) ≠ number of files: {non_matching_folders} ({non_matching_percentage:.1f}%)")
    if errors > 0:
        print(f"Folders with access errors: {errors}")
    print("=" * 50)
    
    return {
        'total_folders': total_folders,
        'matching_folders': matching_folders,
        'non_matching_folders': non_matching_folders,
        'errors': errors
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads (default: auto-detect based on CPU cores)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output showing per-folder results"
    )
    
    args = parser.parse_args()
    
    # Run the analysis
    result = analyze_folder_structure(
        args.folder_path, 
        max_workers=args.max_workers, 
        verbose=args.verbose
    )
    
    if result is None:
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())