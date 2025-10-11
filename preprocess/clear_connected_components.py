"""
Connected Components Module

This module handles connected component analysis using DFS.
"""

import numpy as np
from collections import deque


def dfs_connected_component(mask, start_y, start_x, visited):
    """
    Perform DFS to find all connected pixels of the same component.
    
    Args:
        mask: Binary mask
        start_y, start_x: Starting position
        visited: Visited positions set
    
    Returns:
        List of (y, x) coordinates in the connected component
    """
    height, width = mask.shape
    component = []
    stack = deque([(start_y, start_x)])
    
    # 8-connectivity directions
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while stack:
        y, x = stack.pop()
        
        if (y, x) in visited or y < 0 or y >= height or x < 0 or x >= width:
            continue
            
        if not mask[y, x]:
            continue
            
        visited.add((y, x))
        component.append((y, x))
        
        # Add neighbors to stack
        for dy, dx in directions:
            new_y, new_x = y + dy, x + dx
            if (new_y, new_x) not in visited:
                stack.append((new_y, new_x))
    
    return component


def find_connected_components(mask, min_component_size=1):
    """
    Find all connected components in the mask using DFS.
    
    Args:
        mask: Binary mask
        min_component_size: Minimum size for a component to be considered
    
    Returns:
        List of components, each component is a list of (y, x) coordinates
    """
    height, width = mask.shape
    visited = set()
    components = []
    
    for y in range(height):
        for x in range(width):
            if mask[y, x] and (y, x) not in visited:
                component = dfs_connected_component(mask, y, x, visited)
                if len(component) >= min_component_size:
                    components.append(component)
    
    return components