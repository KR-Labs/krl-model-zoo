#!/usr/bin/env python3
"""
Remove only visual emojis/symbols from repository files.
Does NOT remove regular text characters.
Targets .md, .py, .yml, .yaml files recursively.
"""

import re
from pathlib import Path

# Only target specific visual symbols - NOT Unicode letter ranges
# Using explicit character list to avoid any Unicode range issues
VISUAL_SYMBOLS = re.compile(r'[]')

def remove_visual_symbols(text):
    """Remove visual symbols but preserve all regular text."""
    return VISUAL_SYMBOLS.sub('', text)

def process_file(file_path):
    """Process a single file to remove visual symbols."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = remove_visual_symbols(content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    repo_root = Path(__file__).parent
    
    # File patterns to process
    patterns = ['*.md', '*.py', '*.yml', '*.yaml']
    
    # Paths to exclude
    exclude_paths = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.pytest_cache', 'htmlcov'}
    
    files_to_process = []
    for pattern in patterns:
        for file_path in repo_root.rglob(pattern):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in exclude_paths):
                continue
            files_to_process.append(file_path)
    
    print(f"Found {len(files_to_process)} files to process")
    
    modified_count = 0
    for file_path in files_to_process:
        if process_file(file_path):
            modified_count += 1
            print(f"Modified: {file_path.relative_to(repo_root)}")
    
    print(f"\nProcessing complete!")
    print(f"Modified {modified_count} files")
    print(f"Checked {len(files_to_process)} files total")

if __name__ == '__main__':
    main()
