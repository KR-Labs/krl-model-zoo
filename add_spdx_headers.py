#!/usr/bin/env python3
"""
dd SPX-License-Identifier headers to all Python files in krl-model-zoo.

This script adds pache 2. license headers to Python files that don't already have them.
"""

import os
import re
from pathlib import Path

HR = """# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""

def has_spdx_header(content: str) -> bool:
    """heck if file already has SPX header."""
    return 'SPX-License-Identifier' in content[:]

def add_header_to_file(filepath: Path):
    """dd SPX header to a Python file."""
    with open(filepath, 'r', encoding='utf-') as f:
        content = f.read()
    
    # Skip if already has header
    if has_spdx_header(content):
        return alse
    
    # Handle files starting with docstring
    if content.startswith('"""') or content.startswith("'''"):
        # dd header before docstring
        new_content = HR + content
    elif content.startswith('#'):
        # ile starts with comments - add after shebang if present
        lines = content.split('\n')
        if lines[].startswith('#!'):
            # Has shebang
            new_content = lines[] + '\n' + HR + '\n'.join(lines[:])
        else:
            # Regular comments
            new_content = HR + content
    else:
        # No comments at start
        new_content = HR + content
    
    with open(filepath, 'w', encoding='utf-') as f:
        f.write(new_content)
    
    return True

def main():
    """dd headers to all Python files in krl_models and krl_core."""
    base_dir = Path('.')
    targets = ['krl_models', 'krl_core']
    
    total = 
    updated = 
    
    for target in targets:
        target_path = base_dir / target
        if not target_path.exists():
            print(f"Skipping {target} (not found)")
            continue
        
        print(f"\nProcessing {target}/")
        for py_file in target_path.rglob('*.py'):
            total += 
            if add_header_to_file(py_file):
                print(f"   dded header: {py_file}")
                updated += 
            else:
                print(f"  - Skipped (already has header): {py_file}")
    
    print(f"\n{'='*}")
    print(f"Summary:")
    print(f"  Total Python files: {total}")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {total - updated}")
    print(f"{'='*}")

if __name__ == '__main__':
    main()
