#!/usr/bin/env python3
# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Copyright Header Injection Script

Automatically adds copyright and trademark headers to source files.
Supports Python, Markdown, YAML, and other file types.

Usage:
    python add_copyright_headers.py [--dry-run] [--repo REPO_NAME]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Copyright header templates
PYTHON_HEADER = """# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""

MARKDOWN_HEADER = """---
© 2025 KR-Labs. All rights reserved.  
KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.

SPDX-License-Identifier: Apache-2.0
---

"""

YAML_HEADER = """# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""

# Files and directories to skip
SKIP_PATTERNS = {
    '.git', '.github', '__pycache__', '.pytest_cache', '.venv', 'venv', 
    'node_modules', '.benchmarks', 'htmlcov', '.mypy_cache', 'dist', 'build',
    '.egg-info', 'LICENSE', 'NOTICE', '.DS_Store', '*.pyc', '*.pyo', 
    '*.egg-info', '.coverage', 'coverage.xml', '.gitignore'
}

# File extensions to process
PYTHON_EXTENSIONS = {'.py'}
MARKDOWN_EXTENSIONS = {'.md'}
YAML_EXTENSIONS = {'.yml', '.yaml'}

def should_skip(file_path: Path) -> bool:
    """Check if file should be skipped."""
    # Skip if any part of path matches skip patterns
    for part in file_path.parts:
        if part in SKIP_PATTERNS:
            return True
    
    # Skip if filename matches skip patterns
    if file_path.name in SKIP_PATTERNS:
        return True
    
    # Skip if extension matches skip patterns
    if any(file_path.match(pattern) for pattern in SKIP_PATTERNS if '*' in pattern):
        return True
    
    return False

def has_copyright_header(content: str) -> bool:
    """Check if file already has copyright header."""
    copyright_markers = [
        '© 2025 KR-Labs',
        'Copyright (c) 2025 KR-Labs',
        'KR-Labs™ is a trademark'
    ]
    return any(marker in content[:500] for marker in copyright_markers)

def add_header_to_python(file_path: Path, dry_run: bool = False) -> bool:
    """Add copyright header to Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if has_copyright_header(content):
            return False
        
        # Handle shebang
        if content.startswith('#!'):
            lines = content.split('\n', 1)
            new_content = lines[0] + '\n' + PYTHON_HEADER
            if len(lines) > 1:
                new_content += lines[1]
        else:
            new_content = PYTHON_HEADER + content
        
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return True
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def add_header_to_markdown(file_path: Path, dry_run: bool = False) -> bool:
    """Add copyright header to Markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if has_copyright_header(content):
            return False
        
        new_content = MARKDOWN_HEADER + content
        
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return True
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def add_header_to_yaml(file_path: Path, dry_run: bool = False) -> bool:
    """Add copyright header to YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if has_copyright_header(content):
            return False
        
        new_content = YAML_HEADER + content
        
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return True
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def process_repository(repo_path: Path, dry_run: bool = False) -> Tuple[int, int, int]:
    """Process all files in repository."""
    scanned = 0
    modified = 0
    errors = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Remove skip directories from traversal
        dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]
        
        for file in files:
            file_path = Path(root) / file
            
            if should_skip(file_path):
                continue
            
            scanned += 1
            
            # Process based on file type
            if file_path.suffix in PYTHON_EXTENSIONS:
                if add_header_to_python(file_path, dry_run):
                    modified += 1
                    print(f"✓ {'[DRY RUN] ' if dry_run else ''}Modified: {file_path}")
            
            elif file_path.suffix in MARKDOWN_EXTENSIONS:
                # Skip README files (they have custom footers)
                if file_path.name.upper() == 'README.MD':
                    continue
                if add_header_to_markdown(file_path, dry_run):
                    modified += 1
                    print(f"✓ {'[DRY RUN] ' if dry_run else ''}Modified: {file_path}")
            
            elif file_path.suffix in YAML_EXTENSIONS:
                if add_header_to_yaml(file_path, dry_run):
                    modified += 1
                    print(f"✓ {'[DRY RUN] ' if dry_run else ''}Modified: {file_path}")
    
    return scanned, modified, errors

def main():
    parser = argparse.ArgumentParser(
        description='Add copyright headers to source files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--repo',
        type=str,
        default='.',
        help='Repository path (default: current directory)'
    )
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo).resolve()
    
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    print(f"{'=' * 70}")
    print(f"KRL Copyright Header Injection")
    print(f"{'=' * 70}")
    print(f"Repository: {repo_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'=' * 70}\n")
    
    scanned, modified, errors = process_repository(repo_path, args.dry_run)
    
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"Files scanned:        {scanned}")
    print(f"Files modified:       {modified}")
    print(f"Files already compliant: {scanned - modified}")
    print(f"Errors:               {errors}")
    print(f"{'=' * 70}")
    
    if args.dry_run:
        print("\n⚠️  This was a DRY RUN. No files were actually modified.")
        print("Run without --dry-run to apply changes.")
    else:
        print("\n✅ Copyright header injection complete!")

if __name__ == '__main__':
    main()
