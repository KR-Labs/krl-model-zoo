#!/usr/bin/env python3
# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Copyright Header Verification Script

Verifies that all source files have proper copyright headers.
Used in CI/CD pipelines to enforce IP protection policies.

Usage:
    python verify_copyright_headers.py [--repo REPO_PATH]

Exit codes:
    0 - All files compliant
    1 - Some files missing headers
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Files and directories to skip
SKIP_PATTERNS = {
    '.git', '.github', '__pycache__', '.pytest_cache', '.venv', 'venv', 
    'node_modules', '.benchmarks', 'htmlcov', '.mypy_cache', 'dist', 'build',
    '.egg-info', 'LICENSE', 'NOTICE', '.DS_Store', '*.pyc', '*.pyo', 
    '*.egg-info', '.coverage', 'coverage.xml', '.gitignore'
}

# File extensions to check
CHECK_EXTENSIONS = {'.py', '.yml', '.yaml'}

def should_skip(file_path: Path) -> bool:
    """Check if file should be skipped."""
    for part in file_path.parts:
        if part in SKIP_PATTERNS:
            return True
    
    if file_path.name in SKIP_PATTERNS:
        return True
    
    if any(file_path.match(pattern) for pattern in SKIP_PATTERNS if '*' in pattern):
        return True
    
    return False

def has_copyright_header(file_path: Path) -> bool:
    """Check if file has copyright header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(500)  # Check first 500 chars
        
        copyright_markers = [
            '© 2025 KR-Labs',
            'Copyright (c) 2025 KR-Labs',
            'KR-Labs™ is a trademark'
        ]
        
        return any(marker in content for marker in copyright_markers)
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
        return False

def verify_repository(repo_path: Path) -> Tuple[List[Path], int]:
    """Verify all files in repository."""
    missing_headers = []
    checked = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Remove skip directories from traversal
        dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]
        
        for file in files:
            file_path = Path(root) / file
            
            if should_skip(file_path):
                continue
            
            if file_path.suffix not in CHECK_EXTENSIONS:
                continue
            
            checked += 1
            
            if not has_copyright_header(file_path):
                missing_headers.append(file_path)
                print(f"❌ Missing header: {file_path}")
    
    return missing_headers, checked

def main():
    parser = argparse.ArgumentParser(
        description='Verify copyright headers in source files'
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
    print(f"KRL Copyright Header Verification")
    print(f"{'=' * 70}")
    print(f"Repository: {repo_path}")
    print(f"{'=' * 70}\n")
    
    missing_headers, checked = verify_repository(repo_path)
    
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"Files checked:        {checked}")
    print(f"Files compliant:      {checked - len(missing_headers)}")
    print(f"Files missing headers: {len(missing_headers)}")
    print(f"{'=' * 70}")
    
    if missing_headers:
        print("\n❌ VERIFICATION FAILED!")
        print(f"\n{len(missing_headers)} file(s) are missing copyright headers.")
        print("\nRun the following command to fix:")
        print("    python scripts/security/add_copyright_headers.py")
        sys.exit(1)
    else:
        print("\n✅ All files have proper copyright headers!")
        sys.exit(0)

if __name__ == '__main__':
    main()
