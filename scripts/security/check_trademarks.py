#!/usr/bin/env python3
# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Trademark Notice Verification Script

Verifies that README files contain proper trademark notices.
Used in CI/CD pipelines to enforce IP protection policies.

Usage:
    python check_trademarks.py [--repo REPO_PATH]

Exit codes:
    0 - All README files compliant
    1 - Some README files missing trademark notices
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

def has_trademark_notice(file_path: Path) -> bool:
    """Check if README has trademark notice."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for trademark markers
        trademark_markers = [
            'KR-Labs™',
            '© 2025 KR-Labs',
            'Quipu Research Labs, LLC',
            'Sudiata Giddasira, Inc.'
        ]
        
        # All markers should be present
        return all(marker in content for marker in trademark_markers)
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
        return False

def verify_readme_files(repo_path: Path) -> Tuple[List[Path], List[Path]]:
    """Verify all README files in repository."""
    compliant = []
    non_compliant = []
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden and build directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build'}]
        
        for file in files:
            if file.upper() == 'README.MD':
                file_path = Path(root) / file
                
                if has_trademark_notice(file_path):
                    compliant.append(file_path)
                    print(f"✅ {file_path}")
                else:
                    non_compliant.append(file_path)
                    print(f"❌ {file_path}")
    
    return compliant, non_compliant

def main():
    parser = argparse.ArgumentParser(
        description='Verify trademark notices in README files'
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
    print(f"KRL Trademark Notice Verification")
    print(f"{'=' * 70}")
    print(f"Repository: {repo_path}")
    print(f"{'=' * 70}\n")
    
    compliant, non_compliant = verify_readme_files(repo_path)
    
    total = len(compliant) + len(non_compliant)
    
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"README files found:   {total}")
    print(f"Compliant:            {len(compliant)}")
    print(f"Non-compliant:        {len(non_compliant)}")
    print(f"{'=' * 70}")
    
    if non_compliant:
        print("\n❌ VERIFICATION FAILED!")
        print(f"\n{len(non_compliant)} README file(s) are missing trademark notices.")
        print("\nAdd the following section to the bottom of each README:")
        print("""
---

## Trademark Notice

**KR-Labs™** is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.

---

© 2025 KR-Labs. All rights reserved.
""")
        sys.exit(1)
    else:
        print("\n✅ All README files have proper trademark notices!")
        sys.exit(0)

if __name__ == '__main__':
    main()
