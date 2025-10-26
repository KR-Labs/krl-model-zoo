#!/usr/bin/env python3
# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Build Watermark Injection Script

Injects unique build metadata into packages for tracking and security.
Used in CI/CD pipelines for package builds.

Usage:
    python inject_watermark.py --build-id BUILD_ID --commit-sha COMMIT_SHA
                                --repo REPO_NAME [--compute-checksum]

The watermark includes:
- Unique build ID
- Build timestamp
- Git commit SHA
- Repository name
- Version number
- Optional: Package checksum
"""

import os
import sys
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

def compute_package_checksum(package_dir: Path) -> str:
    """Compute SHA256 checksum of package contents."""
    hasher = hashlib.sha256()
    
    for root, dirs, files in os.walk(package_dir):
        # Skip cache and metadata directories
        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.pytest_cache', '*.egg-info'}]
        
        for file in sorted(files):  # Sort for deterministic hashing
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
                except Exception as e:
                    print(f"⚠️  Warning: Could not read {file_path}: {e}")
    
    return hasher.hexdigest()

def inject_watermark(
    init_file: Path,
    build_id: str,
    commit_sha: str,
    repo: str,
    version: str,
    checksum: Optional[str] = None
) -> bool:
    """Inject watermark into __init__.py file."""
    try:
        # Read existing content
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Build watermark
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        watermark = f'''
# Build Watermark (Auto-generated - DO NOT EDIT)
__watermark__ = {{
    "build_id": "{build_id}",
    "build_timestamp": "{timestamp}",
    "commit_sha": "{commit_sha}",
    "repository": "{repo}",
    "version": "{version}",
'''
        
        if checksum:
            watermark += f'    "checksum": "sha256:{checksum}",\n'
        
        watermark += '}\n'
        
        # Check if watermark already exists
        if '__watermark__' in content:
            # Replace existing watermark
            start = content.find('# Build Watermark')
            end = content.find('}', start) + 1
            if start != -1 and end != 0:
                content = content[:start] + watermark + content[end+1:]
            else:
                print("⚠️  Warning: Found __watermark__ but couldn't locate boundaries")
                content += '\n' + watermark
        else:
            # Append watermark
            content += '\n' + watermark
        
        # Write back
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"❌ Error injecting watermark: {e}")
        return False

def find_init_file(repo_path: Path) -> Optional[Path]:
    """Find main __init__.py file in repository."""
    # Common locations
    possible_paths = [
        repo_path / 'krl_core' / '__init__.py',
        repo_path / 'krl_models' / '__init__.py',
        repo_path / 'src' / '__init__.py',
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Search for any __init__.py in src-like directories
    for pattern in ['src/*/__init__.py', 'krl_*/__init__.py']:
        results = list(repo_path.glob(pattern))
        if results:
            return results[0]
    
    return None

def get_version(repo_path: Path) -> str:
    """Extract version from pyproject.toml or setup.py."""
    # Try pyproject.toml first
    pyproject = repo_path / 'pyproject.toml'
    if pyproject.exists():
        try:
            with open(pyproject, 'r') as f:
                for line in f:
                    if line.strip().startswith('version'):
                        # Extract version from: version = "1.0.0"
                        return line.split('=')[1].strip().strip('"\'')
        except Exception:
            pass
    
    # Fallback to __init__.py
    init_file = find_init_file(repo_path)
    if init_file:
        try:
            with open(init_file, 'r') as f:
                for line in f:
                    if line.strip().startswith('__version__'):
                        return line.split('=')[1].strip().strip('"\'')
        except Exception:
            pass
    
    return '0.0.0'

def main():
    parser = argparse.ArgumentParser(
        description='Inject build watermark into package'
    )
    parser.add_argument(
        '--build-id',
        required=True,
        help='Unique build identifier'
    )
    parser.add_argument(
        '--commit-sha',
        required=True,
        help='Git commit SHA'
    )
    parser.add_argument(
        '--repo',
        required=True,
        help='Repository name (e.g., KR-Labs/krl-model-zoo)'
    )
    parser.add_argument(
        '--compute-checksum',
        action='store_true',
        help='Compute and include package checksum'
    )
    parser.add_argument(
        '--repo-path',
        type=str,
        default='.',
        help='Repository path (default: current directory)'
    )
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path).resolve()
    
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    print(f"{'=' * 70}")
    print(f"KRL Build Watermark Injection")
    print(f"{'=' * 70}")
    print(f"Repository:   {args.repo}")
    print(f"Build ID:     {args.build_id}")
    print(f"Commit SHA:   {args.commit_sha}")
    print(f"{'=' * 70}\n")
    
    # Find __init__.py
    init_file = find_init_file(repo_path)
    if not init_file:
        print("❌ Could not find __init__.py file!")
        sys.exit(1)
    
    print(f"Found init file: {init_file}")
    
    # Get version
    version = get_version(repo_path)
    print(f"Package version: {version}")
    
    # Compute checksum if requested
    checksum = None
    if args.compute_checksum:
        print("Computing package checksum...")
        checksum = compute_package_checksum(init_file.parent)
        print(f"Checksum: sha256:{checksum}")
    
    # Inject watermark
    print("\nInjecting watermark...")
    success = inject_watermark(
        init_file,
        args.build_id,
        args.commit_sha,
        args.repo,
        version,
        checksum
    )
    
    if success:
        print("\n✅ Watermark injection complete!")
        print(f"\nWatermark added to: {init_file}")
    else:
        print("\n❌ Watermark injection failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
