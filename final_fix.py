#!/usr/bin/env python3
"""
Final comprehensive corruption fix script.
"""

import re
from pathlib import Path

def comprehensive_fix(filepath):
    """Apply all corruption fixes to a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix array/tuple indexing
    content = re.sub(r'x\[\]', 'x[1]', content)
    content = re.sub(r'\[\]:', '[0]:', content)
    content = re.sub(r'\[\]\)', '[0])', content)
    content = re.sub(r'\[\],', '[0],', content)
    content = re.sub(r'\[\]\.', '[0].', content)
    content = re.sub(r'key=lambda x: x\[\]', 'key=lambda x: x[1]', content)
    
    # Fix return type annotations  
    content = re.sub(r'-> 0 (\w+):', r'-> \1:', content)
    
    # Fix function parameters with missing defaults
    content = re.sub(r'n: int = \)', 'n: int = 10)', content)
    content = re.sub(r'= \):', '= 0):', content)
    
    # Fix if/comparison statements missing values
    content = re.sub(r'if ([a-z_]+) > \b', r'if \1 > 0', content)
    content = re.sub(r'if ([a-z_]+) >= \b', r'if \1 >= 0', content)
    content = re.sub(r'if ([a-z_]+) < \b', r'if \1 < 0', content)
    content = re.sub(r'if ([a-z_]+) <= \b', r'if \1 <= 0', content)
    content = re.sub(r'if ([a-z_]+) == \b', r'if \1 == 0', content)
    content = re.sub(r'if ([a-z_]+) != \b', r'if \1 != 0', content)
    content = re.sub(r'is \b(?!None)', 'is 0', content)
    content = re.sub(r'else \b$', 'else 0', content, flags=re.MULTILINE)
    
    # Fix comparison in middle of line
    content = re.sub(r'([a-z_]+) > \b', r'\1 > 0', content)
    content = re.sub(r'([a-z_]+) >= \b', r'\1 >= 0', content)
    content = re.sub(r'([a-z_]+) < \b', r'\1 < 0', content)
    content = re.sub(r'([a-z_]+) <= \b', r'\1 <= 0', content)
    content = re.sub(r'([a-z_]+) == \b', r'\1 == 0', content)
    content = re.sub(r'([a-z_]+) != \b', r'\1 != 0', content)
    
    # Fix corrupted numbers in expressions
    content = re.sub(r'\.shape\[\]', '.shape[1]', content)
    content = re.sub(r'\.columns\[\]', '.columns[1]', content)
    content = re.sub(r'\(\)', '(0)', content)
    content = re.sub(r', \)', ', 0)', content)
    
    # Fix special patterns
    content = re.sub(r'mcmc_samples, \)', 'mcmc_samples, 0)', content)
    content = re.sub(r'100100100', '0.5 * ', content)  # Complex multiplication corruption
    content = re.sub(r'100100', '0.5 * ', content)
    content = re.sub(r'cccconfidence', 'confidence', content)
    content = re.sub(r'ccconfidence', 'confidence', content)
    
    # Only write if changes made
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all Python files."""
    fixed = 0
    total = 0
    
    for py_file in Path('krl_models').rglob('*.py'):
        total += 1
        if comprehensive_fix(py_file):
            fixed += 1
            print(f"Fixed: {py_file}")
    
    for py_file in Path('src/krl_models').rglob('*.py'):
        total += 1
        if comprehensive_fix(py_file):
            fixed += 1
            print(f"Fixed: {py_file}")
    
    print(f"\n✅ Fixed {fixed} out of {total} files")

if __name__ == '__main__':
    main()
