#!/usr/bin/env python3
"""
Script to fix corruption caused by emoji removal script.

Fixes the following patterns:
- False → alse
- ValueError → Valuerror
- Dict[str, Any] → ict[str, ny]
- Forest → orest
- SPDX-License → SPX-License
- 2025 → 22
- 0.2.0 → .2.
- DataFrame → atarame
- Forecast → orecast
- Implementation → Simplementation
- Any → ny (in type hints)
- Base → ase
- CPUs → PUs
- Args → rgs
- And many more...
"""

import os
import re
from pathlib import Path

# Define all corruption patterns to fix
REPLACEMENTS = [
    # Critical runtime errors
    (r'\balse\b', 'False'),
    (r'\bValuerror\b', 'ValueError'),
    (r'\bRuntimerror\b', 'RuntimeError'),
    (r'\bKeyerror\b', 'KeyError'),
    (r'\bTyperror\b', 'TypeError'),
    (r'\bAttributerror\b', 'AttributeError'),
    (r'\bIndexrror\b', 'IndexError'),
    
    # Type hints - must fix standalone 'ny' before 'Any]' patterns
    (r'\bict\[', 'Dict['),
    (r'Tuple, ny\b', 'Tuple, Any'),
    (r'List, ny\b', 'List, Any'),
    (r': ny\b', ': Any'),
    (r', ny\b', ', Any'),
    (r'\bny\]', 'Any]'),
    (r'\bict\b', 'Dict'),
    (r'str, ny\)', 'str, Any)'),
    (r', ny\]', ', Any]'),
    (r'Optional\[ny\]', 'Optional[Any]'),
    (r'List\[ny\]', 'List[Any]'),
    
    # Class and variable names
    (r'\borest\b', 'Forest'),
    (r'IsolationForestnomalyModel', 'IsolationForestAnomalyModel'),
    (r'Forestnomaly', 'ForestAnomaly'),
    (r'\bnomalyModel', 'AnomalyModel'),
    (r'\batarame', 'DataFrame'),
    (r'\borecast', 'Forecast'),
    (r'\baseModel', 'BaseModel'),
    (r'\bSRIMModel', 'SARIMAModel'),
    
    # License headers
    (r'SPX-License-Identifier', 'SPDX-License-Identifier'),
    (r'Apache-2\.00', 'Apache-2.0'),
    (r'Apache-2\.', 'Apache-2.0'),
    (r'Copyright \(c\) 22 KR-Labs', 'Copyright (c) 2025 KR-Labs'),
    (r'"Apache-2\.00"', '"Apache-2.0"'),
    
    # Version strings
    (r'version="\.\.', 'version="0.1.0'),
    (r'__version__ = "\.2\.-dev"', '__version__ = "0.2.0-dev"'),
    
    # Documentation strings
    (r'\bUseries\b', 'series'),
    (r'Saverage', 'average'),
    (r'\bSimplementation', 'implementation'),
    (r'\bRunusual', 'unusual'),
    (r'\bRunivariate', 'univariate'),
    (r'\bTestimation', 'estimation'),
    (r'\bTextends', 'extends'),
    (r'\bTextension', 'extension'),
    (r'\bxtends', 'extends'),
    (r'\bxtract', 'extract'),
    (r'\bxample', 'example'),
    (r'\bttempt', 'attempt'),
    (r'\bttribute', 'attribute'),
    (r'\bpproximate', 'approximate'),
    (r'\bMapproximate', 'approximate'),
    (r'\bMimplicit', 'implicit'),
    (r'\bMapplicable', 'applicable'),
    (r'alculate', 'calculate'),
    (r'\(2\)', '(2008)'),  # Reference year
    
    # Method/function names
    (r'\brit\b', 'fit'),
    (r'\brgs:', 'Args:'),
    (r'\bompute', 'compute'),
    (r'\bheck', 'check'),
    (r'\build', 'build'),
    (r'\bount', 'count'),
    (r'\betermine', 'determine'),
    (r'\benumerate', 'enumerate'),
    (r'\bMenumerate', 'enumerate'),
    
    # Common words
    (r'\bPUs\b', 'CPUs'),
    (r'\binancial', 'financial'),
    (r'\bUse ases', 'Use cases'),
    (r'\bI\(', 'CI('),  # Confidence Interval
    (r'\bVM\b', 'VECM'),  # Vector Error Correction Model
    (r'\bR\b', 'AR'),  # AutoRegressive
    (r'\bM\b', 'MA'),  # Moving Average
    (r'\.\.', '0.05'),  # Significance level
    (r' \.', ' 0.1'),  # Probability/proportion
    (r'got \{\}', 'got {steps}'),  # Error message
    (r'must be > ', 'must be > 0'),
    # Fix comparisons that lost the colon
    (r'== 0\n', '== 0:\n'),
    (r'!= 0\n', '!= 0:\n'),
    (r'> 0\n', '> 0:\n'),
    (r'>= 0\n', '>= 0:\n'),
    (r'< 0\n', '< 0:\n'),
    (r'<= 0\n', '<= 0:\n'),
    # Now fix standalone comparisons with space
    (r'== :', '== 0'),  # Comparison with zero
    (r'!= :', '!= 0'),
    (r'> :', '> 0'),
    (r'>= :', '>= 0'),
    (r'< :', '< 0'),
    (r'<= :', '<= 0'),
    (r'if steps <= ', 'if steps <= 0'),
    (r'seasonal_period == ', 'seasonal_period == 0'),
    (r'== -', '== -1'),  # Anomaly detection
    (r'\[\]\.', '[0].'),  # Array indexing
    (r'shape\[\]', 'shape[1]'),
    (r'columns\[\]', 'columns[1]'),
    (r'crit_vals\[2\]', 'crit_vals[2]'),
    (r' \* ', ' * 100'),  # Percentage calculation
    (r'Year', '2025'),  # Copyright year
    (r'll rights', 'All rights'),
    (r'LL,', 'LLC,'),
    
    # Pandas/library specific
    (r'pd\.atarame', 'pd.DataFrame'),
    (r'onfidence', 'confidence'),
    (r'ugmented', 'Augmented'),
    (r'ickey-uller', 'Dickey-Fuller'),
    (r'djustment', 'adjustment'),
    
    # Statistical terms
    (r'\bI\(\)', 'I(1)'),  # Integrated of order 1
    (r'I\(\) Useries', 'I(1) series'),
    
    # Misc corrections
    (r'ach column', 'Each column'),
    (r'ased on', 'Based on'),
    (r'ritical values', 'critical values'),
    (r'aily data', 'Daily data'),
    (r'Toolean', 'boolean'),
]

def fix_file(filepath):
    """Fix corruption in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all replacements
        for pattern, replacement in REPLACEMENTS:
            content = re.sub(pattern, replacement, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix all Python files in krl_models directory."""
    base_dir = Path(__file__).parent
    krl_models_dir = base_dir / 'krl_models'
    src_krl_models_dir = base_dir / 'src' / 'krl_models'
    
    files_fixed = 0
    total_files = 0
    
    # Fix files in krl_models/
    for py_file in krl_models_dir.rglob('*.py'):
        total_files += 1
        if fix_file(py_file):
            files_fixed += 1
            print(f"Fixed: {py_file.relative_to(base_dir)}")
    
    # Fix files in src/krl_models/ if it exists
    if src_krl_models_dir.exists():
        for py_file in src_krl_models_dir.rglob('*.py'):
            total_files += 1
            if fix_file(py_file):
                files_fixed += 1
                print(f"Fixed: {py_file.relative_to(base_dir)}")
    
    print(f"\n✅ Fixed {files_fixed} out of {total_files} files")

if __name__ == '__main__':
    main()
