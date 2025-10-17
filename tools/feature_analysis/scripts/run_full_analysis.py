"""
Master Script: Run Full Feature Analysis Pipeline

Executes all analysis scripts in sequence and generates comprehensive outputs.
"""

import subprocess
import sys
from pathlib import Path

scripts_dir = Path(__file__).parent

print('=' * 80)
print('FEATURE ANALYSIS PIPELINE')
print('=' * 80)
print()

# List of scripts to run in sequence
analysis_scripts = [
    ('spectral_profiling.py', 'Comprehensive Spectral Profiling'),
    ('transition_analysis.py', 'Transition Zone Analysis'),
    ('visualization_suite.py', 'Visualization Generation'),
    ('generate_dummy_spectra.py', 'Dummy Spectra Generation'),
]

# Run each script
for script_name, description in analysis_scripts:
    print(f'\n{"=" * 80}')
    print(f'Running: {description}')
    print(f'Script: {script_name}')
    print('=' * 80)
    print()

    script_path = scripts_dir / script_name

    if not script_path.exists():
        print(f'ERROR: Script not found: {script_path}')
        continue

    # Run the script
    result = subprocess.run([sys.executable, str(script_path)],
                          capture_output=False)

    if result.returncode != 0:
        print(f'\nWARNING: Script exited with code {result.returncode}')
    else:
        print(f'\n[OK] {description} completed successfully')

print('\n' + '=' * 80)
print('PIPELINE COMPLETE')
print('=' * 80)
print()
print('Outputs saved to: feature_analysis/outputs/')
print('  - statistics/: CSV files with numerical results')
print('  - visualizations/: PNG plots (to be generated)')
print('  - reports/: Summary documentation (to be generated)')
print()
print('Next steps:')
print('  1. Review outputs/statistics/*.csv')
print('  2. Review outputs/visualizations/*.png')
print('  3. Read README.md and IMPROVEMENT_PLAN.md')
print('  4. Implement algorithm improvements in ../spectral_classifier/')
