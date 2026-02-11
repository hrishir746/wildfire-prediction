#!/usr/bin/env python3
"""
SAFE TRAINING WRAPPER - UTF-8 Encoding Enforcer

This wrapper script ensures UTF-8 encoding is properly set BEFORE any other
code runs, preventing UnicodeEncodeError in Rovo Dev extension child processes.

Usage:
    python scripts/run_training_safe.py train_extended_gpu
    python scripts/run_training_safe.py train_with_dashboard --epochs 50
    python scripts/run_training_safe.py validate_extended_model
    python scripts/run_training_safe.py test_full_pipeline_512

The wrapper:
1. Forces UTF-8 encoding at the OS level
2. Reconfigures stdout/stderr to UTF-8
3. Sets environment variables
4. Imports and runs the target script
"""

# ============================================================================
# CRITICAL: UTF-8 ENCODING MUST BE SET FIRST, BEFORE ANY OTHER IMPORTS
# ============================================================================
import sys
import os

# Force UTF-8 encoding for stdout and stderr
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Set environment variables for child processes
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# For Windows console
if sys.platform == 'win32':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass

# ============================================================================
# NOW SAFE TO IMPORT OTHER MODULES
# ============================================================================
from pathlib import Path
import importlib.util
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Available scripts
AVAILABLE_SCRIPTS = {
    # Training scripts
    'train_extended_gpu': 'scripts/train_extended_gpu.py',
    'train_gpu_optimized': 'scripts/train_gpu_optimized.py',
    'train_fast_production': 'scripts/train_fast_production.py',
    'train_model': 'scripts/train_model.py',
    'train_with_dashboard': 'scripts/train_with_dashboard.py',
    'finetune_gpu': 'scripts/finetune_gpu.py',
    
    # Validation scripts
    'validate_80_percent': 'scripts/validate_80_percent.py',
    'validate_extended_model': 'scripts/validate_extended_model.py',
    'validate_camp_2018': 'scripts/validate_camp_2018.py',
    'wildfire_validation': 'scripts/wildfire_validation.py',
    
    # Evaluation scripts
    'evaluate_real_fires': 'scripts/evaluate_real_fires.py',
    'evaluate_model': 'scripts/evaluate_model.py',
    'quick_eval_gpu': 'scripts/quick_eval_gpu.py',
    
    # Demo scripts
    'comprehensive_demo': 'scripts/comprehensive_demo.py',
    'quick_demo': 'scripts/quick_demo.py',
    
    # Test scripts
    'test_full_pipeline_512': 'scripts/test_full_pipeline_512.py',
}

def print_safe(msg):
    """Print with guaranteed UTF-8 encoding and error handling"""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Fallback: replace problematic characters
        print(msg.encode('ascii', 'replace').decode('ascii'))

def run_script(script_name, args):
    """Run a script with proper UTF-8 encoding"""
    if script_name not in AVAILABLE_SCRIPTS:
        print_safe(f"ERROR: Unknown script '{script_name}'")
        print_safe(f"\nAvailable scripts:")
        for name in sorted(AVAILABLE_SCRIPTS.keys()):
            print_safe(f"  - {name}")
        return 1
    
    script_path = ROOT / AVAILABLE_SCRIPTS[script_name]
    
    if not script_path.exists():
        print_safe(f"ERROR: Script not found: {script_path}")
        return 1
    
    print_safe("=" * 70)
    print_safe(f"SAFE WRAPPER: Running {script_name}")
    print_safe("=" * 70)
    print_safe(f"Script: {script_path}")
    print_safe(f"UTF-8 Encoding: ENABLED")
    print_safe(f"Args: {args}")
    print_safe("=" * 70)
    print_safe("")
    
    # Modify sys.argv to pass arguments to the target script
    original_argv = sys.argv
    sys.argv = [str(script_path)] + args
    
    try:
        # Load and execute the target script
        spec = importlib.util.spec_from_file_location("__main__", script_path)
        module = importlib.util.module_from_spec(spec)
        
        # Execute the script
        spec.loader.exec_module(module)
        
        return 0
    except Exception as e:
        print_safe(f"\nERROR during script execution:")
        print_safe(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original argv
        sys.argv = original_argv

def main():
    parser = argparse.ArgumentParser(
        description="Safe wrapper for running training scripts with UTF-8 encoding",
        usage="%(prog)s SCRIPT [SCRIPT_ARGS...]"
    )
    parser.add_argument('script', help='Script to run (without .py extension)')
    parser.add_argument('script_args', nargs='*', help='Arguments to pass to the script')
    
    # Parse only the first argument to get the script name
    if len(sys.argv) < 2:
        parser.print_help()
        print_safe("\n" + "=" * 70)
        print_safe("Available scripts:")
        print_safe("=" * 70)
        for category, scripts in [
            ("Training", ['train_extended_gpu', 'train_gpu_optimized', 'train_fast_production', 
                         'train_model', 'train_with_dashboard', 'finetune_gpu']),
            ("Validation", ['validate_80_percent', 'validate_extended_model', 
                           'validate_camp_2018', 'wildfire_validation']),
            ("Evaluation", ['evaluate_real_fires', 'evaluate_model', 'quick_eval_gpu']),
            ("Demo", ['comprehensive_demo', 'quick_demo']),
            ("Test", ['test_full_pipeline_512']),
        ]:
            print_safe(f"\n{category}:")
            for script in scripts:
                print_safe(f"  - {script}")
        print_safe("\n" + "=" * 70)
        return 1
    
    script_name = sys.argv[1]
    script_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    return run_script(script_name, script_args)

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
