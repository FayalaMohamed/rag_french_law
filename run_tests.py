"""
Test runner script for French Legal RAG system.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Run with verbose output
    python run_tests.py --pattern test_text_processing  # Run specific test file

Requires pytest. Install with: pip install pytest
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def find_test_files(test_dir: str = "tests") -> list:
    """Find all test files in the tests directory."""
    test_files = []
    tests_path = Path(test_dir)
    
    if not tests_path.exists():
        print(f"Error: Test directory '{test_dir}' not found")
        return []
    
    for file in tests_path.glob("test_*.py"):
        test_files.append(file.name)
    
    return sorted(test_files)


def run_with_pytest(test_pattern: str = None, verbose: bool = False):
    """Run tests using pytest."""
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if test_pattern:
        cmd.append("-k")
        cmd.append(test_pattern)
    
    print("=" * 60)
    print("Running tests with pytest")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Install it with: pip install pytest")
        return 1


def run_with_unittest(test_file: str = None, verbose: bool = False):
    """Run tests using Python's built-in unittest."""
    import unittest
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("=" * 60)
    print("Running tests with unittest")
    print("=" * 60)
    
    if test_file:
        # Run specific test file
        test_name = test_file.replace(".py", "").replace("test_", "")
        test_module = f"tests.{test_file.replace('.py', '')}"
        
        try:
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(test_module)
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
            return 0 if result.wasSuccessful() else 1
        except Exception as e:
            print(f"Error loading test module: {e}")
            return 1
    else:
        # Run all test files
        test_files = find_test_files()
        
        if not test_files:
            print("No test files found")
            return 1
        
        print(f"\nFound {len(test_files)} test file(s):")
        for f in test_files:
            print(f"  - {f}")
        print()
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_file in test_files:
            test_module = f"tests.{test_file.replace('.py', '')}"
            try:
                tests = loader.loadTestsFromName(test_module)
                suite.addTests(tests)
            except Exception as e:
                print(f"Warning: Could not load {test_file}: {e}")
        
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for French Legal RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py -v                 # Run all tests (verbose)
    python run_tests.py -u                 # Use unittest instead of pytest
    python run_tests.py -p text_processing # Run specific test file pattern
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    parser.add_argument(
        "-u", "--unittest",
        action="store_true",
        help="Use Python's built-in unittest instead of pytest"
    )
    
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        help="Run tests matching pattern (e.g., 'text_processing')"
    )
    
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available test files and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        test_files = find_test_files()
        print("Available test files:")
        for f in test_files:
            print(f"  - {f}")
        return 0
    
    # Check if running from correct directory
    if not os.path.exists("tests"):
        print("Error: Must run from project root directory")
        print("Current directory:", os.getcwd())
        return 1
    
    if args.unittest:
        return run_with_unittest(args.pattern, args.verbose)
    else:
        return run_with_pytest(args.pattern, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
