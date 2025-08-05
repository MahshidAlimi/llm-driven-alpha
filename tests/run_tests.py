#!/usr/bin/env python3
"""
Test runner for the Fixed Income Trading System
"""

import unittest
import sys
import os
import argparse

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """Run all tests"""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test(test_module):
    """Run a specific test module"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{test_module}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test_class(test_module, test_class):
    """Run a specific test class"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{test_module}.{test_class}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    parser = argparse.ArgumentParser(description='Run tests for Fixed Income Trading System')
    parser.add_argument('--module', '-m', help='Run specific test module (e.g., test_universe_selection)')
    parser.add_argument('--class', '-c', dest='test_class', help='Run specific test class')
    parser.add_argument('--all', '-a', action='store_true', help='Run all tests (default)')
    
    args = parser.parse_args()
    
    if args.module:
        if args.test_class:
            success = run_specific_test_class(args.module, args.test_class)
        else:
            success = run_specific_test(args.module)
    else:
        success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 