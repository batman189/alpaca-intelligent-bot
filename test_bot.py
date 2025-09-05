#!/usr/bin/env python3
import os
import sys
sys.path.append('.')

from app import run_test_mode

if __name__ == "__main__":
    print("Running comprehensive bot test...")
    success = run_test_mode()
    sys.exit(0 if success else 1)
