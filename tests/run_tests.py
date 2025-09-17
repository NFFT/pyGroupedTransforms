import os
import subprocess
import sys


# Script to run all tests and return any failures
def run_tests():
    test_files = [
        "tests/nfft_U.py",
        "tests/nfft_ds.py",
        "tests/nfct_U.py",
        "tests/nfct_ds.py",
        "tests/cwwt_U.py",
        "tests/cwwt_ds.py",
    ]

    for test_file in test_files:
        print("Testing ", test_file, "...")
        result = subprocess.run(
            [sys.executable, test_file], capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("Test failed for ", test_file)
            print(result.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    run_tests()
