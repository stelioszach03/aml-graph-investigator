import os
import sys

# Ensure project root is on the import path for `import app.*`
ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

