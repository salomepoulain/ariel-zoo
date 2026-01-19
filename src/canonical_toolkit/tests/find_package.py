import sys

# 1. We define a simple "wrapper" for the import system
original_import = __import__

def noisy_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'yaml' or name.startswith('yaml.'):
        # We look at the stack to see who called the import
        import traceback
        stack = traceback.extract_stack()
        # The second to last entry is the file that called 'import torch'
        caller = stack[-2]
        print(f"\n[yaml] Imported by: {caller.filename} at line {caller.lineno}")
    return original_import(name, globals, locals, fromlist, level)

# 2. Patch the built-in import
import builtins
builtins.__import__ = noisy_import

# 3. Import your toolkit to trigger the search
print("Scanning canonical_toolkit...")
import canonical_toolkit
