# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def check_lib(module_name: str):
    """
    Attempts to import a module by name and returns it if found.
    If the module is not found, it returns None.
    
    Args:
        module_name (str): The name of the module to import.
        
    Returns:
        Module or None: The imported module or None if not found.
    """
    try:
        __import__(module_name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False
