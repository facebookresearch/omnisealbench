# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
from collections import OrderedDict

# Global shared LRU cache
GLOBAL_LRU_CACHE = OrderedDict()

def global_lru_cache(maxsize=64):
    """
    Decorator for LRU cache shared across all decorated functions.
    Usage:
        @global_lru_cache(maxsize=32)
        def foo(x): ...
        @global_lru_cache(maxsize=32)
        def bar(x): ...
        # foo and bar share the same cache
        foo.global_cache  # access the shared cache
        foo.cache_clear()  # clear the shared cache
    """
    def decorator(func):
        cache = GLOBAL_LRU_CACHE

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Key includes function name to avoid collisions
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        def cache_clear():
            cache.clear()

        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator


def clear_global_cache():
    """
    Clear the global LRU cache.
    """
    GLOBAL_LRU_CACHE.clear()
