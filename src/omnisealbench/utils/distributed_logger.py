# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import sys
from typing import Any, Optional

# Store original print function before any modifications
_original_print = print


def get_global_rank() -> int:
    rank = 0
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    elif 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    
    return rank


class CleanStderr:
    """Custom stderr wrapper that cleans repeated rank prefixes"""
    
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, message):
        # Clean repeated rank prefixes
        if message and isinstance(message, str):
            # Remove repeated [rankN]: patterns
            cleaned = re.sub(r'(\[rank\d+\]:\s*){2,}', '[rank0]: ', message)
            # Also handle cases with spaces
            cleaned = re.sub(r'(\[rank\d+\]\s*){2,}', '[rank0] ', cleaned)
            self.original_stderr.write(cleaned)
        else:
            self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
        
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)


def suppress_stderr_non_rank0():
    """Suppress stderr output for non-rank0 processes to prevent duplicate exceptions"""
    rank = get_global_rank()
    if rank != 0:
        # Redirect stderr to devnull for non-rank-0 processes
        sys.stderr = open(os.devnull, 'w')
    else:
        # For rank 0, wrap stderr to clean repeated rank prefixes
        if not isinstance(sys.stderr, CleanStderr):
            sys.stderr = CleanStderr(sys.stderr)


# Factory functions for easy setup
def setup_clean_exception_handling():
    """Setup clean exception handling for distributed environments"""
    import sys
    import traceback
    
    def clean_exception_hook(exc_type, exc_value, exc_traceback):
        """Custom exception hook that only shows exceptions from rank 0"""
        rank = get_global_rank()

        # Only show exceptions from rank 0
        if rank == 0:
            # Clean any duplicate rank prefixes from the exception message
            exc_str = str(exc_value)
            if '[rank' in exc_str:
                # Remove repeated rank prefixes
                exc_str = re.sub(r'(\[rank\d+\]:\s*){2,}', '[rank0]: ', exc_str)
                exc_str = re.sub(r'(\[rank\d+\]\s*){2,}', '[rank0] ', exc_str)
                # If it starts with rank prefix, clean it up
                exc_str = re.sub(r'^\[rank\d+\]:\s*', '', exc_str)
                exc_value = exc_type(exc_str)
            
            # Use the default exception hook but only from rank 0
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = clean_exception_hook


def setup_rank0_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Setup global logging to only output from rank 0.
    
    Args:
        level: Global logging level
        format_string: Custom format string
        log_file: Optional log file (only written from rank 0)
    """
    # Get rank
    rank = get_global_rank()

    # Only setup logging for rank 0
    if rank == 0:
        
        class CleanFormatter(logging.Formatter):
            """Custom formatter that cleans repeated rank prefixes"""
            def format(self, record):
                msg = super().format(record)
                # Clean repeated rank prefixes in log messages
                if '[rank' in msg:
                    msg = re.sub(r'(\[rank\d+\]:\s*){2,}', '[rank0]: ', msg)
                    msg = re.sub(r'(\[rank\d+\]\s*){2,}', '[rank0] ', msg)
                return msg
        
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        handlers.append(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            handlers.append(file_handler)
        
        # Set format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = CleanFormatter(format_string)
        for handler in handlers:
            handler.setFormatter(formatter)
        
        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=handlers
        )
    else:
        # Disable logging for non-main processes
        logging.disable(logging.CRITICAL)


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    _original_print(*args, **kwargs)


def clean_print(*args: Any, **kwargs: Any) -> None:
    """Print function that cleans repeated rank prefixes from arguments."""
    if args:
        cleaned_args = []
        for arg in args:
            if isinstance(arg, str) and '[rank' in arg:
                cleaned = re.sub(r'(\[rank\d+\]:\s*){2,}', '[rank0]: ', str(arg))
                cleaned = re.sub(r'(\[rank\d+\]\s*){2,}', '[rank0] ', cleaned)
                cleaned_args.append(cleaned)
            else:
                cleaned_args.append(arg)
        _original_print(*cleaned_args, **kwargs)
    else:
        _original_print(*args, **kwargs)


def setup_distributed_logging_and_exceptions():
    """
    Complete setup for clean distributed logging and exception handling.
    Call this once at the beginning of your distributed training script.
    """
    # Setup clean exception handling
    setup_clean_exception_handling()

    # Suppress stderr for non-rank0 processes and clean rank prefixes for rank0
    suppress_stderr_non_rank0()

    # Setup rank0-only logging
    setup_rank0_logging()

    # Additional cleanup for environment variables that might cause rank prefix duplication
    rank = get_global_rank()
    if rank == 0:
        # Disable some torchrun/distributed logging that might cause duplication
        os.environ.setdefault('NCCL_DEBUG', '')
        os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', '')

        # Replace the built-in print with our clean_print function
        import builtins
        builtins.print = clean_print


# helper to log CPU and CUDA memory usage (best-effort)
def log_mem(stage: str) -> None:
    import torch
    import traceback
    try:
        import psutil
    except Exception:  # pragma: no cover - optional dependency
        psutil = None
    try:
        lines = []
        if psutil is not None:
            vm = psutil.virtual_memory()
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / 1024 ** 2
            lines.append(
                f"{stage}: RSS={rss_mb:.1f} MiB, total={vm.total/1024**2:.1f} MiB, avail={vm.available/1024**2:.1f} MiB"
            )
        else:
            lines.append(f"{stage}: psutil not available")

        if torch.cuda.is_available():
            try:
                ndev = torch.cuda.device_count()
                for di in range(ndev):
                    alloc = torch.cuda.memory_allocated(di) / 1024 ** 2
                    reserved = torch.cuda.memory_reserved(di) / 1024 ** 2
                    max_reserved = torch.cuda.max_memory_reserved(di) / 1024 ** 2
                    lines.append(
                        f"GPU{di}: alloc={alloc:.1f} MiB reserved={reserved:.1f} MiB max_reserved={max_reserved:.1f} MiB"
                    )
            except Exception as e:  # pragma: no cover - defensive
                lines.append(f"CUDA mem query failed: {e}")
        else:
            lines.append("CUDA not available")

        # print in rank-zero style so logs are consistent
        rank_zero_print(" | ".join(lines))
    except Exception:
        # never fail the task because profiling failed
        try:
            rank_zero_print(f"log_mem failed at stage={stage}: " + traceback.format_exc())
        except Exception:
            pass
