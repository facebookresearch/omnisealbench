
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
import importlib
import io
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import time

from omnisealbench.config import default_omniseal_cache


@contextlib.contextmanager
def start_logging():
    """
    A utility function to start logging in a distributed-safe and context-aware manner.
    If OMNISEAL_DEBUG is set to 1, the logging logic is following:
    - If inside a SLURM job, log to the SLURM job's stdout file.
    - Otherwise, log to a file in OMNISEAL_HOME/logs with a timestamp
    If OMNISEAL_DEBUG is not set or set to 0, do nothing.
    """
    debug_mode = os.getenv("OMNISEAL_DEBUG", "0") == "1"
    should_close = True
    if not debug_mode:
        fd = os.open(os.devnull, os.O_WRONLY)
    else:
        omniseal_home_dir = os.getenv("OMNISEAL_HOME", None)
        if omniseal_home_dir is not None:
            omniseal_home_dir = os.path.expanduser(omniseal_home_dir)

        log_file = None

        # in slurm job, we flush the logs to slurm job's stdout
        if "SLURM_JOB_ID" in os.environ:
            job_id = os.environ.get("SLURM_JOB_ID")
            cmd = f"scontrol show job {job_id}"
            out = subprocess.check_output(shlex.split(cmd), text=True)
            stdout_path = None
            stderr_path = None
            for token in out.split():
                if token.startswith("StdOut="):
                    stdout_path = token.split("=", 1)[1]
                elif token.startswith("StdErr="):
                    stderr_path = token.split("=", 1)[1]
            log_file = stdout_path or stderr_path

        # In local run, we log to OMNISEAL_HOME/logs/omniseal_<timestamp>.log
        if log_file is None and omniseal_home_dir is not None:
            log_dir = os.path.join(omniseal_home_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)

            dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = os.path.join(log_dir, f"omniseal_{dt}.log")
        
        if log_file is not None:
            fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        else:
            try:
                fd = sys.stdout.fileno()
                should_close = False
            except io.UnsupportedOperation:
                fd = os.open(os.devnull, os.O_WRONLY)
    yield fd
    if should_close:
        os.close(fd)


def install_lib(package):
    """Install a package using pip in a distributed-safe manner.
    Args:
        package (str): The package name to install.
    """
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    shared_dir = os.path.expanduser(os.environ.get("OMNISEAL_HOME", default_omniseal_cache + "/pip_cache"))
    os.makedirs(shared_dir, exist_ok=True)
    sentinel = Path(shared_dir) / f".install_done_{package.replace('/', '_')}"
    
    try:
        if importlib.util.find_spec(package) is None:
            # Install the package only on rank 0
            while not sentinel.exists():
                if rank == 0:
                    with start_logging() as fd:
                        os.write(fd, f"Installing package {package}...\n".encode())
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", package],
                            stdout=fd, stderr=fd
                        )
                    sentinel.touch()
                else:
                    # Other ranks wait for the installation to complete, but with a timeout
                    timeout = 300  # seconds (5 minutes)
                    start_time = time.monotonic()
                    if time.monotonic() - start_time > timeout:
                        raise RuntimeError(
                            f"Timeout waiting for package '{package}' to be installed by rank 0."
                        )
                    time.sleep(0.5)
        elif rank == 0:
            with start_logging() as fd:
                os.write(fd, f"Package {package} is already installed.\n".encode())
    finally:
        pass  # Do not delete the sentinel file here; let it persist for synchronization
