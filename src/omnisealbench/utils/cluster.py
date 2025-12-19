# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
import socket
from enum import Enum


# from https://github.com/fairinternal/lavida/blob/main/large_vision_dataset/configs/config.py#L9-L35
class ClusterType(Enum):
    FAIR = "fair"
    AWS = "aws"  # TODO: THE FULL NAME is A100, consider switching to it
    AWS_H100_2 = "aws-h100-2"
    GCP = "gcp"
    RSC = "rsc"
    DEVFB = "devfb"
    UNKNOWN = "unknown"


@functools.lru_cache
def guess_cluster():
    # see also https://github.com/fairinternal/fairseq2-ext/blob/main/src/fairseq2_ext/__init__.py#L24
    uname = os.uname()
    if uname.sysname == "Linux":
        if uname.release.endswith("-aws"):
            dns_domain = socket.getfqdn().split(".", 1)[1]
            # Linux kernel versions on AWS instances are of the form "5.4.0-1051-aws"
            if "fair-aws-h100-2" in dns_domain:
                return ClusterType.AWS_H100_2
            return ClusterType.AWS
        elif uname.release.endswith("-gcp"):
            # Linux kernel versions on GCP instances are of the form "5.13.0-1033-gcp"
            return ClusterType.GCP
        elif uname.nodename.startswith("rsc"):
            # Linux kernel versions on RSC instances are standard ones but hostnames start with "rsc"
            return ClusterType.RSC
        elif uname.nodename.endswith("book.com"):
            return ClusterType.DEVFB

    # we are probably on fair cluster then and FAIR_ENV_CLUSTER should be h1 or h2
    if "FAIR_ENV_CLUSTER" in os.environ:
        return ClusterType.FAIR
    else:
        return ClusterType.UNKNOWN
