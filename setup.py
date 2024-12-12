# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import find_packages, setup

NAME = "omnisealbench"
DESCRIPTION = "Watermarking benchmark evaluation tools for PyTorch"

URL = "https://github.com/facebookresearch/omnisealbench"
AUTHOR = "FAIR Speech & Audio"
EMAIL = "hadyelsahar@meta.com, valeriu@meta.com"
REQUIRES_PYTHON = ">=3.8.0"

for line in open(f"{NAME}/__init__.py"):
    line = line.strip()
    if "__version__" in line:
        context = {}
        exec(line, context)
        VERSION = context["__version__"]

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

REQUIRED = [i.strip() for i in open(HERE / "requirements.txt") if not i.startswith("#")]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author_email=EMAIL,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require={
        "dev": ["coverage", "flake8", "mypy", "pdoc3", "pytest"],
    },
    packages=find_packages(),
    package_data={
        "omnisealbench": [
            "py.typed",
            "assets/*.png",
            "configs/*.yaml",
            "attacks/*.yaml",
            "cards/audio/*.yaml",
            "cards/image/*.yaml",
            "datasets/audio/*.yaml",
            "datasets/image/*.yaml",
        ]
    },
    include_package_data=True,
    license="MIT License",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Image",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Watermarking",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "omnisealbench.evaluate=omnisealbench.evaluate:main",
            "omnisealbench.watermarkgen=omnisealbench.watermarkgen:main",
        ],
    },
)
