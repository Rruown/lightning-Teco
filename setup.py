#!/usr/bin/env python
import glob
import os
from pathlib import Path

from pkg_resources import parse_requirements
from setuptools import setup, find_packages


_PATH_ROOT = os.path.dirname(__file__)

__version__ = "1.8.6"
__author__ = "Tecorigin"
__author_email__ = "hz@tecorigin.com"
__license__ = "BSD 3-clause"
__copyright__ = f"Copyright (c) 2020-2024, {__author__}."
__homepage__ = "https://github.com/Rruown/lightning-Teco.git"
__docs__ = "Lightning support for tecorigin's sdaa accelerators"
__min_required_version__ = __version__


def _load_requirements(path_dir: str = _PATH_ROOT, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(
        open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fopen:
    readme = fopen.read()


def _prepare_extras(requirements_dir: str = _PATH_ROOT) -> dict:
    req_files = [Path(p) for p in glob.glob(
        os.path.join(requirements_dir, "*.txt"))]
    extras = {
        p.stem: _load_requirements(file_name=p.name, path_dir=str(p.parent))
        for p in req_files
        if not p.name.startswith("_")
    }
    # todo: eventually add some custom aggregations such as `develop`
    extras = {name: sorted(set(reqs)) for name, reqs in extras.items()}
    print("The extras are: ", extras)
    return extras


setup(
    name="lightning-teco",
    version=__version__,
    description=__docs__,
    author=__author__,
    author_email=__author_email__,
    url=__homepage__,
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    download_url="https://github.com/Rruown/lightning-Teco.git",
    license=__license__,
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.10",
    setup_requires=["wheel"],
    install_requires=_load_requirements(),
    extras_require=_prepare_extras(),
)
