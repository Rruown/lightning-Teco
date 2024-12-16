#!/usr/bin/env python
import glob
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


_MODULE_NAME = "lightning_teco"
_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRES = _PATH_ROOT


def _load_py_module(fname, pkg=_MODULE_NAME):
    spec = spec_from_file_location(os.path.join(
        pkg, fname), os.path.join(_PATH_ROOT, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module('__about__.py')


def _load_requirements(path_dir: str = _PATH_REQUIRES, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(
        open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fopen:
    readme = fopen.read()


def _prepare_extras(requirements_dir: str = _PATH_REQUIRES) -> dict:
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
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url="https://github.com/Lightning-AI/lightning-teco",
    license=about.__license__,
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.10",
    setup_requires=["wheel"],
    install_requires=_load_requirements(),
    extras_require=_prepare_extras(),
)
