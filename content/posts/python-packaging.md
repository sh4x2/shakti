---
title: Python Packaging
summary: Understanding `pyproject.toml`, distribution, build frontend and backends
date: 2023-07-28
draft: true
---

Some time ago I started seeing `pyproject.toml` in almost every Python package and also tools like [Poetry](https://python-poetry.org/) getting common. So, I started using it too but wondered what happened to `setup.py` and simple old `pip`.

Here is how a `setup.py` would usually look like, you could install this package using `pip install astromech-drivers`.
```
from setuptools import setup, find_packages

setup(
    name='astromech-drivers',
    version='2.2.0',
    description='Drivers for the class-2 astromech droids.',
    url='https://starwars.fandom.com/wiki/Astromech_droid',
    author='sh4x2',
    author_email='sh4x2@industrialautomation.com',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    python_requires='>=3.8',
)
```

So I started looking for answer to the question...

## Why `pyproject.toml`?

To package your python module, you need to build its distribution. This distribution can then be used by the users to install the package on their system.

Python separates the build tooling into `backend` and `frontend`. `backend` refers to tools like `setuptools` that are used to build a distribution. `frontend` refers to tools like `pip` that serves as an interface to communicate with a `backend`.
The `setup.py` above uses a `setuptools` backend and `pip` as a frontend. For a long time, `setuptools` acted as a gatekeeper for Python build systems making it impossible to use anything else in its place.

[PEP 517](https://peps.python.org/pep-0517/) came with a change, it specifies a standard format to develop other build backends like `setuptools`. Complimentarily, [PEP 518](https://peps.python.org/pep-0518/) specifies the format in which the build frontend and backend can interface using a `pyproject.toml` file. `pyproject.toml` is now the default way to build distribution and `setup.py` is only used if `pyproject.toml` is not preset.

Here is how a corresponding `pyproject.toml` file looks like for the `astromech-drivers` package.
```
[project]
name = "astromech-drivers"
version = "2.2.0"
description = "Drivers for the class-2 astromech droids"
authors = [
    {name = "sh4x2", email = "sh4x2@industrialautomation.com"},
]
requires-python = ">=3.8"

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools", "setuptools-scm"]
build-backend = "setuptools.build_meta"
```

Key thing to notice here is the `build-system` specification. This is used to specify the requirements as to which build backend is supposed to be used. In `setup.py`, you had to assume that this requirement was already satisfied before attempting to build the package. Using `pyproject.toml` makes it easier for tools like `pip` to first fetch the build backend requirements before attempting to build the package.


## Distribution

There are two types of distributions
- Source Distribution: Commonly referred to as `sdist` is a distribution of source code and any other files needed to compile and install the package.
- Build Distribution: Commonly referred to as `wheel` is a binary distribution that contains the pre-compiled binaries of the package. This means no additionally compilation step is necessary during installation and it is faster to install.

Build frontends first look for build distributions and fall back to source distributions while installation. Most packages provide both, preferably build distribution but at least source distribution. When you publish your package to a repository like PyPI, you basically upload your distribution for others to download, for example with `pip install astromech-drivers`.
You might have seen python `eggs`, these are older distribution format in `setuptools` that are now deprecated in favour of `wheels`


