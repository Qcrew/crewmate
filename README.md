# Qcrew crewmate

## Docs

Check out the documentation [here](https://qcrew.github.io/crewmate/).

## Installation

```bash
pip install crewmate
```

## Develop

```bash
pip install -e .
```

## Build

```bash
python setup.py bdist_wheel sdist
```

## Publish

```bash
twine upload dist/*
```

If you get this error while trying to upload:

HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists. See https://pypi.org/help/#file-name-reuse for more information.

Delete the contents of the dist/ folder and make sure you increased the version number in setup.py
