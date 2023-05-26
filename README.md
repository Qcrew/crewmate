# Title

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

ERROR    HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
         File already exists. See https://pypi.org/help/#file-name-reuse for more information.

=> change version number
