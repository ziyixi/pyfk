#!/bin/bash

${PYTHON} -m pip install poetry-core@https://github.com/python-poetry/poetry-core/archive/refs/tags/1.1.0a7.zip
# the pip version seems to be too old
${PYTHON} -m pip install .