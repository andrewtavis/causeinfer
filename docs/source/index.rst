.. image:: https://raw.githubusercontent.com/andrewtavis/causeinfer/main/.github/resources/logo/causeinfer_logo_transparent.png
    :width: 612
    :height: 164
    :align: center
    :target: https://github.com/andrewtavis/causeinfer
============

|rtd| |ci| |python_package_ci| |codecov| |pyversions| |pypi| |pypistatus| |license| |coc| |codestyle| |colab|

.. |rtd| image:: https://img.shields.io/readthedocs/causeinfer.svg?logo=read-the-docs
    :target: http://causeinfer.readthedocs.io/en/latest/

.. |pr_ci| image:: https://img.shields.io/github/actions/workflow/status/andrewtavis/causeinfer/.github/workflows/pr_ci.yaml?branch=main&label=ci&logo=ruff
    :target: https://github.com/andrewtavis/causeinfer/actions/workflows/pr_ci.yaml

.. |python_package_ci| image:: https://img.shields.io/github/actions/workflow/status/andrewtavis/causeinfer/.github/workflows/python_package_ci.yaml?branch=main&label=build&logo=pytest
    :target: https://github.com/andrewtavis/causeinfer/actions/workflows/python_package_ci.yaml

.. |codecov| image:: https://codecov.io/gh/andrewtavis/causeinfer/branch/main/graphs/badge.svg
    :target: https://codecov.io/gh/andrewtavis/causeinfer

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/causeinfer.svg?logo=python&logoColor=FFD43B&color=306998
    :target: https://pypi.org/project/causeinfer/

.. |pypi| image:: https://img.shields.io/pypi/v/causeinfer.svg?color=4B8BBE
    :target: https://pypi.org/project/causeinfer/

.. |pypistatus| image:: https://img.shields.io/pypi/status/causeinfer.svg
    :target: https://pypi.org/project/causeinfer/

.. |license| image:: https://img.shields.io/github/license/andrewtavis/causeinfer.svg
    :target: https://github.com/andrewtavis/causeinfer/blob/main/LICENSE.txt

.. |coc| image:: https://img.shields.io/badge/coc-Contributor%20Covenant-ff69b4.svg
    :target: https://github.com/andrewtavis/causeinfer/blob/main/.github/CODE_OF_CONDUCT.md

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |colab| image:: https://img.shields.io/badge/%20-Open%20in%20Colab-097ABB.svg?logo=google-colab&color=097ABB&labelColor=525252
    :target: https://colab.research.google.com/github/andrewtavis/causeinfer

Machine learning based causal inference/uplift in Python

Installation
------------

``causeinfer`` is available for installation via `uv <https://docs.astral.sh/uv/>`_ (recommended) or `pip <https://pypi.org/project/causeinfer/>`_.

.. code-block:: shell

    # Using uv (recommended - fast, Rust-based installer):
    uv pip install causeinfer

    # Or using pip:
    pip install causeinfer

.. code-block:: shell

    # For a development build of the package:
    git clone https://github.com/andrewtavis/causeinfer.git
    cd causeinfer

    # With uv (recommended):
    uv sync --all-extras  # install all dependencies
    source .venv/bin/activate  # activate venv (macOS/Linux)
    # .venv\Scripts\activate  # activate venv (Windows)

    # Or with pip:
    python -m venv .venv  # create virtual environment
    source .venv/bin/activate  # activate venv (macOS/Linux)
    # .venv\Scripts\activate  # activate venv (Windows)
    pip install -e .

.. code-block:: python

    import causeinfer

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   standard_algorithms/index
   evaluation/index
   data/index
   utils
   notes

Project Indices
===============

* :ref:`genindex`
