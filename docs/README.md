# The `nvitop`'s Documentation

This directory contains the documentation of `nvitop`, the one-stop solution for GPU process management.

### Requirements  <!-- markdownlint-disable heading-increment -->

- `sphinx`
- `sphinx-autoapi`
- `sphinx-autobuild`
- `sphinx-copybutton`
- `sphinx-rtd-theme`
- `make`

### Steps to build the documentation  <!-- markdownlint-disable heading-increment -->

```bash
cd docs  # navigate to this directory
python3 -m venv --upgrade-deps .venv
source .venv/bin/activate
pip3 install -r ../requirements.txt -r requirements.txt
sphinx-autobuild --watch ../nvitop --open-browser source build
```
