Welcome to nvitop's documentation!
==================================

|GitHub|_ |Python Version|_ |PyPI Package|_ |Package Status|_ |Documentation Status|_ |Downloads|_ |GitHub Repo Stars|_ |License|_ |Tweet|_

An interactive NVIDIA-GPU process viewer, the one-stop solution for GPU process management.

.. image:: https://user-images.githubusercontent.com/16078332/171005261-1aad126e-dc27-4ed3-a89b-7f9c1c998bf7.png
    :align: center

.. |GitHub| image:: https://img.shields.io/badge/GitHub-Homepage-blue?logo=github
.. _GitHub: https://github.com/XuehaiPan/nvitop

.. |Python Version| image:: https://img.shields.io/badge/Python-3.5%2B-brightgreen.svg
.. _Python Version: https://pypi.org/project/nvitop

.. |PyPI Package| image:: https://img.shields.io/pypi/v/nvitop?label=PyPI
.. _PyPI Package: https://pypi.org/project/nvitop

.. |Package Status| image:: https://img.shields.io/pypi/status/nvitop?label=Status
.. _Package Status: https://pypi.org/project/nvitop

.. |Documentation Status| image:: https://readthedocs.org/projects/nvitop/badge/?version=latest
.. _Documentation Status: https://nvitop.readthedocs.io/en/latest/?badge=latest

.. |Downloads| image:: https://static.pepy.tech/personalized-badge/nvitop?period=month&left_color=grey&right_color=blue&left_text=Downloads/month
.. _Downloads: https://pepy.tech/project/nvitop

.. |GitHub Repo Stars| image:: https://img.shields.io/github/stars/XuehaiPan/nvitop?label=Stars&logo=github&color=brightgreen
.. _GitHub Repo Stars: https://github.com/XuehaiPan/nvitop

.. |License| image:: https://img.shields.io/github/license/XuehaiPan/nvitop?label=License
.. _License: https://github.com/XuehaiPan/nvitop#license

.. |Tweet| image:: https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2FXuehaiPan%2Fnvitop
.. _Tweet: https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FXuehaiPan%2Fnvitop

------

Installation
""""""""""""

.. code:: bash

    pip3 install --upgrade nvitop

.. note::

    Python 3.5+ is required, and Python versions lower than 3.5 is not supported.

Install from PyPI (|PyPI Package|_ / |Package Status|_):

.. code:: bash

    pip3 install --upgrade nvitop

Install the latest version from GitHub (|Commit Count|):

.. code:: bash

    pip3 install git+https://github.com/XuehaiPan/nvitop.git#egg=nvitop

Or, clone this repo and install manually:

.. code:: bash

    git clone --depth=1 https://github.com/XuehaiPan/nvitop.git
    cd nvitop
    pip3 install .

If this repo is useful to you, please star ⭐️ it to let more people know 🤗. |GitHub Repo Stars|_

.. |Commit Count| image:: https://img.shields.io/github/commits-since/XuehaiPan/nvitop/v0.6.1

------

Quick Start
"""""""""""

.. code-block:: python

    from nvitop import Device

    devices = Device.all()  # or Device.cuda.all()
    for device in devices:
        processes = device.processes()  # type: Dict[int, GpuProcess]
        sorted_pids = sorted(processes)

        print(device)
        print(f'  - Fan speed:       {device.fan_speed()}%')
        print(f'  - Temperature:     {device.temperature()}C')
        print(f'  - GPU utilization: {device.gpu_utilization()}%')
        print(f'  - Total memory:    {device.memory_total_human()}')
        print(f'  - Used memory:     {device.memory_used_human()}')
        print(f'  - Free memory:     {device.memory_free_human()}')
        print(f'  - Processes ({len(processes)}): {sorted_pids}')
        for pid in sorted_pids:
            print(f'    - {processes[pid]}')
        print('-' * 80)

------

.. toctree::
    :maxdepth: 4
    :caption: API Reference:

    apis/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
