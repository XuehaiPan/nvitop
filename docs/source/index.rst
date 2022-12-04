Welcome to nvitop's documentation!
==================================

|GitHub|_ |Python Version|_ |PyPI Package|_ |Package Status|_ |Conda Package|_ |Documentation Status|_ |Downloads|_ |GitHub Repo Stars|_ |License|_ |Tweet|_

An interactive NVIDIA-GPU process viewer, the one-stop solution for GPU process management.

.. figure:: https://user-images.githubusercontent.com/16078332/171005261-1aad126e-dc27-4ed3-a89b-7f9c1c998bf7.png
    :align: center

    The CLI from ``nvitop``.

.. |GitHub| image:: https://img.shields.io/badge/GitHub-Homepage-blue?logo=github
.. _GitHub: https://github.com/XuehaiPan/nvitop

.. |Python Version| image:: https://img.shields.io/badge/Python-3.6%2B-brightgreen
.. _Python Version: https://pypi.org/project/nvitop

.. |PyPI Package| image:: https://img.shields.io/pypi/v/nvitop?label=pypi&logo=pypi
.. _PyPI Package: https://pypi.org/project/nvitop

.. |Conda Package| image:: https://img.shields.io/conda/vn/conda-forge/nvitop?label=conda&logo=condaforge
.. _Conda Package: https://anaconda.org/conda-forge/nvitop

.. |Conda-forge Package| image:: https://img.shields.io/conda/v/conda-forge/nvitop?logo=condaforge
.. _Conda-forge Package: https://anaconda.org/conda-forge/nvitop

.. |Package Status| image:: https://img.shields.io/pypi/status/nvitop?label=status
.. _Package Status: https://pypi.org/project/nvitop

.. |Documentation Status| image:: https://img.shields.io/readthedocs/nvitop?label=docs&logo=readthedocs
.. _Documentation Status: https://nvitop.readthedocs.io

.. |Downloads| image:: https://static.pepy.tech/personalized-badge/nvitop?period=total&left_color=grey&right_color=blue&left_text=downloads
.. _Downloads: https://pepy.tech/project/nvitop

.. |GitHub Repo Stars| image:: https://img.shields.io/github/stars/XuehaiPan/nvitop?label=stars&logo=github&color=brightgreen
.. _GitHub Repo Stars: https://github.com/XuehaiPan/nvitop

.. |License| image:: https://img.shields.io/github/license/XuehaiPan/nvitop?label=license
.. _License: https://github.com/XuehaiPan/nvitop#license

.. |Tweet| image:: https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2FXuehaiPan%2Fnvitop
.. _Tweet: https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FXuehaiPan%2Fnvitop

------

Installation
""""""""""""

**It is highly recommended to install nvitop in an isolated virtual environment.** Simple installation and run via `pipx <https://pypa.github.io/pipx>`_:

.. code:: bash

    pipx run nvitop

Install from PyPI (|PyPI Package|_ / |Package Status|_):

.. code:: bash

    pip3 install --upgrade nvitop

.. note::

    Python 3.6+ is required, and Python versions lower than 3.6 is not supported.

Install from conda-forge (|Conda-forge Package|_):

.. code:: bash

    conda install -c conda-forge nvitop

Install the latest version from GitHub (|Commit Count|):

.. code:: bash

    pip3 install git+https://github.com/XuehaiPan/nvitop.git#egg=nvitop

Or, clone this repo and install manually:

.. code:: bash

    git clone --depth=1 https://github.com/XuehaiPan/nvitop.git
    cd nvitop
    pip3 install .

If this repo is useful to you, please star ⭐️ it to let more people know 🤗. |GitHub Repo Stars|_

.. |Commit Count| image:: https://img.shields.io/github/commits-since/XuehaiPan/nvitop/v0.11.0

------

Quick Start
"""""""""""

A minimal script to monitor the GPU devices based on APIs from ``nvitop``:

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
        print('-' * 120)

Another more advanced approach with coloring:

.. code-block:: python

    import time

    from nvitop import Device, GpuProcess, NA, colored

    print(colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='red', attrs=('bold',)))

    devices = Device.cuda.all()  # or `Device.all()` to use NVML ordinal instead
    separator = False
    for device in devices:
        processes = device.processes()  # type: Dict[int, GpuProcess]

        print(colored(str(device), color='green', attrs=('bold',)))
        print(colored('  - Fan speed:       ', color='blue', attrs=('bold',)) + f'{device.fan_speed()}%')
        print(colored('  - Temperature:     ', color='blue', attrs=('bold',)) + f'{device.temperature()}C')
        print(colored('  - GPU utilization: ', color='blue', attrs=('bold',)) + f'{device.gpu_utilization()}%')
        print(colored('  - Total memory:    ', color='blue', attrs=('bold',)) + f'{device.memory_total_human()}')
        print(colored('  - Used memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_used_human()}')
        print(colored('  - Free memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_free_human()}')
        if len(processes) > 0:
            processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
            processes.sort(key=lambda process: (process.username, process.pid))

            print(colored(f'  - Processes ({len(processes)}):', color='blue', attrs=('bold',)))
            fmt = '    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}  {gpu_memory:>8}  {sm:>3}  {command:<}'.format
            print(colored(fmt(pid='PID', username='USERNAME',
                              cpu='CPU%', host_memory='HOST-MEM', time='TIME',
                              gpu_memory='GPU-MEM', sm='SM%',
                              command='COMMAND'),
                          attrs=('bold',)))
            for snapshot in processes:
                print(fmt(pid=snapshot.pid,
                          username=snapshot.username[:7] + ('+' if len(snapshot.username) > 8 else snapshot.username[7:8]),
                          cpu=snapshot.cpu_percent, host_memory=snapshot.host_memory_human,
                          time=snapshot.running_time_human,
                          gpu_memory=(snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A'),
                          sm=snapshot.gpu_sm_utilization,
                          command=snapshot.command))
        else:
            print(colored('  - No Running Processes', attrs=('bold',)))

        if separator:
            print('-' * 120)
        separator = True

.. figure:: https://user-images.githubusercontent.com/16078332/177041142-fe988d58-6a97-4559-84fd-b51204cf9231.png
    :align: center

    An example monitoring script built with APIs from ``nvitop``.

Please refer to section `More than a Monitor <https://github.com/XuehaiPan/nvitop#more-than-a-monitor>`_ in README for more examples.

------

.. toctree::
    :maxdepth: 4
    :caption: API Reference

    core/device
    core/process
    core/host
    core/collector
    core/libnvml
    core/libcuda
    core/utils
    select
    callbacks

------

Module Contents
"""""""""""""""

.. automodule:: nvitop.version
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: nvitop.NaType
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autodata:: nvitop.NA

.. autoclass:: nvitop.NotApplicableType

.. autodata:: nvitop.NotApplicable

.. automodule:: nvitop
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :noindex: Device
    :exclude-members: NA, NaType, NotApplicable, NotApplicableType

------

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
