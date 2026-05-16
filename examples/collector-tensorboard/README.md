# Collector → TensorBoard

Drives [`ResourceMetricCollector`][collector] around a tiny [PyTorch] training loop and logs both the loss and the resource metrics into [TensorBoard]. The `add_scalar_dict` helper (5 lines) is inlined in the script so the file is self-contained.

## APIs Used

- [`nvitop.ResourceMetricCollector`][collector]
- [`nvitop.CudaDevice.all()`][cuda-all]

## Install + Run

```bash
pip install -r examples/collector-tensorboard/requirements.txt
python3 examples/collector-tensorboard/collector_tensorboard.py
```

The script writes a [TensorBoard] event file under `runs/` and prints the location at the end. View it with `tensorboard --logdir runs`.

See [`../README.md`](../README.md) for the full example index.

[PyTorch]: https://pytorch.org
[TensorBoard]: https://www.tensorflow.org/tensorboard
[collector]: https://nvitop.readthedocs.io/en/latest/api/collector.html#nvitop.ResourceMetricCollector
[cuda-all]: https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.CudaDevice.all
