# nvitop Examples

Runnable reference scripts that exercise the public `nvitop` API. Each subfolder is fully self-contained: one runnable `.py` file, a `README.md`, and (if extra dependencies are required) a `requirements.txt`.

## Index

| Folder                                               | What it shows                                                                               | Extra deps                 |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------- |
| [`monitor-minimal/`](./monitor-minimal/)             | Minimal plain-text GPU monitor using `Device.all()`.                                        | —                          |
| [`monitor-colored/`](./monitor-colored/)             | Colored monitor with per-process snapshots.                                                 | —                          |
| [`take-snapshots/`](./take-snapshots/)               | Every form of `take_snapshots` — NVML, CUDA, single-device, processes off.                  | —                          |
| [`collector-tensorboard/`](./collector-tensorboard/) | Log `ResourceMetricCollector` output to [TensorBoard] around a tiny [PyTorch] loop.         | [`torch`], [`tensorboard`] |
| [`collector-csv/`](./collector-csv/)                 | Append `ResourceMetricCollector` samples to a CSV file via [pandas].                        | [`pandas`]                 |
| [`collector-background/`](./collector-background/)   | Run the collector on a daemon thread via `collect_in_background`.                           | —                          |
| [`select-devices-api/`](./select-devices-api/)       | Programmatic CUDA device selection (`nvitop.select_devices`), the API behind `nvisel`.      | —                          |
| [`callbacks/`](./callbacks/)                         | ML-framework callbacks ([Keras], [Lightning]) and a [TensorBoard] helper built on `nvitop`. | varies per file            |

## Running an Example

```bash
# Folders without a requirements file only need `nvitop` itself
python3 examples/<folder>/<script>.py

# Folders with a requirements file
pip install -r examples/<folder>/requirements.txt
python3 examples/<folder>/<script>.py
```

The `callbacks/` folder ships per-framework `requirements-<framework>.txt` files so you can pull in only the framework you actually use.

[Keras]: https://keras.io
[Lightning]: https://lightning.ai
[PyTorch]: https://pytorch.org
[TensorBoard]: https://www.tensorflow.org/tensorboard
[pandas]: https://pandas.pydata.org
[`pandas`]: https://pandas.pydata.org
[`tensorboard`]: https://www.tensorflow.org/tensorboard
[`torch`]: https://pytorch.org
