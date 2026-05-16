# ML-Framework Callbacks

Self-contained `GpuStatsLogger` callbacks for [Keras] and [Lightning], plus a tiny [TensorBoard] helper. Each file has zero dependencies on the others — copy a single `.py` into your project and adapt the imports.

## Files and Dependencies

| File             | Install command                               | Purpose                                                                        |
| ---------------- | --------------------------------------------- | ------------------------------------------------------------------------------ |
| `keras.py`       | `pip install -r requirements-keras.txt`       | [Keras] `Callback` logging GPU stats per training batch.                       |
| `lightning.py`   | `pip install -r requirements-lightning.txt`   | [Lightning] `Callback` (`lightning.pytorch.callbacks.Callback`) for GPU stats. |
| `tensorboard.py` | `pip install -r requirements-tensorboard.txt` | `add_scalar_dict` — batched `SummaryWriter.add_scalar` for a flat metric dict. |

Each `requirements-*.txt` pins only what that single file needs, so you do not pull in [TensorFlow] when you only want the [Lightning] callback.

See [`../README.md`](../README.md) for the full example index.

[Keras]: https://keras.io
[Lightning]: https://lightning.ai
[TensorBoard]: https://www.tensorflow.org/tensorboard
[TensorFlow]: https://www.tensorflow.org
