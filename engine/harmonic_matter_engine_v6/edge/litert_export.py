from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence


@contextmanager
def _suppress_stderr():
    """Best-effort suppression of noisy native-library stderr logs.

    Some TF/TFLite components emit info/warn logs directly to stderr from native
    code, bypassing Python/absl log filtering.
    """

    try:
        import sys

        fd = sys.stderr.fileno()
        saved = os.dup(fd)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), fd)
                yield
        finally:
            os.dup2(saved, fd)
            os.close(saved)
    except Exception:
        yield


class LiteRTCompiler:
    """Compiles JAX-SPH kernels to LiteRT (TFLite) for Mobile NPU deployment."""

    def __init__(self):
        pass

    def convert_jax_to_tflite(
        self,
        jax_func: Any,
        sample_inputs: Sequence[Any],
        output_path: str | Path = "engine_v6_npu.tflite",
        quantization: str = "none",
    ) -> bytes:
        """Convert a JAX function to a TFLite flatbuffer.

        `jax_func` should be a pure function with Tensor-like inputs/outputs.
        `sample_inputs` are used to define fixed input shapes for the converter.
        """
        output_path = Path(output_path)

        print(">>> EDGE: Converting JAX kernel to TensorFlow...")
        # Reduce noisy third-party logs/warnings (TensorFlow/Keras/JAX2TF).
        # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=filter INFO+WARNING+ERROR.
        # We set 3 to suppress known-noisy converter logs; conversion failures still raise exceptions.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        warnings.filterwarnings(
            "ignore",
            message=r"In the future `np\.object` will be defined as the corresponding NumPy scalar\.",
            category=FutureWarning,
        )
        try:
            from absl import logging as absl_logging

            absl_logging.set_verbosity(absl_logging.ERROR)
            absl_logging.set_stderrthreshold("error")
        except Exception:
            pass
        try:
            import tensorflow as tf
            from jax.experimental import jax2tf
        except Exception as e:
            raise RuntimeError(
                "TensorFlow and jax2tf are required to produce a real .tflite. "
                "Install 'tensorflow-cpu' (or tensorflow) and retry."
            ) from e

        tf_func = jax2tf.convert(jax_func, with_gradient=False)

        # Build a tf.function with fixed shapes for TFLite.
        specs = []
        for x in sample_inputs:
            x = tf.convert_to_tensor(x)
            specs.append(tf.TensorSpec(shape=x.shape, dtype=x.dtype))

        @tf.function(input_signature=specs)
        def wrapped(*args):
            return tf_func(*args)

        # Suppress AutoGraph warnings triggered by internal helper functions.
        try:
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
        try:
            tf.autograph.set_verbosity(0)
        except Exception:
            pass

        concrete = wrapped.get_concrete_function()

        print(">>> EDGE: Converting to TFLite flatbuffer...")
        # Providing a trackable object avoids the deprecated conversion path warning.
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapped)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        q = (quantization or "none").lower().strip()
        if q not in {"none", "dynamic", "float16", "int8"}:
            raise ValueError(
                "quantization must be one of: none, dynamic, float16, int8"
            )

        def _rep_dataset() -> Iterable[list[tf.Tensor]]:
            # Minimal representative dataset derived from sample inputs.
            # Yields a few slightly perturbed samples to calibrate ranges.
            base = [tf.convert_to_tensor(x) for x in sample_inputs]
            for _ in range(25):
                out = []
                for t in base:
                    if t.dtype.is_floating:
                        noise = tf.random.normal(
                            tf.shape(t), stddev=0.01, dtype=t.dtype
                        )
                        out.append(t + noise)
                    else:
                        out.append(t)
                yield out

        if q == "float16":
            print(
                ">>> EDGE: Enabling float16 quantization (weights/activations where possible)."
            )
            converter.target_spec.supported_types = [tf.float16]

        if q == "int8":
            print(
                ">>> EDGE: Enabling full int8 quantization (uses representative dataset)."
            )
            converter.representative_dataset = _rep_dataset
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        with _suppress_stderr():
            tflite_model = converter.convert()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_model)
        print(f">>> EDGE: Saved TFLite model: {output_path.resolve()}")
        return tflite_model
