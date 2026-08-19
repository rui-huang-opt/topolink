from json import loads, dumps
from typing import Protocol

import msgspec
import numpy as np
import numpy.random as npr
from numpy.typing import NDArray


class BasicMeta(msgspec.Struct, frozen=True):
    dtype: str
    shape: tuple[int, ...]


class QuantizeMeta(msgspec.Struct, frozen=True):
    dtype: str
    shape: tuple[int, ...]
    scale: float


class Transform(Protocol):
    def encode(self, state: NDArray[np.float64]) -> tuple[bytes, NDArray[np.number]]:
        """
        Encode the state into a payload with its data type.

        Args:
            state (NDArray[np.float64]): The state to encode.

        Returns:
            tuple[bytes, NDArray[np.number]]: A tuple containing the meta data and the encoded payload.
        """
        ...

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray[np.float64]:
        """
        Decode the payload back into the original state using the meta data.

        Args:
            meta (bytes): The meta data.

            payload (bytes): The encoded payload.

        Returns:
            NDArray[np.float64]: The decoded state.
        """
        ...


class Identity:
    """
    Identity transform that performs no transformation.
    """

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        meta = BasicMeta(dtype=state.dtype.str, shape=state.shape)
        meta_bytes = msgspec.msgpack.encode(meta)
        return meta_bytes, state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=BasicMeta)
        dtype = np.dtype(meta.dtype)
        return np.frombuffer(payload, dtype=dtype).reshape(meta.shape)


class Quantize:
    """
    Quantize transform that scales and converts the state to a specified integer type.

    Parameters
    ----------
    scale : float
        The scale factor for quantization.
    """

    def __init__(self, dtype: str = "int8"):
        self.dtype = dtype

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        dtype = np.dtype(self.dtype)
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max

        abs_state = np.abs(state)
        nonzero = abs_state[abs_state > 0]
        if nonzero.size == 0:
            scale = 1.0
        else:
            scale = float(np.max(nonzero) / max_val)

        scaled_state = state / scale
        rounded_state = np.round(scaled_state)
        clipped_state = np.clip(rounded_state, min_val, max_val)
        quantized_state = clipped_state.astype(dtype, copy=False)

        meta = QuantizeMeta(dtype=self.dtype, shape=quantized_state.shape, scale=scale)
        meta_bytes = msgspec.msgpack.encode(meta)
        return meta_bytes, quantized_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=QuantizeMeta)
        dtype = np.dtype(meta.dtype)
        quantized_state = np.frombuffer(payload, dtype=dtype).reshape(meta.shape)
        return quantized_state.astype(np.float64, copy=False) * meta.scale


class DPMechanism:
    """
    Differential Privacy mechanism that adds Laplace noise to the state.

    Parameters
    ----------
    epsilon : float
        The privacy budget parameter.

    sensitivity : float
        The sensitivity of the function to which noise is added.
    """

    def __init__(self, epsilon: float, sensitivity: float):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self._scale = sensitivity / epsilon

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        noise = npr.laplace(0, self._scale, size=state.shape)
        noisy_state = state + noise
        dtype = noisy_state.dtype.str
        meta = BasicMeta(dtype=dtype, shape=noisy_state.shape)
        meta_bytes = msgspec.msgpack.encode(meta)
        return meta_bytes, noisy_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=BasicMeta)
        dtype = np.dtype(meta.dtype)
        return np.frombuffer(payload, dtype=dtype).reshape(meta.shape)


class GaussianNoise:
    """
    Gaussian noise mechanism that adds Gaussian noise to the state.

    Parameters
    ----------
    loc : float
        The mean of the Gaussian noise.

    scale : float
        The standard deviation of the Gaussian noise.
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        self.loc = loc
        self.scale = scale

    def encode(self, state: NDArray) -> tuple[bytes, NDArray]:
        noise = npr.normal(self.loc, self.scale, state.shape)
        noisy_state = state + noise
        dtype = noisy_state.dtype.str
        meta = BasicMeta(dtype=dtype, shape=noisy_state.shape)
        meta_bytes = msgspec.msgpack.encode(meta)
        return meta_bytes, noisy_state

    def decode(self, meta_bytes: bytes, payload: bytes) -> NDArray:
        meta = msgspec.msgpack.decode(meta_bytes, type=BasicMeta)
        dtype = np.dtype(meta.dtype)
        return np.frombuffer(payload, dtype=dtype).reshape(meta.shape)
