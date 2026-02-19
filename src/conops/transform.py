import json
from typing import Protocol

import numpy as np
import numpy.random as npr
from numpy.typing import NDArray


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

    def decode(self, meta: bytes, payload: bytes) -> NDArray[np.float64]:
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

    def encode(self, state: NDArray[np.float64]) -> tuple[bytes, NDArray[np.number]]:
        dtype = state.dtype.str
        meta = json.dumps({"dtype": dtype}).encode()
        return meta, state

    def decode(self, meta: bytes, payload: bytes) -> NDArray[np.float64]:
        meta_dict = json.loads(meta.decode())
        dtype = np.dtype(meta_dict["dtype"])
        return np.frombuffer(payload, dtype=dtype).astype(np.float64, copy=False)


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

    def encode(self, state: NDArray[np.float64]) -> tuple[bytes, NDArray[np.number]]:
        dtype_ = np.dtype(self.dtype)
        min_val = np.iinfo(dtype_).min
        max_val = np.iinfo(dtype_).max

        abs_state = np.abs(state)
        nonzero = abs_state[abs_state > 0]
        if nonzero.size == 0:
            scale = 1.0
        else:
            scale = np.max(nonzero) / max_val

        scaled_state = state / scale
        rounded_state = np.round(scaled_state)
        clipped_state = np.clip(rounded_state, min_val, max_val)
        quantized_state = clipped_state.astype(dtype_, copy=False)

        meta = json.dumps({"scale": scale, "dtype": self.dtype}).encode()
        return meta, quantized_state

    def decode(self, meta: bytes, payload: bytes) -> NDArray[np.float64]:
        meta_dict = json.loads(meta.decode())
        dtype = np.dtype(meta_dict["dtype"])
        scale = meta_dict["scale"]
        quantized_state = np.frombuffer(payload, dtype=dtype)
        return quantized_state.astype(np.float64, copy=False) * scale


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

    def encode(self, state: NDArray[np.float64]) -> tuple[bytes, NDArray[np.number]]:
        noise = npr.laplace(0, self._scale, size=state.shape)
        noisy_state = state + noise
        dtype = noisy_state.dtype.str
        meta = json.dumps({"dtype": dtype}).encode()
        return meta, noisy_state

    def decode(self, meta: bytes, payload: bytes) -> NDArray[np.float64]:
        meta_dict = json.loads(meta.decode())
        dtype = np.dtype(meta_dict["dtype"])
        return np.frombuffer(payload, dtype=dtype).astype(np.float64, copy=False)


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

    def encode(self, state: NDArray[np.float64]) -> tuple[bytes, NDArray[np.float64]]:
        noise = npr.normal(self.loc, self.scale, state.shape)
        noisy_state = state + noise
        dtype = noisy_state.dtype.str
        meta = json.dumps({"dtype": dtype}).encode()
        return meta, noisy_state

    def decode(self, meta: bytes, payload: bytes) -> NDArray[np.float64]:
        meta_dict = json.loads(meta.decode())
        dtype = np.dtype(meta_dict["dtype"])
        return np.frombuffer(payload, dtype=dtype).astype(np.float64, copy=False)
