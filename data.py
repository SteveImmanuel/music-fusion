import torch
import numpy as np

from nptyping import NDArray


def mu_law_encode(data: NDArray, mu: int = 255) -> NDArray:
    """
    Quantize waveform amplitudes.
    data range: [-inf, inf]
    """
    data = np.sign(data) * (np.log(1 + mu * np.abs(data)) / np.log(1 + mu))
    data = np.floor((data + 1) / 2 * mu)
    return data.astype(np.int64)


def mu_law_decode(data: NDArray, mu: int = 255) -> NDArray:
    """
    Recovers waveform from quantized values.
    data range: [0, mu]
    """
    data = data.astype(np.float32)
    data = 2 * (data / mu) - 1
    data = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return data


if __name__ == "__main__":
    data = np.random.randn(10) * 1023
    print(data)
    data = mu_law_encode(data)
    print(data)
    data = mu_law_decode(data)
    print(data)
    data = mu_law_encode(data)
    print(data)
    data = mu_law_decode(data)
    print(data)
