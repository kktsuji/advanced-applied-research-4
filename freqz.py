import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz


def _plot_freqz(arr):
    arr_np = np.array(arr)
    arr_np = arr_np / (np.sum(arr_np) + 1e-10)
    w, h = freqz(arr_np)

    plt.title("Frequency Response")
    plt.plot(w, 20 * np.log10(abs(h) + 10e-5), "b", label=str(arr))
    plt.ylim([-50.0, 1.0])
    plt.ylabel("Amplitude [dB]")
    plt.xlabel("Frequency [radians / sample]")
    plt.grid()
    plt.legend()
    plt.axis("tight")

    if not os.path.exists("./out"):
        os.makedirs("./out")
    plt.savefig(f"./out/freqz_{arr}.png")


if __name__ == "__main__":
    arr = list(map(int, sys.argv[1:]))
    _plot_freqz(arr)
