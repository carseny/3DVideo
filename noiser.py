import numpy as np
from enum import Enum
from noise import pnoise3, snoise3


class Noise:
    class Type(Enum):
        PERLIN = pnoise3
        SIMPLEX = snoise3

    def __init__(
        self, height, width, scale=10.0, t_scale=10.0, octaves=1, type=Type.PERLIN
    ):
        self.height = height
        self.width = width
        self.scale = scale
        self.t_scale = t_scale
        self.octaves = octaves
        self.type = type
        self.x0, self.y0, self.t0 = np.random.rand(3) * 1e5
        self.t = 0

        self.__vectorize = np.vectorize(
            lambda x, y, t: self.type.value(x, y, t, octaves=self.octaves)
        )

    def get(
        self,
        x: float | np.ndarray = 0,
        y: float | np.ndarray = 0,
        t: float | np.ndarray = 0,
    ) -> np.ndarray:
        # 向量化采样噪声
        return self.__vectorize(
            x / self.scale + self.x0,
            y / self.scale + self.y0,
            t / self.t_scale + self.t0,
        )

    def next(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None, mask=None
    ) -> np.ndarray:
        self.t += 1
        if x is None or y is None:
            # 生成坐标网格
            x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            if mask is not None:
                x = x[mask]
                y = y[mask]
        return self.get(x, y, t=self.t)


class RGBNoise:
    def __init__(
        self,
        h: int,
        w: int,
        scale=10.0,
        t_scale=10.0,
        octaves=1,
        type=Noise.Type.PERLIN,
    ):
        self.noises = [
            Noise(h, w, scale, t_scale, octaves, type) for _ in range(3)
        ]  # rgb

    def get(
        self,
        x: float | np.ndarray = 0,
        y: float | np.ndarray = 0,
        t: float | np.ndarray = 0,
    ) -> np.ndarray:
        noise_rgb = np.stack([noise.get(x, y, t) for noise in self.noises], axis=-1)
        return noise_rgb * 0.25 + 0.5

    def next(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None, mask=None
    ) -> np.ndarray:
        noise_rgb = np.stack([noise.next(x, y, mask) for noise in self.noises], axis=-1)
        return noise_rgb * 0.25 + 0.5
