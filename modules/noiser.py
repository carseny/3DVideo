import numpy as np
from enum import Enum
from noise import pnoise3, snoise3
from typing import Any


class Noise:
    class Type(Enum):
        PERLIN = pnoise3
        SIMPLEX = snoise3
        NORMAL = np.random.rand()

    def __init__(
        self,
        height: int,
        width: int,
        hw_scale=10.0,
        t_scale=10.0,
        octaves=1,
        type=Type.PERLIN,
    ):
        """三维噪声生成器，会从np.random中随机初始化种子偏移量

        Args:
            height (int): 高度
            width (int): 宽度
            hw_scale (float, optional): 高度和宽度噪声尺度
            t_scale (float, optional): 时间噪声尺度
            octaves (int, optional): 采样八度数
            type (_type_, optional): 噪声类型
        """
        self.height = height
        self.width = width
        self.hw_scale = hw_scale
        self.t_scale = t_scale
        self.octaves = octaves
        self.type = type
        self.x0, self.y0, self.t0 = np.random.rand(3) * 1e5
        self.t = 0

        match type:
            case Noise.Type.PERLIN | Noise.Type.SIMPLEX:
                self.__vectorize = np.vectorize(
                    lambda x, y, t: type.value(x, y, t, octaves=self.octaves)
                )
            case _:
                # 默认使用随机噪声
                self.__vectorize = lambda x, y, t: np.random.rand(
                    *np.broadcast_shapes(x.shape, y.shape, t.shape)
                )

    def get(
        self,
        x: float | np.ndarray[Any, np.dtype[np.floating]] = 0,
        y: float | np.ndarray[Any, np.dtype[np.floating]] = 0,
        t: float | np.ndarray[Any, np.dtype[np.floating]] = 0,
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        """应用当前偏移量与缩放因子，采样噪声值"""
        # 向量化采样噪声
        noise = self.__vectorize(
            x / self.hw_scale + self.x0,
            y / self.hw_scale + self.y0,
            t / self.t_scale + self.t0,
        )
        if not isinstance(noise, np.ndarray):
            raise TypeError("noise must be a numpy array")
        return noise.astype(np.floating)

    def next(
        self,
        x: np.ndarray[Any, np.dtype[np.floating]] | None = None,
        y: np.ndarray[Any, np.dtype[np.floating]] | None = None,
        mask=None,
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        """生成下一个时间步噪声帧

        Args:
            x (np.ndarray | None, optional): x 采样点坐标数组，如果为 None，则生成整个图像的噪声
            y (np.ndarray | None, optional): y 采样点坐标数组，如果为 None，则生成整个图像的噪声
            mask (_type_, optional): 生成的区域掩码，仅在x和y为None时有效
        """
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
        """三通道噪声生成器"""
        self.noises = [
            Noise(h, w, scale, t_scale, octaves, type) for _ in range(3)
        ]  # rgb

    def get(
        self,
        x: float | np.ndarray[Any, np.dtype[np.floating]] = 0,
        y: float | np.ndarray[Any, np.dtype[np.floating]] = 0,
        t: float | np.ndarray[Any, np.dtype[np.floating]] = 0,
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        noise_rgb = np.stack([noise.get(x, y, t) for noise in self.noises], axis=-1)
        return noise_rgb * 0.25 + 0.5

    def next(
        self,
        x: np.ndarray[Any, np.dtype[np.floating]] | None = None,
        y: np.ndarray[Any, np.dtype[np.floating]] | None = None,
        mask=None,
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        noise_rgb = np.stack([noise.next(x, y, mask) for noise in self.noises], axis=-1)
        return noise_rgb * 0.25 + 0.5
