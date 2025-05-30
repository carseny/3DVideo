import numpy as np
import cv2
from typing import Any

from .noiser import Noise, RGBNoise

__all__ = ["StereoVideo", "process_video"]
chunk_size = 16384


class StereoVideo:
    def __init__(
        self,
        h: int,
        w: int,
        fps=30.0,
        split=5,
        shift: float | None = None,
        shift_scale=0.02,
        hw_scale=0.02,
        t_scale=0.4,
        octave=4,
        noise_type=Noise.Type.SIMPLEX,
        fast_render=False,
    ):
        """深度图转换为平行眼立体视频Filter

        Args:
            h (int): 图像高度
            w (int): 图像宽度
            fps (float): 原视频帧率，用于计算时间噪声尺度
            split (int, optional): 立体图像分割数，1为双眼各一个图像，以此类推
            shift (_type_, optional): 偏移像素数，控制视差大小
            shift_scale (float, optional): 偏移像素相对比例，仅在shift为None时有效
            hw_scale (float, optional): 高度和宽度噪声相对尺度
            t_scale (float, optional): 时间噪声相对尺度
            octave (int, optional): 噪声采样八度数
            noise_type (_type_, optional): 噪声类型
            fast_render (bool, optional): 是否降低深度量化精度来加速渲染（会导致立体效果下降）
        """
        # 图像的相对尺寸
        scale = min(h, w)
        # 计算偏移像素数
        if shift is None:
            shift = scale * shift_scale
        # 存储参数
        self.h = h
        self.w = w
        self.shift = shift
        self.split = split
        self.split_w = w // split
        self.fast_render = fast_render
        # 初始化噪声生成器
        self.back_noise = RGBNoise(
            h, self.split_w, scale * hw_scale, fps * t_scale, octave, noise_type
        )
        # 平行眼辅助锚点掩码
        self.anchor_mask = (
            (np.arange(self.h)[:, None] - scale * 0.02) ** 2
            + (np.arange(self.split_w) - self.split_w / 2) ** 2
        ) < (scale * 0.015) ** 2

    def apply(
        self, frame: np.ndarray[Any, np.dtype[np.floating]]
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        """将一帧图像应用平行眼立体滤镜

        Args:
            frame (np.ndarray): 深度图shape(h, w)，range[0, 1]。值越大越远
        Returns:
            np.ndarray: 立体图 (h, w, 3)，range[0, 1]
        """
        result = np.zeros((*self.result_shape, 3), dtype=np.float32)
        x_map = np.empty(self.result_shape, dtype=np.float32)
        y_map = np.empty(self.result_shape, dtype=np.float32)
        x_map[...] = np.arange(self.h, dtype=np.float32)[:, None]
        y_map[...] = np.arange(self.split_w + self.w, dtype=np.float32)

        while (mask := y_map < self.w).any():
            shift = np.empty((mask.sum(),), dtype=np.float32)
            for chunk in range(0, mask.sum(), chunk_size):
                shift[None, chunk : chunk + chunk_size] = (
                    cv2.remap(
                        frame,
                        y_map[mask][None, chunk : chunk + chunk_size],
                        x_map[mask][None, chunk : chunk + chunk_size],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,  # 边界处理（常量填充）
                        borderValue=[0],  # 填充值（可选）
                    )
                    * self.shift
                )
            y_map[mask] += shift + self.split_w
        y_map -= self.w
        if self.fast_render:
            # 快速渲染模式下，只采样整数位置的噪声值
            tmp = self.back_noise.next(
                np.arange(self.h, dtype=np.float32)[:, None],
                np.arange(int(y_map.max()) + 1, dtype=np.float32),
            )
            result[...] = tmp[x_map.astype(np.int32), y_map.astype(np.int32)]
        else:
            # 普通渲染模式下，采样所有位置的噪声值
            result[:, ...] = self.back_noise.next(x_map, y_map)
        # 生成圆形平行眼锚点
        result[
            :,
            (self.split // 2) * self.split_w : (self.split // 2 + 1) * self.split_w,
        ][self.anchor_mask] = 0.999
        result[
            :,
            (self.split // 2 + 1) * self.split_w
            + int(self.shift / 2) : (self.split // 2 + 2) * self.split_w
            + int(self.shift / 2),
        ][self.anchor_mask] = 0.999

        return result

    @property
    def result_shape(self):
        return (self.h, self.split_w + self.w)


def process_video(
    input,
    output,
    target_height=None,
    num_frame: int | None = None,
    display=False,
    quality=100,
    **kwargs,
):
    cap = cv2.VideoCapture(input)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if num_frame is None:
            num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if target_height is not None:
            w, h = int(w * target_height / h), target_height
        filter = StereoVideo(h, w, fps, **kwargs)

        writer = cv2.VideoWriter(
            output,
            cv2.VideoWriter.fourcc(*"mp4v"),
            fps,
            filter.result_shape[::-1],
        )
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)
        try:
            count = 0
            while True:
                count += 1
                ret, frame = cap.read()
                if not ret or count > num_frame:
                    break

                # 转换为灰度图像
                frame = cv2.resize(frame, (w, h))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if display:
                    cv2.imshow("input", frame)
                    cv2.waitKey(1)

                # 归一化
                frame = frame.astype(np.float32) / 255

                # 转换帧
                frame = filter.apply(frame)

                frame = (frame * 256).astype(np.uint8)
                if display:
                    cv2.imshow("output", frame)
                    cv2.waitKey(1)
                # 写入输出流
                writer.write(frame)
        finally:
            writer.release()
    finally:
        cap.release()
