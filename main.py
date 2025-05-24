import subprocess as sp
import numpy as np
import shlex
import cv2
from noise import pnoise3, snoise3
from enum import Enum

from video import get_video_info, Video


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
        self.x, self.y, self.t = np.random.rand(3) * 1e5

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
            x / self.scale + self.x, y / self.scale + self.y, t / self.t_scale + self.t
        )

    def next(self, mask=None) -> np.ndarray:
        self.t += 1
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

    def next(self, mask=None) -> np.ndarray:
        noise_rgb = np.stack([noise.next(mask=mask) for noise in self.noises], axis=-1)
        return (noise_rgb * 64 + 128).astype(np.uint8)


def generate_coords(m, n) -> np.ndarray:
    coords = np.empty((m, n, 2), dtype=np.int32)
    coords[..., 0] = np.arange(m)[:, None]  # 列索引作为x坐标
    coords[..., 1] = np.arange(n)  # 行索引作为y坐标
    return coords


class DualVideo:
    def __init__(self, h: int, w: int, fps: float, shift=None):
        scale = min(h, w)
        if shift is None:
            shift = scale // 60

        self.h = h
        self.w = w
        self.shift = shift
        self.fps = fps

        self.back_noise = RGBNoise(h, w, scale * 0.05, fps * 0.2, 4, Noise.Type.SIMPLEX)
        self.front_noise = RGBNoise(
            h, w, scale * 0.05, fps * 0.2, 4, Noise.Type.SIMPLEX
        )

        coords = generate_coords(h, w)
        self.anchor = (
            (coords - np.asarray([scale * 0.02, w / 2])[None, None]) ** 2
        ).sum(axis=-1) < (
            scale * 0.015
        ) ** 2  # 生成圆形视角锚点

    def step(self, frame: np.ndarray):
        frame = frame * self.shift
        mask = frame >= 1
        back_image = self.back_noise.next()
        front_image = np.zeros_like(back_image)
        if mask.sum() > 0:
            front_image[mask] = self.front_noise.next(mask)

        tmp = np.hstack((back_image, back_image))
        left = tmp[:, : self.w, :]
        right = tmp[:, self.w :, :]
        for shift in range(1, self.shift):
            mask = np.logical_and(shift <= frame, frame < shift + 1)
            res_w = self.w - shift
            res_mask = mask[:, shift:]
            left[mask] = front_image[mask]
            right[:, :res_w, :][res_mask] = front_image[:, shift:, :][res_mask]

        left[self.anchor] = 255
        right[self.anchor] = 255

        return tmp

    @property
    def result_shape(self):
        return (self.h, self.w * 2, 3)


class StereoVideo:
    def __init__(self, h: int, w: int, fps: float, split=8, shift=None):
        scale = min(h, w)
        if shift is None:
            shift = scale // 60
        if w % split != 0:
            raise ValueError("Width must be divisible by split")

        self.h = h
        self.w = w
        self.shift = shift
        self.fps = fps
        self.split = split
        self.split_w = w // split

        self.back_noise = RGBNoise(h, w, scale * 0.02, fps * 0.2, 4)

        coords = generate_coords(h, self.split_w)
        self.anchor = (
            (coords - np.asarray([scale * 0.02, self.split_w / 2])[None, None]) ** 2
        ).sum(axis=-1) < (
            scale * 0.015
        ) ** 2  # 生成圆形视角锚点

    def step(self, frame: np.ndarray):
        back_image = self.back_noise.next()

        result = np.zeros(self.result_shape, dtype=np.uint8)
        result[:, : self.split_w] = back_image
        for split in range(self.split):
            result_area = result[
                :, (split + 1) * self.split_w : (split + 2) * self.split_w
            ]
            # 复制前一块的图像到当前块
            result_area[:] = result[
                :, split * self.split_w : (split + 1) * self.split_w
            ]
            for shift in range(1, self.shift):
                frame_area = frame[:, split * self.split_w : (split + 1) * self.split_w]
                mask = np.logical_and(
                    (shift / self.shift) <= frame_area,
                    frame_area < (shift + 1) / self.shift,
                )
                raise NotImplementedError("Please implement this method")

            if split == self.split // 2 or split == self.split // 2 - 1:
                result_area[self.anchor] = 255

    @property
    def result_shape(self):
        return (self.h, self.w // self.split * (self.split + 1), 3)


def process_videos(input, output, target_height=360, frames=1_000_000, **kwargs):
    # 获取视频信息（假设两个视频参数相同）
    w, h, fps = get_video_info(input)
    w, h = int(w * target_height / h), target_height

    filter = DualVideo(h, w, fps, **kwargs)

    input_kwargs = {"input_file": input, "output_args": {"vf": f"scale={w}:{h}"}}
    output_kwargs = {
        "output_file": output,
        "input_args": {
            "s": f"{'x'.join(map(str, filter.result_shape[1::-1]))}",
            "r": f"{fps}",
        },
        "output_args": {},
    }
    if True:
        # intel qsv encoder
        output_kwargs["output_args"] |= {
            "c:v": "h264_qsv",
            "preset": "medium",
            "global_quality": "10",
            "look_ahead": "1",
        }
    else:
        # libvpx encoder
        output_kwargs["output_args"] |= {
            "c:v": "libx264",
            "preset": "medium",
            "crf": "10",
            "look_ahead": "1",
        }

    input_Video = Video(input_file=input, output_args={"vf": f"scale={w}:{h}"})
    output_Video = Video(**output_kwargs)

    with input_Video as video_in, output_Video as video_out:
        count = 0
        for frame in video_in:
            # 转换为灰度图像
            frame = frame.mean(-1)

            # coords = generate_coords(h, w)
            # dist = (((coords - [h / 2, w / 2])) ** 2).sum(axis=-1) ** 0.5
            # frame = np.sin(dist * 0.2 + count * -0.1) * 128 + 128

            cv2.imshow("input", frame)
            cv2.waitKey(1)

            # 转换帧
            result = filter.step(frame.astype(np.float32) / 256)

            cv2.imshow("result", result)
            cv2.waitKey(1)

            # 写入输出流
            video_out.write(result)

            count += 1
            if count > frames:
                break


if __name__ == "__main__":
    process_videos(
        r"badapple.mp4",
        "output.mp4",
        target_height=240,
        frames=200,
    )
