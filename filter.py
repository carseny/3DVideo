import numpy as np
import cv2

from noiser import Noise, RGBNoise
from video import get_video_info, Video

chunk_size = 16384


class StereoVideo:
    def __init__(
        self,
        h: int,
        w: int,
        fps: float,
        split=5,
        shift=None,
        shift_scale=0.015,
        hw_scale=0.02,
        t_scale=0.4,
        octave=4,
        noise_type=Noise.Type.SIMPLEX,
        fast_render=False,
    ):
        scale = min(h, w)
        if shift is None:
            shift = scale * shift_scale
        # if w % split != 0:
        #     raise ValueError("Width must be divisible by split")

        self.h = h
        self.w = w
        self.shift = shift
        self.split = split
        self.split_w = w // split
        self.fast_render = fast_render

        self.back_noise = RGBNoise(
            h, self.split_w, scale * hw_scale, fps * t_scale, octave, noise_type
        )

        self.anchor = (
            (np.arange(self.h)[:, None] - scale * 0.02) ** 2
            + (np.arange(self.split_w) - self.split_w / 2) ** 2
        ) < (
            scale * 0.015
        ) ** 2  # 生成圆形平行眼锚点

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """apply stereo filter to frame
        Args:
            frame (np.ndarray): depth frame (h, w)
        Returns:
            np.ndarray: Stereo graph (h, w, 3)
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
                        borderValue=0,  # 填充值（可选）
                    )
                    * self.shift
                )
            y_map[mask] += shift + self.split_w
        y_map -= self.w
        if self.fast_render:
            tmp = self.back_noise.next(
                np.arange(self.h, dtype=np.float32)[:, None],
                np.arange(int(y_map.max()) + 1, dtype=np.float32),
            )
            result[...] = tmp[x_map.astype(np.int32), y_map.astype(np.int32)]
        else:
            result[:, ...] = self.back_noise.next(x_map, y_map)

        result[
            :,
            (self.split // 2) * self.split_w : (self.split // 2 + 1) * self.split_w,
        ][self.anchor] = 0.999
        result[
            :,
            (self.split // 2 + 1) * self.split_w
            + int(self.shift/2) : (self.split // 2 + 2) * self.split_w
            + int(self.shift/2),
        ][self.anchor] = 0.999

        return result

    @property
    def result_shape(self):
        return (self.h, self.split_w + self.w)


def process_videos(
    input,
    output,
    target_height=None,
    frames: int | None = None,
    qsv=False,
    display=False,
    **kwargs,
):
    # 获取视频信息（假设两个视频参数相同）
    w, h, fps = get_video_info(input)
    if target_height is not None:
        w, h = int(w * target_height / h), target_height

    filter = StereoVideo(h, w, fps, **kwargs)

    input_kwargs = {"input_file": input, "output_args": {"vf": f"scale={w}:{h}"}}
    output_kwargs = {
        "output_file": output,
        "input_args": {
            "s": f"{'x'.join(map(str, filter.result_shape[1::-1]))}",
            "r": f"{fps}",
        },
        "output_args": {},
    }
    if qsv:
        # intel qsv encoder
        output_kwargs["output_args"] |= {
            "c:v": "h264_qsv",
            "preset": "slow",
            "global_quality": "10",
            "look_ahead": "1",
        }
    else:
        # libvpx encoder
        output_kwargs["output_args"] |= {
            "c:v": "libx264",
            "preset": "slow",
            "crf": "10",
        }
    input_Video = Video(**input_kwargs)
    output_Video = Video(**output_kwargs)

    with input_Video as video_in, output_Video as video_out:
        count = 0
        for frame in video_in:
            count += 1
            if frames is not None and count > frames:
                break

            # 转换为灰度图像，归一化
            frame = frame.mean(-1, dtype=np.float32) / 256

            if display:
                cv2.imshow("input", frame)
                cv2.waitKey(1)

            # 转换帧
            output = filter.apply(frame)

            if display:
                cv2.imshow("output", output)
                cv2.waitKey(1)

            output = (output * 256).astype(np.uint8)
            # 写入输出流
            video_out.write(output)
