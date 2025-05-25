import numpy as np
import cv2

from noiser import Noise, RGBNoise
from video import get_video_info, Video


class StereoVideo:
    def __init__(self, h: int, w: int, fps: float, split=5, shift=None):
        scale = min(h, w)
        if shift is None:
            shift = scale * 0.015
        if w % split != 0:
            raise ValueError("Width must be divisible by split")

        self.h = h
        self.w = w
        self.shift = shift
        self.split = split
        self.split_w = w // split

        self.back_noise = RGBNoise(
            h, self.split_w, scale * 0.03, fps * 0.3, 4, type=Noise.Type.SIMPLEX
        )

        self.anchor = (
            (np.arange(self.h)[:, None] - scale * 0.02) ** 2
            + (np.arange(self.split_w) - self.split_w / 2) ** 2
        ) < (
            scale * 0.015
        ) ** 2  # 生成圆形平行眼锚点

    def apply(self, frame: np.ndarray):
        shift_mat = (1 - frame) * self.shift

        result = np.zeros((*self.result_shape, 3), dtype=np.float32)

        x_map = np.arange(self.h, dtype=np.float32)[:, None]
        y_map = np.arange(self.split_w * 2, dtype=np.float32)
        tmp = self.back_noise.next(x_map, y_map)
        result[:, : self.split_w] = (
            tmp[:, : self.split_w] * np.linspace(0.0, 1.0, self.split_w)[:, None]
        )
        result[:, : self.split_w] += (
            tmp[:, self.split_w :] * np.linspace(1.0, 0.0, self.split_w)[:, None]
        )

        for split in range(self.split):
            result_split = result[
                :, (split + 1) * self.split_w : (split + 2) * self.split_w
            ]
            result_split[result_split == 0] = result[
                :, split * self.split_w : (split + 1) * self.split_w
            ][result_split == 0]
            for shift in range(int(shift_mat.max().item()) + 1):
                area_start = max(0, split * self.split_w - shift)
                area_end = (split + 1) * self.split_w - shift
                area_length = area_end - area_start
                shift_area = shift_mat[:, area_start:area_end]
                # mask = np.logical_and(shift <= shift_area, shift_area < shift + 1)
                mask = np.logical_and(shift <= shift_area, shift_area < shift + 2)

                result_split[:, -area_length:][mask] = result[:, area_start:area_end][
                    mask
                ]

        result[
            :, (self.split // 2) * self.split_w : (self.split // 2 + 1) * self.split_w
        ][self.anchor] = 0.999
        result[
            :,
            (self.split // 2 + 1) * self.split_w : (self.split // 2 + 2) * self.split_w,
        ][self.anchor] = 0.999

        return result

    @property
    def result_shape(self):
        return (self.h, self.w // self.split * (self.split + 1))


def process_videos(
    input, output, target_height=360, frames: int | None = None, **kwargs
):
    # 获取视频信息（假设两个视频参数相同）
    w, h, fps = get_video_info(input)
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
    if True:
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
            "look_ahead": "1",
        }

    input_Video = Video(**input_kwargs)
    output_Video = Video(**output_kwargs)

    with input_Video as video_in, output_Video as video_out:
        count = 0
        for frame in video_in:
            count += 1
            if frames is not None and count > frames:
                break

            # if count < 100:
            #     continue

            # 转换为灰度图像，归一化
            frame = frame.mean(-1, dtype=np.float32) / 256
            # 平滑化图像
            frame = cv2.GaussianBlur(frame, (0, 0), h / 360)

            cv2.imshow("input", frame)
            cv2.waitKey(1)

            # 转换帧
            result = filter.apply(frame)

            cv2.imshow("result", result)
            cv2.waitKey(1)

            result = (result * 256).astype(np.uint8)
            # 写入输出流
            video_out.write(result)


if __name__ == "__main__":
    process_videos(
        r"badapple.mp4",
        "output.mp4",
        # target_height=240,
        frames=200,
        # split=1,
    )
