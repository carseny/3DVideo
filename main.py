import numpy as np
import cv2

from noiser import Noise, RGBNoise
from video import get_video_info, Video


class DualVideo:
    def __init__(self, h: int, w: int, fps: float, shift=None):
        scale = min(h, w)
        if shift is None:
            shift = scale * 0.01

        self.h = h
        self.w = w
        self.shift = shift

        self.back_noise = RGBNoise(
            h, w, scale * 0.03, fps * 0.3, 4, type=Noise.Type.SIMPLEX
        )
        # self.noise = Noise(h, w, scale * 0.05, fps * 0.2, 2, type=Noise.Type.SIMPLEX)
        # images = [
        #     RGBNoise(h, w, scale * 0.03, fps * 0.3, 4, type=Noise.Type.SIMPLEX).next()
        #     for _ in range(4)
        # ]
        # self.back_images = images[:2]
        # self.front_images = images[:2]

        self.anchor = (
            (np.arange(self.h)[:, None] - scale * 0.02) ** 2
            + (np.arange(self.w) - w / 2) ** 2
        ) < (
            scale * 0.015
        ) ** 2  # 生成圆形平行眼锚点

    def step(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: np.ndarray, shape=(h, w), dtype=float32, [0,1]
        """
        shift_mat = frame * self.shift

        # noise = self.noise.next()[..., None]
        # back_image = np.select([noise < 0, 0 <= noise], self.back_images)
        # front_image = np.select([noise < 0, 0 <= noise], self.front_images)
        # cv2.imshow("noise", (noise > 0).astype(np.float32))

        result = np.zeros((*self.result_shape, 3), dtype=np.float32)

        y_shifted = np.zeros((self.h, self.w), dtype=np.float32)
        for shift in range(1, int(shift_mat.max().item()) + 1):
            mask = np.logical_and(shift <= shift_mat, shift_mat < shift + 1)
            res_w = self.w - shift
            res_mask = mask[:, shift:]
            y_shifted[:, :res_w][res_mask] = shift
        y_shifted[...] += np.arange(self.w)
        y_shifted[...] = np.maximum.accumulate(y_shifted, axis=1)

        x_map = np.arange(self.h, dtype=np.float32)[:, None]
        y_map = np.hstack(
            (np.broadcast_to(np.arange(self.w), (self.h, self.w)), y_shifted),
            dtype=np.float32,
        )
        result[...] = self.back_noise.next(x_map, y_map)

        result[:, : self.w, :][self.anchor] = 255
        result[:, self.w :, :][self.anchor] = 255

        return result

    @property
    def result_shape(self):
        return (self.h, self.w * 2)


class StereoVideo:
    def __init__(self, h: int, w: int, fps: float, split=4, shift=None):
        scale = min(h, w)
        if shift is None:
            shift = scale * 0.01
        if w % split != 0:
            raise ValueError("Width must be divisible by split")

        self.h = h
        self.w = w
        self.shift = shift
        self.split = split
        self.split_w = w // split

        self.back_noise = RGBNoise(
            h, w, scale * 0.05, fps * 0.3, 4, type=Noise.Type.SIMPLEX
        )

        self.anchor = (
            (np.arange(self.h)[:, None] - scale * 0.02) ** 2
            + (np.arange(self.split_w) - self.split_w / 2) ** 2
        ) < (
            scale * 0.015
        ) ** 2  # 生成圆形平行眼锚点

    def step(self, frame: np.ndarray):
        shift_mat = (1 - frame) * self.shift

        result = np.zeros((*self.result_shape, 3), dtype=np.float32)

        x_map = np.arange(self.h, dtype=np.float32)[:, None]
        y_map = np.arange(self.split_w, dtype=np.float32)
        half = self.split_w // 2
        tmp = self.back_noise.next(x_map, y_map)
        tmp[:, : self.split_w] *= (1 - np.abs(np.linspace(-1.0, 1.0, self.split_w)))[
            :, None
        ]
        result[:, : self.split_w] = tmp
        result[:, :half] += tmp[:, -half:]
        result[:, half : self.split_w] += tmp[:, :half]

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
        ][self.anchor] = 255
        result[
            :,
            (self.split // 2 + 1) * self.split_w : (self.split // 2 + 2) * self.split_w,
        ][self.anchor] = 255

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

    # filter = DualVideo(h, w, fps, **kwargs)
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
            frame = cv2.GaussianBlur(frame, (0, 0), h / 240)

            cv2.imshow("input", frame)
            cv2.waitKey(1)

            # 转换帧
            result = filter.step(frame)

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
        # frames=200,
        # split=1,
    )
