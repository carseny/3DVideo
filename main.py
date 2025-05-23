import subprocess as sp
import numpy as np
import shlex
import cv2
from noise import pnoise3, snoise3
from scipy.stats import norm


class Noise:
    def __init__(self, width, height, scale=10.0, t_scale=10.0, octaves=1):
        self.width = width
        self.height = height
        self.scale = scale
        self.t_scale = t_scale
        self.octaves = octaves
        self.x, self.y, self.t = np.random.rand(3) * 10000

        # 向量化计算噪声
        self.v_pnoise2 = np.vectorize(
            lambda x, y, t: pnoise3(x, y, t, octaves=self.octaves)
        )

        # # 生成坐标网格
        # x = np.arange(self.width) + self.x
        # y = np.arange(self.height) + self.y
        # self.xx, self.yy = np.meshgrid(x / self.scale, y / self.scale)

    def get(
        self,
        x=None,
        y=None,
        t=None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if t is None:
            t = self.t
        x = np.arange(self.width) / self.scale + x
        y = np.arange(self.height) / self.scale + y
        xx, yy = np.meshgrid(x, y)
        if mask is None:
            noise = self.v_pnoise2(xx, yy, t)
        else:
            # 生成噪声图
            noise = np.zeros((self.height, self.width))
            noise[mask] = self.v_pnoise2(xx[mask], yy[mask])
        return noise

    def next(self, mask: np.ndarray | None = None) -> np.ndarray:
        self.t += 1 / self.t_scale
        return self.get(self.x, self.y, self.t, mask=mask)


class StereoVideo:
    def __init__(self, h: int, w: int, fps: float, shift=None):
        scale = min(h, w)
        if shift is None:
            shift = scale // 60
            
        self.height = h
        self.width = w
        self.shift = shift
        n = 4
        self.boundary = list(np.linspace(0, 1, num=2**n)) + [1]
        
        self.noiser = Noise(w, h, scale * 0.13, fps * 0.5)
        # self.noisers = [Noise(w, h, scale * 0.13, fps * 0.5) for _ in range(n)]

        image_noise = Noise(w, h, scale=scale * 0.01)
        images = [
            np.stack(
                [(image_noise.get(t=t) * 128 + 128).astype(np.uint8) for t in rgb],
                axis=-1,
            )
            for rgb in (np.random.rand(2**n * 2, 3) * 10000)
        ]
        # images = np.random.randint(
        #     0, 256, size=(2**n * 2, h, w, 3), dtype=np.uint8
        # )
        self.front_images = images[: 2**n]
        self.back_images = images[2**n :]

        coords = np.empty((h, w, 2), dtype=np.float64)
        coords[..., 0] = np.arange(h)[:, None]  # 列索引作为x坐标
        coords[..., 1] = np.arange(w)  # 行索引作为y坐标
        self.anchor = (
            (coords - np.asarray([scale * 0.02, w / 2])[None, None]) ** 2
        ).sum(axis=-1) < (
            scale * 0.015
        ) ** 2  # 生成圆形视角锚点

    def step(self, mask: np.ndarray):
        noise = np.broadcast_to(
            self.noiser.next()[..., None], (self.height, self.width, 3)
        )
        noise = norm.cdf(noise, loc=0, scale=1 / 2)

        cv2.imshow("noise", noise)
        cv2.waitKey(1)

        condlist = []
        for lower, upper in zip(self.boundary[:-1], self.boundary[1:]):
            condlist.append(np.logical_and(lower <= noise, noise < upper))
        back_image = np.select(condlist, list(self.back_images))
        front_image = np.select(condlist, list(self.front_images))

        tmp = np.hstack((back_image, back_image))
        left = tmp[:, : self.width, :]
        right = tmp[:, self.width :, :]

        left[mask] = front_image[mask]
        res_w = self.width - self.shift
        res_mask = mask[:, :res_w]
        right[:, :res_w, :][res_mask] = front_image[:, self.shift :, :][res_mask]

        left[self.anchor] = 255
        right[self.anchor] = 255

        return tmp


def generate_coords(m, n) -> np.ndarray:
    coords = np.empty((m, n, 2), dtype=np.int32)
    coords[..., 0] = np.arange(m)[:, None]  # 列索引作为x坐标
    coords[..., 1] = np.arange(n)  # 行索引作为y坐标
    return coords


def get_video_info(input_file):
    """使用 ffprobe 获取视频信息"""
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "{input_file}"'
    output = sp.check_output(shlex.split(cmd)).decode().strip()
    width, height, fps = output.split(",")
    return int(width), int(height), eval(fps)  # 安全地转换分数


def process_videos(
    input, output, target_height=360, frames=1_000_000, **kwargs
):
    # 获取视频信息（假设两个视频参数相同）
    w, h, fps = get_video_info(input)
    scale = target_height / h
    w, h = int(w * scale), int(h * scale)
    frame_size = w * h * 3  # RGB24

    # 创建 FFmpeg 进程
    read_cmd = ["-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    ffmpeg_cmd = ["ffmpeg", "-i", input] + read_cmd
    # 输出视频配置
    output_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w*2}x{h}",  # 根据合并结果调整尺寸
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "h264_qsv",  # 使用 QSV 编码器
        "-preset",
        "medium",  # QSV 的 preset 可选：veryfast, faster, fast, medium, slow
        "-global_quality",
        "1",  # 替代 crf（QSV 用 global_quality 控制质量，范围 1-51）
        "-look_ahead",
        "1",  # 可选：启用前瞻优化（0=关闭，1=开启）
        # "-c:v",
        # "libx264",
        # "-preset",
        # "fast",
        # "-crf",
        # "23",
        output,
    ]

    with sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, stderr=sp.DEVNULL) as proc_in:
        with sp.Popen(output_cmd, stdin=sp.PIPE) as proc_out:
            video = StereoVideo(h, w, fps, **kwargs)

            for count in range(frames):
                # 读取原始帧数据
                raw = proc_in.stdout.read(frame_size)
                if not raw:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))

                # 转换为 numpy 数组
                # 合并帧
                merged = video.step((frame.mean(-1) < 128))

                cv2.imshow("Merged", merged)
                cv2.waitKey(1)

                # 写入输出流
                proc_out.stdin.write(merged.tobytes())


if __name__ == "__main__":
    process_videos(
        r"E:\Videos\csy\material\BadApple\output_video.mp4",
        "output.mp4",
        target_height=240,
        frames=200,
    )
