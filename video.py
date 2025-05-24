import subprocess as sp
import numpy as np
import shlex
from collections import OrderedDict


class Video:
    def __init__(
        self,
        input_file: str | None = None,
        output_file: str | None = None,
        input_args: dict[str, str] | None = None,
        output_args: dict[str, str] | None = None,
    ):
        if input_args is None:
            input_args = {}
        else:
            input_args = OrderedDict(**input_args)
        if output_args is None:
            output_args = {}
        else:
            output_args = OrderedDict(**output_args)

        if input_file is None:
            input_args |= OrderedDict(f="rawvideo", pix_fmt="rgb24", i="-")
        else:
            input_args |= OrderedDict(i=input_file)
        if output_file is None:
            output_args |= OrderedDict(f="rawvideo", pix_fmt="rgb24", y="-")  # -y
        else:
            output_args |= OrderedDict(y=output_file)  # -y

        cmd = ["ffmpeg"]
        for k, v in (input_args | output_args).items():
            cmd.append("-" + k)
            if v:
                cmd.append(v)

        print(shlex.join(cmd))

        self.proc = sp.Popen(
            cmd,
            stdin=sp.PIPE if input_file is None else None,
            stdout=sp.PIPE if output_file is None else None,
            stderr=sp.DEVNULL if output_file is None else None,
        )

        if input_file is None:
            self.out_w, self.out_h = map(int, input_args.get("s", "0x0").split("x"))
            self.out_fps = float(input_args.get("r", "0"))
            # 计算每一帧的字节数
            self.out_size = self.out_w * self.out_h * 3
        if output_file is None:
            self.in_w, self.in_h, self.in_fps = get_video_info(input_file)
            vf = output_args.get("vf", "")
            vf = dict(filter.split("=") for filter in vf.split(","))
            if scale := vf.get("scale"):
                self.in_w, self.in_h = map(int, scale.split(":"))
            # 计算每一帧的字节数
            self.in_size = self.in_w * self.in_h * 3

    def __enter__(self):
        self.proc = self.proc.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame is None:
            raise StopIteration()
        return frame

    def read(self) -> np.ndarray | None:
        if self.proc.stdout is None:
            raise ValueError("Cannot read from an output stream")
        # 读取原始帧数据
        raw = self.proc.stdout.read(self.in_size)
        if not raw:
            return None
        # 转换为 numpy 数组
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.in_h, self.in_w, 3))
        return frame

    def write(self, frame: np.ndarray):
        if self.proc.stdin is None:
            raise ValueError("Cannot write to an input stream")
        bytes_ = frame.tobytes()
        if len(bytes_) != self.out_size:
            raise ValueError("Frame size does not match the expected size")
        # 将 numpy 数组转换为原始帧数据并写入管道
        self.proc.stdin.write(bytes_)


def get_video_info(input_file):
    """使用 ffprobe 获取视频信息"""
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "{input_file}"'
    output = sp.check_output(shlex.split(cmd)).decode().strip()
    width, height, fps = output.split(",")
    return int(width), int(height), eval(fps)

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
        "10",  # 替代 crf（QSV 用 global_quality 控制质量，范围 1-51）
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
            filter = DualVideo(h, w, fps, **kwargs)
