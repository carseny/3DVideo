from modules import *


if __name__ == "__main__":
    process_video(
        # 输入视频路径
        r"input.mp4",
        # 输出视频路径
        "output.mp4",
        # 目标视频缩放高度
        target_height=360,
        # 需要处理的帧数，默认处理所有帧
        num_frame=None,
        # 是否显示处理过程中的视频
        display=True,
        # 立体图像分割数（大于1会有重影）
        split=5,
        # 最大偏移像素个数，控制视差大小
        shift=None,
        # 偏移像素相对比例，仅在shift为None时有效
        shift_scale=0.05,
        # 是否降低深度精度来加速渲染（会导致立体效果下降）
        fast_render=False,
    )
