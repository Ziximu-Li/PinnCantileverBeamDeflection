import imageio.v2 as imageio

# 修改帧持续时间和图片数量来调整GIF效果
num_frames = 11  # 帧数
frame_duration = 1500  # 每帧持续时间，单位为毫秒

with imageio.get_writer(uri='Cantilever_softplus.gif', mode='I', duration=frame_duration, loop=0) as writer:
    for i in range(num_frames):
        writer.append_data(imageio.imread(f'Cantilever_{i}.png'))
