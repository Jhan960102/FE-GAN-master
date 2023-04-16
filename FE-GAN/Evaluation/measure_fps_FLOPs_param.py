# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: measure_fps_FLOPs_param.py
# @Date: 2022/11/7 10:17

'''
calculate the fps, FLOPs, number of parameters of each model
'''


import time
import torch
from torchinfo import summary


def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps


from PyTorch.nets import fegan        # generator of FE-GAN

if __name__ == '__main__':
    # fps
    print("fps of model...")
    net = fegan.Generator().cuda()
    data = torch.randn((1, 3, 256, 256)).cuda()
    measure_inference_speed(net, (data, ))
    print()

    # FLOPs and parameters
    print('FLOPs and parameters...')
    model = fegan.Generator().to('cuda')
    in_3 = (1, 3, 256, 256)
    summary(model, in_3)