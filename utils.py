import math


def output_shape(inp_tensor: tuple, kernel_size: tuple, stride: tuple, padding: tuple, batch_size=1, output_channel=1):
    return batch_size, \
           output_channel, \
           math.floor((inp_tensor[-2] - kernel_size[-2] + padding[-2] + stride[-2]) / stride[-2]), \
           math.floor((inp_tensor[-1] - kernel_size[-1] + padding[-1] + stride[-1]) / stride[-1])
