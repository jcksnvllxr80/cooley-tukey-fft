# This is a sample Python script.
from typing import List
import numpy as np
from math import sqrt, pi, ceil, pow, log


def do_fft(input_list) -> None:
    print(f'input, {input_list}')
    fft_output_list = fft(input_list)
    print(f'fft output: {approximate_complex_list(fft_output_list)}')
    ifft_output_list = ifft(fft_output_list)
    print(f'ifft output: {approximate_list(realize_list(ifft_output_list))}')


def do_convolution(input1, input2) -> None:
    print(f'input1, {input1}; input1, {input2}')
    convolution_output_list = convolve(input1, input2)
    print(f'convolution output: {convolution_output_list}')


def do_multiplication(input1, input2) -> None:
    print(f'input1, {input1}; input1, {input2}')
    convolution_output_list = convolve(input1, input2)
    print(f'multiplication output: {convolution_output_list}')


def approximate_complex_list(list_of_complex_floats) -> List:
    return list(map(lambda x: np.round(x), list_of_complex_floats))


def approximate_list(list_of_floats) -> List:
    return list(map(lambda x: int(np.round(x)), list_of_floats))


def realize_list(list_of_complex_nums) -> List:
    return [abs(i) for i in list_of_complex_nums]


def is_power_of_two(num) -> bool:
    sq_root = sqrt(num)
    # print(f"num -> {num}")
    return sq_root == int(sq_root)


def next_square(n) -> int:
    # print(f"n -> {n}")
    return int(pow(2, ceil(log(n, 2))))


def zero_pad_left(list_to_pad, num_zeros) -> List:
    for z in [0]*num_zeros:
        list_to_pad.insert(0, z)
    return list_to_pad


def fft(p) -> List:
    """
    Given an array of coefficients, p,
    Recursively perform a Cooley-Tukey Fast Fourier Transform
    """
    n = len(p)
    if n == 1:
        return p

    if not is_power_of_two(n):
        p = zero_pad_left(p, next_square(n) - len(p))
        n = len(p)

    n_over_two = int(n / 2)

    w = np.exp(2j * pi * np.arange(n) / n)
    # print(f"w -> {w}")

    p_e, p_o = p[::2], p[1::2]  # even_powered_coeffs, odd_powered_coeffs
    y_e, y_o = fft(p_e), fft(p_o)

    y = [0] * n

    for i in range(n_over_two):
        y[i] = y_e[i] + (w[i] * y_o[i])
        # print(f"y[{i}] = {y_e[i] + (w[i] * y_o[i])}")
        y[n_over_two + i] = y_e[i] - (w[i] * y_o[i])
        # print(f"y[{n_over_two + i}] = {y_e[i] - (w[i] * y_o[i])}")
    return y


def ifft(p) -> List:
    """
    Given an array of coefficients, p,
    Recursively perform an inverse Cooley-Tukey Fast Fourier Transform
    """
    n = len(p)
    if n == 1:
        return p

    if not is_power_of_two(n):
        zero_pad_left(p, next_square(n) - len(p))
        n = len(p)

    n_over_two = int(n / 2)

    w = np.exp(-2j * pi * np.arange(n) / n)
    # print(f"w -> {w}")

    p_e, p_o = p[::2], p[1::2]  # even_powered_coeffs, odd_powered_coeffs
    y_e, y_o = ifft(p_e), ifft(p_o)

    y = [0] * n

    for i in range(n_over_two):
        y[i] = (y_e[i] + (w[i] * y_o[i]))/2
        # print(f"y[{i}] = {y[i]}")
        y[n_over_two + i] = (y_e[i] - (w[i] * y_o[i]))/2
        # print(f"y[{n_over_two + i}] = {y[n_over_two + i]}")

    return y


def convolve(list1, list2) -> List:
    return ifft(multiply_pointwise(fft(list1), fft(list2)))


def multiply_pointwise(list1, list2) -> List:
    list1_len = len(list1)
    list2_len = len(list2)
    if list1_len != list2_len:
        if list1_len > list2_len:
            zero_pad_left(list2, list1_len - list2_len)
        else:
            zero_pad_left(list1, list2_len - list1_len)

    print(f"list1: {list1}; list2: {list2}")
    return [list1[i] * list2[i] for i in range(len(list1))]


if __name__ == '__main__':
    do_fft([5, 4, 3, 2, 1])
    # do_convolution([1, 1, 0], [1, 0, 0, 1, 0])
    # do_multiplication([1, 2], [6])
