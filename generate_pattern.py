# -*- coding: utf-8 -*-
"""
Author: Pan Zhang
"""
from superpixel_utility import laguerre_gaussian, phase_to_superpixel, plot_gaussian_laguerre_cartesian, plot_phase_diagram,bessel_beam
import numpy as np
import scipy
class DmdPattern():
    def __init__(self,pattern: str, width: int, height: int, gray_scale: int=255):
        """
        A class to generate mask pattern for DMD
        :param pattern: the name of pattern. Can be "hadamard" or "random"
        :param width: the width of image
        :param height: the height of image
        :param gray_scale: the gray scale of image. Range from 0 to 255.
        """
        self.pattern = pattern
        self.width = width
        self.height = height
        self.hadmard_size = width**2
        self.gray_scale = gray_scale
    def execute(self,length=1, random_sparsity:int = 1):
        """
        The execution function for mask generation.
        :param random_sparsity: The percentage of elements in random mask to be 1.
        :return:
        Under hadamard mode:
        List: A list contain whole set of hadamard pattern that drive the DMD to point to left.
        List: A list contain whole set of hadamard pattern that drive the DMD to point to right.
        Under random mode:
        nd.array: A single random pattern with given sparsity that point left.
        nd.array: A single random pattern with given sparsity that point right.
        """
        print("spcae")
        if self.pattern == "hadamard":
            positive_image = hadmard_matrix(self.hadmard_size)
            if self.hadmard_size <128:
                positive_image = walsh_to_hadmard_mask(positive_image)
            elif self.hadmard_size>128:
                positive_image = walsh_to_hadmard_mask(positive_image)
            def to_conjugate(pattern):
                pattern = (pattern==0).astype(int).astype(np.uint8)
                return pattern
            positive_image_list = list(positive_image)
            negative_image_list = list(map(to_conjugate, positive_image_list))
            return  positive_image_list[:int(len(positive_image)*length)], negative_image_list[:int(len(positive_image)*length)]



        elif self.pattern == "random":
            positive_image_list = []
            negative_image_list = []
            for i in range(self.width*self.height*length):
                positive = random_pattern(self.width, self.height, random_sparsity)
                negative = (positive==0).astype(int)
                positive_image_list.append(positive)
                negative_image_list.append(negative)
            return  positive_image_list, negative_image_list

        elif self.pattern == "raster":
            positive_image_list = []
            negative_image_list = []
            for i in range(self.width):
                for j in range(self.height):
                    positive = np.zeros((self.width, self.height))
                    positive[i][j] = 1
                    negative = (positive==0).astype(int)
                    positive_image_list.append(positive.astype(np.uint8))
                    negative_image_list.append(negative.astype(np.uint8))
            return  positive_image_list, negative_image_list
        elif self.pattern == "fourier":
            fourier_mask = []
            phase = [0,np.pi/2, np.pi, np.pi*3/2]
            for i in range(self.width):
                for j in range(self.height):
                    for k in range(len(phase)):
                        fourier_mask.append(generate_fourier_mask(i,j,self.width,phase[k]))
            return fourier_mask
class SuperpixelPattern:
    def __init__(self, superpixel_size=4):
        self.x = np.e**(np.arange(0,superpixel_size**2,1)*2*np.pi/superpixel_size**2*1j)/superpixel_size**2
        self.superpixel_size = superpixel_size
    def superpixel_pattern(self, phase_matrix):
        target_matrix,_ = phase_to_superpixel(phase_matrix, self.x)
        return target_matrix[:, :, np.newaxis].astype(np.uint8)

    def plane_wave(self, phase):
        return_array = np.exp(1j*phase)*np.ones((228, 285))
        target_matrix,_ = phase_to_superpixel(return_array, self.x)
        return target_matrix[:, :, np.newaxis].astype(np.uint8)

    def alignment_pattern_horizontal(self):
        a = np.ones((912, 4)) * 255
        b = np.zeros((912, 4))
        return_array = np.zeros((912, 0))
        for i in range(143):
            return_array = np.hstack((return_array, a))
            return_array = np.hstack((return_array, b))
        return return_array[:, :, np.newaxis].astype(np.uint8)

    def alignment_pattern_vertical(self):
        a = np.ones((4, 1140)) * 255
        b = np.zeros((4, 1140))
        return_array = np.zeros((0, 1140))
        for i in range(114):
            return_array = np.vstack((return_array, a))
            return_array = np.vstack((return_array, b))
        return return_array[:, :, np.newaxis].astype(np.uint8)
    def LG_mode(self,l=2 ,p=1, range=(-0.004,0.004), plot=True):
        min,max = range
        x = np.linspace(min, max, 285)+4/285*0.01
        y = np.linspace(min, max, 228)+4/228*0.01
        X, Y = np.meshgrid(x, y)
        Z = laguerre_gaussian(x=X, y=Y, z=0.001, w0=0.001, k=2*np.pi/(633*10**-9), l=l, p=p)
        intensity = np.abs(Z)**2
        print(np.max(Z))
        target_matrix,_ = phase_to_superpixel(Z, self.x)
        if plot:
            plot_phase_diagram(Z)
            plot_gaussian_laguerre_cartesian(l,p, X, Y, intensity)
        return target_matrix, Z
    def B_mode(self,m = 1, range=(-0.000003,0.000003), plot=True):
        min,max = range
        x = np.linspace(min, max, 285)+4/285*max
        y = np.linspace(min, max, 228)+4/228*max
        X, Y = np.meshgrid(x, y)
        bessel_beam_matrix = bessel_beam(x=X, y=Y, z=0.01, w0=0.001, k=2 * np.pi / (633 * 10 ** -9), m=m)
        intensity = np.abs(bessel_beam_matrix)**2
        print(np.max(intensity))
        target_matrix,error = phase_to_superpixel(bessel_beam_matrix, self.x)
        if plot:
            plot_phase_diagram(bessel_beam_matrix)
            plot_gaussian_laguerre_cartesian(1,0, X, Y, intensity)
        return target_matrix, bessel_beam_matrix, error
    def scanning(self,position, input_matrix):
        nth_x, nth_y = position
        temp_matrix = np.ones((912,1140))*255
        target_matrix = input_matrix[nth_x*4:nth_x*4+4, nth_y*4:nth_y*4+4]
        #print(target_matrix)
        temp_matrix[nth_x*4:nth_x*4+4, nth_y*4:nth_y*4+4] = target_matrix
        return temp_matrix
    def plane_wave_scanning(self,x_position, row_num=285, column_num=228):
        #If plane wave, turn on first micromirror
        #if pi phase shift, turn on the eigth micromirror
        plane_wave = np.array([[255,255,255,255],[0,0,0,0],[0,0,0,0],[0,255,255,255]])
        phase_shift_array = np.array([[0,0,0,0],[0,255,255,255],[255,255,255,255],[0,0,0,0]])
        small_rows, small_cols = plane_wave.shape
        large_rows, large_cols = row_num * small_rows, column_num * small_cols
        large_array = np.zeros((large_rows, large_cols), dtype=plane_wave.dtype)
        for i in range(row_num):
            for j in range(column_num):
                large_array[i * small_rows:(i + 1) * small_rows, j * small_cols:(j + 1) * small_cols] = plane_wave
        def phase_shift(row, column, input_array):
            input_array[row * small_rows:(row + 1) * small_rows,
            column * small_cols:(column + 1) * small_cols] = phase_shift_array
            return input_array

        return phase_shift(x_position[0],x_position[1],large_array).astype(np.uint8)[:, :, np.newaxis]
def three_dimension(pattern):
    def inner_loop(two_dimension_pattern):
        return two_dimension_pattern.T[:,:,np.newaxis]
    return list(map(inner_loop, pattern))
##########################This section generates the superpixel patterns
def superpixel(phase_matrix, pixel_size=4):
    """
    Input
    :param pixel_size:
    :return:
    """
    phase_set = np.array(find_unique_numbers_2d(phase_matrix))
    one_super_pixel = np.arange(0,pixel_size**2,1)/pixel_size**2*np.pi


    return
def find_unique_numbers_2d(array_2d):
    unique_numbers = set()

    for row in array_2d:
        unique_numbers.update(row)

    return list(unique_numbers)

def replace_number_with_array(original_matrix, target_number, replacement_array):
    def replace_row(row):
        return [
            replacement_array[k] if cell == target_number else cell
            for k, cell in enumerate(row)
        ]

    result_matrix = list(map(replace_row, original_matrix))
    return result_matrix
##########################
def generate_fourier_mask(u,v,N, phi):
    """
    Generate a Fourier mask for a square image of size N pixels.

    Args:
    N (int): The number of pixels along one dimension of the square image.
    phi (float): The phase term, which should vary between 0 and 2π.

    Returns:
    numpy.ndarray: The generated Fourier mask.
    """
    x, y= np.meshgrid(np.arange(N), np.arange(N))
    mask = np.cos(2 * np.pi * (u * x + v * y) / N + phi)
    return mask
def embed(pattern):
    height,width = pattern[0].shape
    DMD_height = 1140
    DMD_width = 912

    height_start = int(DMD_height/2) - int(height/2)
    height_end = int(DMD_height/2)+int(height/2)
    width_start = int(DMD_width/2)-int(height/2)
    width_end = int(DMD_width/2)+int(height/2)
    new_pattern = []
    def inner_loop(two_dimension_pattern):
        one = (np.ones((DMD_height, DMD_width)) * 128).astype(np.uint8)
        one[height_start:height_end, width_start: width_end] = two_dimension_pattern * 255
        return one.T[:,:, np.newaxis]
    return list(map(inner_loop, pattern))
###############################################################################following code do random pattern
def random_pattern(width, height, sparsity):
    mask_array = (np.random.rand(height, width) < sparsity).astype(int)
    return mask_array.astype(np.uint8)
############################################################################### following code do hadmard mask
def bit_reverse_permutation(num_bits):
    data = np.linspace(0,2**num_bits-1, 2**num_bits)
    n = len(data)
    num_bits = len(bin(n - 1)) - 2
    result = [0] * n
    for i in range(n):
        reversed_index = int(format(i, f'0{num_bits}b')[::-1], 2)
        result[reversed_index] = data[i]

    return result
def generate_gray_code(n):
    if n <= 0:
        return [""]
    smaller_gray_codes = generate_gray_code(n - 1)
    result = []
    for code in smaller_gray_codes:
        result.append("0" + code)
    for code in reversed(smaller_gray_codes):
        result.append("1" + code)

    return result

def gray_code_permutation(num_bits):
    gray_codes = generate_gray_code(num_bits)
    decimal_permutation = [int(code, 2) for code in gray_codes]
    return decimal_permutation

def hadmard_matrix(system_size):
    """
    generate hadmard matrix using scipy library
    :param system_size: The width or height of hadmard matrix.
    :return:
        array: Two dimension array
        array: two dimension array
    """
    if system_size <= 128*128:
        hadamard_matrix = scipy.linalg.hadamard(system_size)
        hadamard_matrix = (hadamard_matrix == 1).astype(int)
        hadamard_matrix = hadamard_matrix.astype(np.uint8)
        return hadamard_matrix
    if system_size > 128*128:
        hadamard_matrix = scipy.linalg.hadamard(128*128)
        hadamard_matrix = (hadamard_matrix == 1).astype(int)
        hadamard_matrix = hadamard_matrix.astype(np.uint8)
        for _ in range(int(np.sqrt(system_size) / np.sqrt(128*128))):
            x_top = np.hstack((hadamard_matrix, hadamard_matrix)).astype(np.uint8)
            x_bottom = np.hstack((hadamard_matrix, (hadamard_matrix*-1).astype(np.uint8)))
            hadamard_matrix = np.vstack((x_top, x_bottom)).astype(np.uint8)

        return  hadamard_matrix

def walsh_to_hadmard_mask(input_matrix, percent_length=1):
    """
    Map the hadmard matrix into walsh matrix
    :param input_matrix: 2D array
    :return: walsh matrix
    """
    small_matrix_size = int(np.sqrt(len(input_matrix[0])))
    num_rows, num_cols = input_matrix.shape
    num_small_matrices = num_rows // small_matrix_size
    small_matrices = []

    reverse_bit_string = bit_reverse_permutation(int(np.log2(num_rows)))
    gray_code_string = generate_gray_code(int(np.log2(num_rows)))
    for i in range(len(gray_code_string)):
        gray_code_string[i] = int(gray_code_string[i], 2)

    def mapping(n):
        n = gray_code_string[int(reverse_bit_string[n])]
        return n
    mapping_list = [mapping(i) for i in range(num_rows)]
    mapping_list = np.array(mapping_list)
    new_list = []
    for i in range(len(mapping_list)):
        new_list.append(np.where(mapping_list==i)[0][0])
    new_list = np.array(new_list)
    for i in range(num_small_matrices):
        for j in range(num_small_matrices):
            start_row = i * small_matrix_size
            end_row = start_row + small_matrix_size
            start_col = j * small_matrix_size
            end_col = start_col + small_matrix_size
            small_matrix = []
            row_number = np.linspace(start_row, end_row - 1, end_row-start_row).astype(int)
            for n in range(len(row_number)):
                small_matrix.append(input_matrix[new_list[row_number[n]], start_col:end_col])
            small_matrix = np.array(small_matrix).astype(np.uint8)
            small_matrices.append(small_matrix)
    length = int(percent_length*len(small_matrix)**2)
    return np.array(small_matrices)[0:length]
def walsh_to_hadmard_mask(input_matrix, percent_length=1):
    """
    Map the hadmard matrix into walsh matrix
    :param input_matrix: 2D array
    :return: walsh matrix
    """
    small_matrix_size = int(np.sqrt(len(input_matrix[0])))
    num_rows, num_cols = input_matrix.shape
    num_small_matrices = num_rows // small_matrix_size
    small_matrices = []

    reverse_bit_string = bit_reverse_permutation(int(np.log2(num_rows)))
    gray_code_string = generate_gray_code(int(np.log2(num_rows)))
    for i in range(len(gray_code_string)):
        gray_code_string[i] = int(gray_code_string[i], 2)

    def mapping(n):
        n = gray_code_string[int(reverse_bit_string[n])]
        return n
    mapping_list = [mapping(i) for i in range(num_rows)]
    mapping_list = np.array(mapping_list)
    new_list = []
    for i in range(len(mapping_list)):
        new_list.append(np.where(mapping_list==i)[0][0])
    new_list = np.array(new_list)
    for i in range(num_small_matrices):
        for j in range(num_small_matrices):
            start_row = i * small_matrix_size
            end_row = start_row + small_matrix_size
            start_col = j * small_matrix_size
            end_col = start_col + small_matrix_size
            small_matrix = []
            row_number = np.linspace(start_row, end_row - 1, end_row-start_row).astype(int)
            for n in range(len(row_number)):
                small_matrix.append(input_matrix[new_list[row_number[n]], start_col:end_col])
            small_matrix = np.array(small_matrix).astype(np.uint8)
            small_matrices.append(small_matrix)
    length = int(percent_length*len(small_matrix)**2)
    return np.array(small_matrices)[0:length]