from itertools import combinations
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.special import hermite
import numpy as np
import re
########################################################
def parse_complex(complex_str):
    if complex_str == "0":
        return complex(0, 0)
    # Check if the string contains 'i'
    if 'i' in complex_str:
        # Use regular expression to extract real and imaginary parts
        match = re.match(r'([-+]?\d*\.?\d*)([-+]?\d*\.?\d*)i', complex_str)
        if match:
            real_part = float(match.group(1))
            imag_part = float(match.group(2))
            return complex(real_part, imag_part)
        else:
            raise ValueError("Invalid complex number string")
    else:
        # Treat the input as a real number and represent it as complex
        real_part = float(complex_str)
        return complex(real_part, 0)
def read_data(filename):
    # Initialize lists to store data
    first_column = []
    rest_columns = []

    # Read data from file
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            # Extract the first column
            first_column.append(parse_complex(data[0]))

            # Extract the rest of the columns
            rest_columns.append([int(entry) for entry in data[1:]])

        # Convert lists to numpy arrays
    first_column = np.array(first_column)
    rest_columns = np.array(rest_columns)

    return first_column, rest_columns
def get_combination_sums_and_indices(input_list):
    sums_list = []
    indices_list = []

    for r in range(1, len(input_list) + 1):
        for combo_indices in combinations(range(len(input_list)), r):
            combo_sum = sum(input_list[i] for i in combo_indices)
            sums_list.append(combo_sum)
            indices_list.append(list(combo_indices))

    return sums_list, indices_list
def remove_complex_duplicates(arr,combination, precision=10):
    getcontext().prec = precision
    unique_elements = []
    unique_combination = []
    seen_elements = set()

    for i, num in enumerate(arr):
        rounded_real = round(Decimal(num.real), precision)
        rounded_imag = round(Decimal(num.imag), precision)

        if (rounded_real, rounded_imag) not in seen_elements:
            unique_elements.append(num)
            seen_elements.add((rounded_real, rounded_imag))
            unique_combination.append(combination[i])

    return unique_elements,unique_combination
def plot_gaussian_laguerre_cartesian(l, p, X, Y, Z):

    plt.contourf(X, Y, Z, cmap='viridis', levels=100)
    plt.colorbar(label='Intensity')
    plt.title(f'Gaussian Laguerre, l={l}, p={p}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return None
def phase_to_superpixel(phase_matrix,pixel_phase, error=10**-3):
    """
    match the element in phase matrix with elements in the complex plan
    within given error margin.
    :param phase_matrix:
    :param error:
    :return:
    """
    superpixel_matrix = []
    error_map = []
    error_scale = np.linspace(error, 1,100)
    sum_array, combo = get_combination_sums_and_indices(pixel_phase)
    sum_array, combo = remove_complex_duplicates(sum_array, combo)
    num_row = phase_matrix.shape[0]
    num_column = phase_matrix.shape[1]
    sum_array = np.array(sum_array)
    phase_matrix = np.ravel(phase_matrix)
    for i in range(len(phase_matrix)):
        temp = abs(sum_array - phase_matrix[i])
        min = np.min(temp)
        min_indices = np.where(temp == min)[0]
        if min < error:
            superpixel_matrix.append(combo[min_indices[0]])
            error_map.append(error)
        else:
            for i in range(len(error_scale)):
                if min < error_scale[i]:
                    superpixel_matrix.append(combo[min_indices[0]])
                    error_map.append(error)
                    break
    #now map the resize the error map with each element represent as a superpixel
    #meanwhile use the fill the superpixel with masks
    def create_array_with_indices(indices, shape=(4,4)):
        array = np.zeros(shape, dtype=int)
        for index in indices:
            row = int(np.floor(index/shape[0]))
            column = np.mod(index,shape[1])
            array[row][column] = 255
        return array

    temp_superpixel = np.zeros((4, 0))
    count = 0

    two_dimension_superpixel_matrix = np.zeros((0,num_column*4))
    for i in range(len(superpixel_matrix)):

        temp_superpixel = np.hstack((temp_superpixel,create_array_with_indices(superpixel_matrix[i])))
        count+=1
        if count==num_column:
            two_dimension_superpixel_matrix = np.vstack((two_dimension_superpixel_matrix, temp_superpixel))
            count=0
            temp_superpixel = np.zeros((4,0))

    return two_dimension_superpixel_matrix

def phase_to_superpixel(phase_matrix, error=10**-3):
    superpixel_matrix = []
    error_map = []

    sum_array, combo = read_data("look_up_table.txt")

    num_row = phase_matrix.shape[0]
    num_column = phase_matrix.shape[1]
    sum_array = np.array(sum_array)
    phase_matrix = np.ravel(phase_matrix)
    #print(np.min(np.abs(sum_array)))
    for i in range(len(phase_matrix)):
        temp = np.abs(sum_array - phase_matrix[i])
        #print(phase_matrix[i])
        #print(temp[10:20])
        #print(sum_array[10:20])
        min = np.min(temp)
        min_indices = np.where(temp == min)[0]
        #print(min)
        #print(min_indices)
        superpixel_matrix.append(combo[min_indices[0]])
        error_map.append(min)
    #now map the resize the error map with each element represent as a superpixel
    #meanwhile use the fill the superpixel with masks
    def create_array_with_indices(indices):

        return indices.reshape(4,4)*255

    temp_superpixel = np.zeros((4, 0))
    count = 0

    two_dimension_superpixel_matrix = np.zeros((0,num_column*4))
    for i in range(len(superpixel_matrix)):

        temp_superpixel = np.hstack((temp_superpixel,create_array_with_indices(superpixel_matrix[i])))
        count+=1
        if count==num_column:
            two_dimension_superpixel_matrix = np.vstack((two_dimension_superpixel_matrix, temp_superpixel))
            count=0
            temp_superpixel = np.zeros((4,0))

    return two_dimension_superpixel_matrix, np.array(error_map).reshape(num_row,num_column)


def hermite_gaussian(x, y, z, w0, k, m, n):
    """
    Generate the intensity distribution of a Hermite-Gaussian beam.

    Parameters:
        x (numpy.ndarray): x-coordinate.
        y (numpy.ndarray): y-coordinate.
        z (float): Axial distance.
        w0 (float): Beam waist.
        k (float): Wavenumber.
        m (int): Mode index in x-direction.
        n (int): Mode index in y-direction.

    Returns:
        numpy.ndarray: Intensity distribution.
    """
    # Calculate beam radius at distance z
    w_z = w0 * np.sqrt(1 + (z * 2/ (k * w0**2))**2)#double check this
    R = z*(1+1/(z * 2/ (k * w0**2))**2)
    # Normalization factor
    # Hermite polynomials
    H_m = hermite(m)(np.sqrt(2) * x / w_z)
    H_n = hermite(n)(np.sqrt(2) * y / w_z)
    # Gaussian envelope
    gaussian = np.exp(-(x**2 + y**2) / w_z**2)
    gaussian_two = np.exp(-1j*k*(x**2+y**2)/(2*R)+1j*k*z-1j*(n+m+1)*np.arctan2(z, k * w0**2 / 2))
    # Intensity distribution
    E = H_m * H_n * gaussian * gaussian_two

    return E/np.max(np.abs(E))
def plot_intensity(x, y, intensity):
    """
    Plot the intensity distribution.

    Parameters:
        x (numpy.ndarray): x-coordinate values.
        y (numpy.ndarray): y-coordinate values.
        intensity (numpy.ndarray): Intensity values.
    """
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x, y, intensity, shading='auto')
    plt.colorbar(label='Intensity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Intensity Distribution of Hermite-Gaussian Beam')
    plt.grid(True)
    plt.show()
def laguerre_gaussian(l,p,x, y, z = 1, w_0 = 1, lambda_0 = 1):

    z_0 = np.pi*w_0**2/lambda_0
    r = (x**2 + y**2)**0.5
    dis = z/z_0
    pho = r/w_0
    R = (1+dis**2)**0.5
    lagu = genlaguerre(p,abs(l))(2*pho**2/R**2)
    k = 2*np.pi/lambda_0
    return ((pho/R)**abs(l)/R * lagu *
            np.exp(-pho**2/R**2) *
            np.exp(1j*k*z + 1j*dis*pho**2/(R)**2-1j*(l+2*p+1)*np.arctan(dis)-1j*abs(l)*np.arctan(y/x)))
def laguerre_gaussian(x, y, z, w0, k, l, p):
    """
    Generate the intensity distribution of a Hermite-Gaussian beam.

    Parameters:
        x (numpy.ndarray): x-coordinate.
        y (numpy.ndarray): y-coordinate.
        z (float): Axial distance.
        w0 (float): Beam waist.
        k (float): Wavenumber.
        m (int): Mode index in x-direction.
        n (int): Mode index in y-direction.

    Returns:
        numpy.ndarray: Intensity distribution.
    """
    # Calculate beam radius at distance z
    w_z = w0 * np.sqrt(1 + (z * 2/ (k * w0**2))**2)#double check this
    R = z*(1+1/(z * 2/ (k * w0**2))**2)
    r = np.sqrt(x**2 +y **2)
    term_one = w0/w_z*(np.sqrt(2)*r/w_z)**l * np.exp(-r**2/w_z**2)
    lagu = genlaguerre(p, abs(l))(2 * r ** 2 / w_z ** 2)
    term_two = np.exp(-1j*k*r**2/(2*R))*np.exp(-1j*l*np.arctan2(y,x))*np.exp(1j*(l+2*p+1)*np.arctan2(z, (k * w0**2)/2))
    E = term_one*lagu*term_two
    return E/np.max(np.abs(E))


def bessel_beam(x, y, z, w0, k, m):
    """
    Generate a Bessel beam intensity profile using SciPy's special functions.

    Parameters:
        x, y (ndarray): Input coordinates.
        z (float): Propagation distance.
        w0 (float): Waist size.
        k (float): Wave number.
        m (int): Order of the Bessel function (default is 1).

    Returns:
        ndarray: Intensity profile of the Bessel beam.
    """
    from scipy.special import jv
    r = (x**2+y**2)**0.5
    E = np.exp(1j*k*z)*jv(m, k*r)*np.exp(1j*m*np.arctan2(y,x))
    return E / np.max(np.abs(E))
def plot_phase_diagram(complex_matrix):
    phase = np.angle(complex_matrix)
    plt.imshow(phase, cmap='hsv')
    plt.colorbar()
    plt.title('Phase Diagram')
    plt.show()
def asymetrical_bessel_beam(n,x,y,c):
    r = (x**2+y**2)**0.5
    w = 0.001
    phi = np.arctan(y/x)
    part_one = np.exp(-r**2/w**2)
    alpha = 0.08 *1/(13.68*10**-6)
    part_two = (alpha*r/(alpha*r+2*c*np.exp(1j*phi)))**(n/2)
    from scipy.special import jn
    Jn = jn(n, np.sqrt(alpha*r*(alpha*r+2*c*np.exp(1j*phi))))
    return np.exp(1j*n*phi) * Jn * part_one * part_two
def main():
    from simulation_DMD import plot_pixel
    Nx = 285
    Ny = 228
    X = np.linspace(-0.0025, 0.0025, num=Nx)
    Y = np.linspace(-0.0025, 0.0025, num=Ny)
    xv, yv = np.meshgrid(X, Y)
    matrix = hermite_gaussian(x=xv, y=yv, z=0.001, w0=0.001, k=2*np.pi/(633*10**-9), m=0, n=0)
    #matrix = laguerre_gaussian(x=xv, y=yv, z=0.0000000000000000000001, w0=0.001, k=2*np.pi/(633*10**-9), l=0, p=0)
    #print(np.min(matrix))
    a,b = phase_to_superpixel(matrix)
    plot_pixel(-1*a+255)
    print(a)
    plot_phase_diagram(matrix)
    plot_gaussian_laguerre_cartesian(1,0,X,Y,np.abs(matrix)**2)
    plot_pixel(b)
    print(np.sum(b)/(b.shape[0]*b.shape[1]))
#main()
def test():
    from simulation_DMD import plot_pixel
    a = np.array([[-0.01903+0.0078825j, 0.1j],[0.9j, 0.9j]])
    a,b = phase_to_superpixel(a)
    print(a)
    print(b)
    plot_pixel(a)
    plot_pixel(b)
def test_bessel():
    from simulation_DMD import plot_pixel
    Nx = 285
    Ny = 228
    X = np.linspace(-0.000001, 0.000001, num=Nx)
    Y = np.linspace(-0.000001, 0.000001, num=Ny)
    xv, yv = np.meshgrid(X, Y)
    bessel_beam_matrix = bessel_beam(x=xv, y=yv, z=0.01, w0=0.001, k=2*np.pi/(633*10**-9), m=2)
    print(bessel_beam_matrix)
    matrix, _ = phase_to_superpixel(bessel_beam_matrix)
    intensity = np.abs(bessel_beam_matrix)**2
    plot_intensity(xv, yv, intensity)
    plot_phase_diagram(bessel_beam_matrix)
    print(bessel_beam_matrix)
    plot_pixel(matrix)
#test_bessel()
#test()
#main()