import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

def shift_center(Fh2, shift_amount):
    num_row, num_col = Fh2.shape
    frequency_x_range = np.max(Fh2) - np.min(Fh2)
    # Calculate the shift amount modulo the image size
    shift_amount_x = shift_amount


    # Shift the center
    shifted_Fh2 = np.roll(Fh2, (shift_amount_x, 0), axis=(1, 0))

    return shifted_Fh2
def plot_wavefront(phase_angle_two):
    # Calculate the gradient of the phase to identify phase jumps
    gradient_x, gradient_y = np.gradient(phase_angle_two)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    # Apply thresholding to identify significant phase jumps
    threshold = np.percentile(gradient, 99)
    mask = gradient > threshold

    # Unwrap the phase angles while ignoring significant phase jumps
    phase_angle_two_continuous = np.unwrap(phase_angle_two, axis=0, discont=1, mask=mask)
    phase_angle_two_continuous = np.unwrap(phase_angle_two_continuous, axis=1, discont=1, mask=mask)

    plt.imshow(phase_angle_two_continuous, cmap='hsv')
    plt.title('Continuous Wavefront')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar(label='Phase (radians)')
    plt.show()
def plot_phase_diagram_3d(phase_angle_two, elev=40, azim=60):
    # Generate coordinates for the phase angle matrix
    x = np.arange(phase_angle_two.shape[1])
    y = np.arange(phase_angle_two.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, phase_angle_two, cmap='hsv')
    ax.set_title('Phase Diagram (3D)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Phase (radians)')
    fig.colorbar(surf, label='Phase (radians)')

    # Set custom view angle
    ax.view_init(elev=elev, azim=azim)

    plt.show()
def callsubstruction(image, image_2):
    frequencies = np.fft.fft2(image)
    frequencies_shifted = np.fft.fftshift(frequencies)  # Shift the zero frequency component to the center

    frequencies_two = np.fft.fft2(image_2)
    frequencies_shifted_two = np.fft.fftshift(frequencies_two)  # Shift the zero frequency component to the center

    # Display original image and frequency domain image
    plt.subplot(121), plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(122), plt.imshow(np.log(np.abs(frequencies_shifted)))
    plt.title('Frequency Domain Image')
    plt.show()
    plt.imshow(np.log(np.abs(frequencies_shifted)))
    plt.title('Frequency Domain Image')
    plt.show()
    plt.subplot(121), plt.imshow(image_2)
    plt.title('Original Image')
    plt.plot(122), plt.imshow(np.log(np.abs(frequencies_shifted_two)))
    plt.title('Frequency Domain Image')
    plt.show()
    # Prompt user to input coordinates
    print("Please input the coordinates of the top-left and bottom-right corners of the window.")
    top_left_x = 370
    top_left_y = 340
    bottom_right_x = 400
    bottom_right_y = 360

    top_left_x = 345
    top_left_y = 0
    bottom_right_x = 430
    bottom_right_y = 800

    top_left_x = 255+220
    top_left_y = 300+170
    bottom_right_x = 400+260
    bottom_right_y = 370+240

    def angle(frequencies_shifted):
        # Perform inverse Fourier transform only on the selected window
        selected_window = np.zeros_like(frequencies_shifted)
        selected_window[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1

        frequencies_selected = frequencies_shifted * selected_window
        reconstructed_image = shift_center(frequencies_selected, int(frequencies_selected.shape[1]/2-(bottom_right_x+top_left_x)/2))
        plt.imshow(np.log(np.abs(reconstructed_image)))
        plt.title('Frequency Domain Image')
        plt.show()
        reconstructed_image = np.fft.ifft2(np.fft.ifftshift(reconstructed_image))
        reconstructed_image = reconstructed_image / np.max(np.abs(reconstructed_image))

        phase_angle = np.angle(reconstructed_image)
        # Adjust phase values to be within -π to π
        #phase_angle = np.mod(phase_angle + np.pi, 2*np.pi) - np.pi
        #reconstructed_image[reconstructed_image==0] = np.average(reconstructed_image)
        plt.imshow(np.log(np.abs((frequencies_selected))))
        plt.show()
        return phase_angle

    def adjust_phase_matrix(phase_matrix):
        # Convert the phase matrix to a numpy array
        phase_matrix = np.array(phase_matrix)

        # Iterate through each value in the phase matrix
        while (phase_matrix > np.pi).any() or (phase_matrix < -np.pi).any():
            # Subtract 2π from values greater than π
            phase_matrix[phase_matrix > np.pi] -= 2 * np.pi
            # Add 2π to values less than -π
            phase_matrix[phase_matrix < -np.pi] += 2 * np.pi

        return phase_matrix

    def gaussian_smooth(matrix, sigma):
        smoothed_matrix = median_filter(matrix,size=sigma)
        return smoothed_matrix
    phase_angle = angle(frequencies_shifted)
    phase_angle_two = angle(frequencies_shifted_two)
    unwrap = np.unwrap(phase_angle_two, discont=np.pi/4, axis=1, period=6.283185307179586)
    unwrap = np.unwrap(unwrap, discont=np.pi/4, axis=0, period=6.283185307179586)
        # Plot reconstructed image in the phase diagram
    #plot_phase_diagram_3d(unwrap)
    plt.imshow(unwrap, cmap='hsv')
    plt.title('Reconstructed Image (Phase Diagram)')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Phase (radians)')
    plt.show()

    plt.imshow(phase_angle, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('Reconstructed Image (Phase Diagram)')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Phase (radians)')
    plt.show()
    substruction = gaussian_smooth(phase_angle-phase_angle_two, 1)
    substruction = adjust_phase_matrix(substruction)
    plt.imshow(substruction, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('Reconstructed Image (Phase Diagram)')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Phase (radians)')
    plt.show()
    #plot_wavefront(phase_angle_two)

    # Plot reconstructed image in the phase diagram
    #plot_phase_diagram_3d(substruction)
    #plot_phase_diagram_3d(substruction)
    return substruction