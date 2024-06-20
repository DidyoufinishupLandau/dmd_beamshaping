# dmd_beamshaping
This is the python program to control dmd forming different pattern to modulate the incident beam.
To use the code, strictly follow following steps.
1. Define the resolution of pixel. (Nx, Ny)
   For example, our DMD has resolution 1140, 912. The superpixel size is 4 so that
   Nx = 285
   Ny = 228.
3. Define the size of beam.
   X = np.linspace(-0.002, 0.002, num=Nx)
   
   Y = np.linspace(-0.002, 0.002, num=Ny)
   
   X and Y will be feed into superpixel utility file to compute the transverse size of the beam.
   
5. compute meshgrid
   
   xv, yv = np.meshgrid(X, Y)
   
7. import superpixel utility.
   
   gaussian_hermite_phase = hermite_gaussian(x=xv, y=yv, z=0.01, w0=0.0005, k=2 * np.pi / (633 * 10 ** -9), m=m, n=n)
   
   You will need to define the properties of the beam. Such as propagation distance from beam waist, z. Beam waist size, W0. The wave vector, K. Depending on       
   different types of higher order gaussian beams, call LG, HG, and B mode beam.
6* import generate pattern.
   
   #sample usage
   
   sp = SuperpixelPattern(superpixel_size = 4) #initialize main class, superpixel size  = 4
   
   pw = sp.plane_wave() # compute plane wave
   
   lg = sp.LG_mode(l = 2, p = 1, plot = True)
   
9. Phase reconstruction
   import inverse_fourier_transform
   
   image1 = cv2.imread('LG10.tif', cv2.IMREAD_GRAYSCALE)
   
   image2 = cv2.imread('reference.tif', cv2.IMREAD_GRAYSCALE)
   
   #reconstructed phase diagram

   phase_matrix = callsubstruction(image1,image2)
   
   #plot phase matrix will give you phase diagram.
   
