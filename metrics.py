import cv2 as cv
import numpy as np
import math
import os

class MSEandPSNR():
    """
    A class for PSNR and MSE calculation

    ...

    Methods
    -------
    readImage()
        Read the image with small, original, and large size
    mean_squared_error(processed_image, filtered_image)
        Calculate the mean squared error
    peak_signal_to_noise_ratio(mse)
        Calculate the peak signal to noise ratio
    psnr_AverageFilter(greyscale_image_small, greyscale_image_original, greyscale_image_large)
        Calculate the PSNR for average filter image
    psnr_MedianFilter(greyscale_image_small, greyscale_image_original, greyscale_image_large)
        Calculate the PSNR for median filter image
    psnr_GaussianFilter(greyscale_image_small, greyscale_image_original, greyscale_image_large)
        Calculate the PSNR for Gaussian filter image
    """
    def readImage(self):
        """
        Read the image with small, original, and large size

        Returns
        -------
        greyscale_image_small : ndarray
            The value for greyscale small image
        greyscale_image_original : ndarray
            The value for greyscale original image
        greyscale_image_large : ndarray
            The value for greyscale large image
        """
        #read the pre-processed image and the filtered image
        greyscale_image_small = cv.imread('108073_small.tiff')
        greyscale_image_original = cv.imread('108073_original.tiff')
        greyscale_image_large = cv.imread('108073_large.tiff')

        return greyscale_image_small, greyscale_image_original, greyscale_image_large
        
    
    def mean_squared_error(self, processed_image, filtered_image):
        """
        Calculate the MSE (mean squared error)

        Parameters
        ----------
        processed_image : ndarray
            The value of processed image
        filtered_image : ndarray
            The value of filtered image
        """
        #calculate the mean squared error of the images
        A = np.float32(processed_image)
        B = np.float32(filtered_image)
        mse = np.mean((A - B) ** 2)
        #Zero values of MSE is not importance because no noise is present in the image 
        if mse == 0:
            print("\nThere is no noise is present in the image.")
        else:
            print("Mean Squared Error: ", mse)
            #proceed to peak signal-to-noise ratio calculation
            self.peak_signal_to_noise_ratio(mse) 


    def peak_signal_to_noise_ratio(self, mse):
        """
        Calculate the PSNR (peak-to-signal-ratio) value

        Parameter
        ---------
        mse : float
            The value of mean squared error
        """
        d = 256 #Initialize the maximum pixel
        psnr = 20 * math.log10((d-1) / math.sqrt(mse))#calculate PSNR
        print("The value of peak signal-to-noise ratio is : ",psnr, "dB")

    def psnr_AverageFilter(self, greyscale_image_small, greyscale_image_original, greyscale_image_large):
        """
        Calculate the PSNR of average filtered image

        Parameters
        ----------
        greyscale_image_small : ndarray
            The value for greyscale small image
        greyscale_image_original : ndarray
            The value for greyscale original image
        greyscale_image_large : ndarray
            The value for greyscale large image
        """
        #Perform PSNR calculation
        print('==== Peak signal-to-noise ratio ====')
        #PSNR of average filter image with different size and kernel size
        print('\nAverage filter image:')
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    file_name = '108073_small_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    average_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_small, average_filter_image)
                elif i == 1:
                    file_name = '108073_original_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    average_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_original, average_filter_image)
                elif i == 2:
                    file_name = '108073_large_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    average_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_large, average_filter_image)

                kernel_size = kernel_size + 2

    def psnr_MedianFilter(self, greyscale_image_small, greyscale_image_original, greyscale_image_large):
        """
        Calculate the PSNR of median filtered image

        Parameters
        ----------
        greyscale_image_small : ndarray
            The value for greyscale small image
        greyscale_image_original : ndarray
            The value for greyscale original image
        greyscale_image_large : ndarray
            The value for greyscale large image
        """
        #PSNR of median filter image with different size and kernel size
        print('\nMedian filter image:')
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    file_name = '108073_small_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    median_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_small, median_filter_image)
                elif i == 1:
                    file_name = '108073_original_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    median_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_original, median_filter_image)
                elif i == 2:
                    file_name = '108073_large_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    median_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_large, median_filter_image)

                kernel_size = kernel_size + 2

    def psnr_GaussianFilter(self, greyscale_image_small, greyscale_image_original, greyscale_image_large):
        """
        Calculate the PSNR of Gaussian filtered image

        Parameters
        ----------
        greyscale_image_small : ndarray
            The value for greyscale small image
        greyscale_image_original : ndarray
            The value for greyscale original image
        greyscale_image_large : ndarray
            The value for greyscale large image
        """
        #PSNR of gaussian filter image with different size and kernel size
        print('\nGaussian filter image:')
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    file_name = '108073_small_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    gaussian_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_small, gaussian_filter_image)
                elif i == 1:
                    file_name = '108073_original_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    gaussian_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_original, gaussian_filter_image)
                elif i == 2:
                    file_name = '108073_large_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    gaussian_filter_image = cv.imread(file_name)
                    self.mean_squared_error(greyscale_image_large, gaussian_filter_image)

                kernel_size = kernel_size + 2

class CompressionRatio():
    """
    A class for compression ratio calculation

    ...

    Methods
    -------
    compression_ratio(image, file)
        Compute the compressio ratio
    compressionRatio_greyscaleImage(greyscale_image_small, greyscale_image_file_small, greyscale_image_original, greyscale_image_file_original, greyscale_image_large, greyscale_image_file_large)
        Compute the compression ratio for greyscale image
    compressionRatio_AverageFilterImage()
        Compute the compression ratio for average filter image
    compressionRatio_MedianFilterImage()
        Compute the compression ratio for median filter image
    compressionRatio_GaussianFilterImage()
        Compute the compression ratio for Gaussian filter image
    """
    
    def compression_ratio(self, image, file):
        """
        Compute the compression ratio

        Parameters
        ----------
        image : ndarray
            The value of the image
        file : str
            The name of the image file
        """
        # Get the height, width, & number of channels in image data object
        rows, cols, channels = image.shape

        # Compute the uncompressed size
        oi = rows * cols * channels
        print("Uncompressed size:", oi, "bytes")

        # Retrieve the compressed size or the physical size
        infoTIFF = os.stat(file)
        print("Compressed size:", infoTIFF.st_size, "bytes")

        # Compute the compression ratio.
        cr = float(infoTIFF.st_size) / oi
        print("Compression ratio:", cr)
        crPercent = cr*100 # Convert ratio to percentage.
        print("Compression ratio in percentage:", crPercent, "%")

    def compressionRatio_greyscaleImage(self, greyscale_image_small, greyscale_image_file_small, greyscale_image_original, greyscale_image_file_original, greyscale_image_large, greyscale_image_file_large):
        """
        Compute the compression ratio of different greyscale image size

        Parameters
        ----------
        greyscale_image_small : ndarray
        greyscale_image_file_small : str
        greyscale_image_original : ndarray
        greyscale_image_file_original : str
        greyscale_image_large : ndarray
        greyscale_image_file_large : str
        """
        #Compute compression ratio
        print('==== Compression ratio ====')
        #Compression ratio of greyscale image with different size and kernel size
        print('\nGreyscale image')
        for i in range (0, 3):
            if i == 0:
                print('\n'+greyscale_image_file_small)
                self.compression_ratio(greyscale_image_small, greyscale_image_file_small)
            elif i == 1:
                print('\n'+greyscale_image_file_original)
                self.compression_ratio(greyscale_image_original, greyscale_image_file_original)
            elif i == 2:
                print('\n'+greyscale_image_file_large)
                self.compression_ratio(greyscale_image_large, greyscale_image_file_large)

    def compressionRatio_AverageFilterImage(self):
        """
        Compute the compression ratio of average filter image with different size and kernel size
        """        
        print('\nAverage filter image:')
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    file_name = '108073_small_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    average_filter_image = cv.imread(file_name)
                    self.compression_ratio(average_filter_image, file_name)
                elif i == 1:
                    file_name = '108073_original_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    average_filter_image = cv.imread(file_name)
                    self.compression_ratio(average_filter_image, file_name)
                elif i == 2:
                    file_name = '108073_large_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    average_filter_image = cv.imread(file_name)
                    self.compression_ratio(average_filter_image, file_name)

                kernel_size = kernel_size + 2

    def compressionRatio_MedianFilterImage(self):
        """
        Compression ratio of median filter image with different size and kernel size
        """
        print('\nMedian filter image:')
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    file_name = '108073_small_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    median_filter_image = cv.imread(file_name)
                    self.compression_ratio(median_filter_image, file_name)
                elif i == 1:
                    file_name = '108073_original_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    median_filter_image = cv.imread(file_name)
                    self.compression_ratio(median_filter_image, file_name)
                elif i == 2:
                    file_name = '108073_large_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    median_filter_image = cv.imread(file_name)
                    self.compression_ratio(median_filter_image, file_name)

                kernel_size = kernel_size + 2

    def compressionRatio_GaussianFilterImage(self):
        """
        Compression ratio of gaussian filter image with different size and kernel size
        """
        print('\nGaussian filter image:')
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    file_name = '108073_small_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    gaussian_filter_image = cv.imread(file_name)
                    self.compression_ratio(gaussian_filter_image, file_name)
                elif i == 1:
                    file_name = '108073_original_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    gaussian_filter_image = cv.imread(file_name)
                    self.compression_ratio(gaussian_filter_image, file_name)
                elif i == 2:
                    file_name = '108073_large_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff'
                    print('\n'+file_name)
                    gaussian_filter_image = cv.imread(file_name)
                    self.compression_ratio(gaussian_filter_image, file_name)

                kernel_size = kernel_size + 2

if __name__ == '__main__':
    #Initailize the class for mse and psnr
    #MSE is the mean square error
    #PSNR is teh peak to signal ratio
    msePsnr = MSEandPSNR()
    #Read the image of small, original, and large size
    greyscale_image_small, greyscale_image_original, greyscale_image_large = msePsnr.readImage()
    #Compute the PSNR for average filter image
    msePsnr.psnr_AverageFilter(greyscale_image_small, greyscale_image_original, greyscale_image_large)
    #Compute the PSNR for median filter image
    msePsnr.psnr_MedianFilter(greyscale_image_small, greyscale_image_original, greyscale_image_large)
    #Compute the PSNR ratio for Gaussian filter image
    msePsnr.psnr_GaussianFilter(greyscale_image_small, greyscale_image_original, greyscale_image_large)
    
    #Initialize the compression ratio class
    compressionRatio = CompressionRatio()
    #read the greyscale image accordingly
    #The small size
    greyscale_image_small = cv.imread('108073_small.tiff')
    greyscale_image_file_small = '108073_small.tiff'
    #The original size
    greyscale_image_original = cv.imread('108073_original.tiff')
    greyscale_image_file_original = '108073_original.tiff'
    #The large size
    greyscale_image_large = cv.imread('108073_large.tiff')
    greyscale_image_file_large = '108073_large.tiff'
    #Compute the compression ratio for greyscale image
    compressionRatio.compressionRatio_greyscaleImage(greyscale_image_small, greyscale_image_file_small, greyscale_image_original, greyscale_image_file_original, greyscale_image_large, greyscale_image_file_large)
    #Compute the compressio ratio for average filter image
    compressionRatio.compressionRatio_AverageFilterImage()
    #Compute the compressio ratio for median filter image
    compressionRatio.compressionRatio_MedianFilterImage()
    #Compute the compressio ratio for Gaussian filter image
    compressionRatio.compressionRatio_GaussianFilterImage()

