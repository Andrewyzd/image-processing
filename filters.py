import cv2 as cv
import numpy as np

class Preprocessing():
    """
    A class for pre-processing

    Methods
    -------
    grayscale_Transformation(image)
        Convert the colour image to greyscale image
    resizeImage(greyImg)
        Resize the size of the image
    saveInTIFF(resized_image, imageDimension)
        Save the image in TIFF format
    """

    def grayscale_Transformation(self, image):
        """
        Convert the colour image to greyscale image

        Parameter
        ---------
        image : str
            The file name of the image
        """
        img = cv.imread(image)#read image
        greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#convert to grey colour
        self.resizeImage(greyImg)#pass to resize

    def resizeImage(self, greyImg):
        """
        Resize the image

        Parameter
        ---------
        greyImg : ndarray
            The value of greyscale image
        """

        scale_percent_small = 50 #define 50% scale percent of original size
        scale_percent_original = 100#define original scale percent of original size
        scale_percent_large = 200#define 200% scale percent of original size

        original_width = greyImg.shape[1]#extract width of the original image
        original_height = greyImg.shape[0]#extract height of the original image

        #Scale the image to small dimension 
        small_image_width = int(original_width * scale_percent_small / 100)#scale image horizontally
        small_image_height = int(original_height * scale_percent_small / 100)#scale image vertically
        small_dimension = (small_image_width, small_image_height)#set new dimension of size

        #Scale the image originally
        original_image_width = int(original_width * scale_percent_original / 100)#scale image horizontally
        original_image_height = int(original_height * scale_percent_original / 100)#scale image vertically
        original_dimension = (original_image_width, original_image_height)#set new dimension of size

        #Scale the image to large dimension
        large_image_width = int(original_width * scale_percent_large / 100)#scale image horizontally
        large_image_height = int(original_height * scale_percent_large / 100)#scale image vertically
        large_dimension = (large_image_width, large_image_height)#set new dimension of size

        #resize the image
        resized_image_small = cv.resize(greyImg, small_dimension, interpolation = cv.INTER_AREA)#resize the image to small dimension 
        resized_image_original = cv.resize(greyImg, original_dimension, interpolation = cv.INTER_AREA)#resize the image originally
        resized_image_large = cv.resize(greyImg, large_dimension, interpolation = cv.INTER_AREA)#resize the image to large dimension

        #passed for saving
        self.saveInTIFF(resized_image_small, 'small')
        self.saveInTIFF(resized_image_original, 'original')
        self.saveInTIFF(resized_image_large, 'large')

    def saveInTIFF(self, resized_image, imageDimension):
        """
        Save the image in TIFF format

        Parameters
        ----------
        resized_image : ndarray
            The resized image
        imageDimension
            The size labelling of the image
        """
        #save the processed image object to tagged image file format (TIFF)
        #set the name of the image according to the size
        if imageDimension == 'small':
            cv.imwrite('108073_small.tiff', resized_image)
        elif imageDimension == 'original':
            cv.imwrite('108073_original.tiff', resized_image)
        elif imageDimension == 'large':
            cv.imwrite('108073_large.tiff', resized_image)

class LowPassFilter():
    """
    A class for lowpass filter

    Methods
    -------
    readImage()
        Read the images
    avgOperator(img, ksize)
        Calculate the averaging filter
    averagingFilter(image_small, image_original, image_large)
        Perform averaging filter
    medianFilter(image_small, image_original, image_large)
        Perform median filter
    gaussianFilter(image_small, image_original, image_large)
        Perform Gaussian filter
    """
    
    def readImage(self):
        """
        Read the image of different sizes in TIFF format

        Returns
        -------
        image_small : ndarray
            The value of small image
        image_original : ndarray
            The value of original image
        image_large : ndarray
            The value of large image
        """
        #Read the image accordingly
        image_small = cv.imread('108073_small.tiff')
        image_original = cv.imread('108073_original.tiff')
        image_large = cv.imread('108073_large.tiff')

        return image_small, image_original, image_large
        
    #Averaging filter function
    def avgOperator(self, img, ksize):
        """
        Calculate the average filter

        Parameters
        ----------
        img : ndarray
            The value of the image
        ksize : int
            The value of kernel size

        Returns
        -------
        smoothen : ndarray
            The value of average filtered image
        """
        kernel = np.ones((ksize, ksize), np.float32) / (ksize*ksize)
        smoothen = cv.filter2D(img, -1, kernel)
        return smoothen

    def averagingFilter(self, image_small, image_original, image_large):
        """
        Perform average filter

        Parameters
        ----------
        image_small : ndarray
            The value of small greyscale image
        image_original : ndarray
            The value of orginal greyscale image
        image_large : ndarray
            The value of large greyscale image
        """
        #Averaging filter to image with small, original, and large dimensions
        #Apply kernel size of 3x3, 5x5, 7x7
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    average_filter_image = self.avgOperator(image_small, kernel_size)
                    cv.imwrite('108073_small_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', average_filter_image)
                elif i == 1:
                    average_filter_image = self.avgOperator(image_original, kernel_size)
                    cv.imwrite('108073_original_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', average_filter_image)
                elif i == 2:
                    average_filter_image = self.avgOperator(image_large, kernel_size)
                    cv.imwrite('108073_large_average_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', average_filter_image)

                kernel_size = kernel_size + 2

    def medianFilter(self, image_small, image_original, image_large):
        """
        Perform median filter

        Parameters
        ----------
        image_small : ndarray
            The value of small greyscale image
        image_original : ndarray
            The value of orginal greyscale image
        image_large : ndarray
            The value of large greyscale image
        """
        #Median filter to image with small, original, and large dimensions
        #Apply kernel size of 3x3, 5x5, 7x7
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    median_filter_image = cv.medianBlur(image_small, kernel_size)
                    cv.imwrite('108073_small_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', median_filter_image)
                elif i == 1:
                    median_filter_image = cv.medianBlur(image_original, kernel_size)
                    cv.imwrite('108073_original_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', median_filter_image)
                elif i == 2:
                    median_filter_image = cv.medianBlur(image_large, kernel_size)
                    cv.imwrite('108073_large_median_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', median_filter_image)

                kernel_size = kernel_size + 2

    def gaussianFilter(self, image_small, image_original, image_large):
        """
        Perform Gaussian filter

        Parameters
        ----------
        image_small : ndarray
            The value of small greyscale image
        image_original : ndarray
            The value of orginal greyscale image
        image_large : ndarray
            The value of large greyscale image
        """
        #Gaussian filter to image with small, original, and large dimensions
        #Apply kernel size of 3x3, 5x5, 7x7
        for i in range (0, 3):
            kernel_size = 3
            for j in range(0, 3):
                if i == 0:
                    gaussian_filter_image = cv.GaussianBlur(image_small, (kernel_size, kernel_size), 0)
                    cv.imwrite('108073_small_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', gaussian_filter_image)
                elif i == 1:
                    gaussian_filter_image = cv.GaussianBlur(image_original, (kernel_size, kernel_size), 0)
                    cv.imwrite('108073_original_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', gaussian_filter_image)
                elif i == 2:
                    gaussian_filter_image = cv.GaussianBlur(image_large, (kernel_size, kernel_size), 0)
                    cv.imwrite('108073_large_gaussian_filter_'+str(kernel_size)+'x'+str(kernel_size)+'.tiff', gaussian_filter_image)

                kernel_size = kernel_size + 2
    
if __name__ == '__main__':
    #Initialize the pre-processing class
    preprocessing = Preprocessing()
    preprocessing.grayscale_Transformation('108073.jpg')#start pre-process
    print('Done pre-processing')
    #Initialize the lowpass filter class
    low_pass_filter = LowPassFilter()
    #Read the image of different size
    image_small, image_original, image_large = low_pass_filter.readImage()
    #Perform averaging filter to image with different size and kernel size
    low_pass_filter.averagingFilter(image_small, image_original, image_large)
    #Perform median filter to image with different size and kernel size
    low_pass_filter.medianFilter(image_small, image_original, image_large)
    #Perform Gaussian filter to image with different size and kernel size
    low_pass_filter.gaussianFilter(image_small, image_original, image_large)
    print('Done lowpass filter')
