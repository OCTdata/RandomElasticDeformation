import numpy as np
from scipy import ndimage
import random


class RandomElasticDeformation(object):
    """ Elastic transformation

    Args:
        grid_size (int): size of the square grid
        sigma (float): standard deviation of the displacement
        p (float): probability of the image being transformed. Default value is 1
        has_mask (bool): Specifies whether sample has mask. Default is True.

    """

    def __init__(self, grid_size=3, sigma=10, p=0.5, has_mask=True):
        # grid_size  determines the size of the square grid used to deform the image
        # Sigma determines the strength of the deformation
        # p is the probability of the image being transformed
        # has_mask specifies wheter sample is only a image or an image and a mask
        self.grid_size = grid_size
        self.sigma = sigma
        self.p = p
        self.has_mask = has_mask

    def __call__(self, sample):
        """
         Applies the elastic transformation to the input image and/or mask.

         Args:
             sample (dict): a dictionary containing 'image' and/or 'mask' keys

         Returns:
             dict: a dictionary containing the transformed 'image' and/or 'mask'
         """

        image, mask = np.array(sample['image']), np.array(sample['mask']) if self.has_mask else None

        # Randomly decide whether to apply the transformation
        if random.random() < self.p:
            # Get the shape of the input image
            image_shape = image.shape
            # Generate a coordinate map of the input image
            image_coordinates = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing='ij')
            # Generate a coordinate map of the grid using the image size
            deformation_coordinates = np.meshgrid(np.linspace(0, self.grid_size - 1, image_shape[0]), np.linspace(0, self.grid_size - 1,
                                                                                             image_shape[1]),
                             indexing='ij')
            deformation_grid = [self.grid_size, self.grid_size]

            # Generate the deformation for each dimension (i,j) and combine it with the corresponding coordinates
            for i in range(len(image_shape)):
                # Each cell is assigned with arbitrary values (u ̂_j,v ̂_j) which express the horizontal and vertical
                # magnitude of deformation
                displacement = np.random.randn(*deformation_grid) * self.sigma
                # The displacement vector (u_i,v_i) is determined by the application of a bicubic spline
                # interpolation between the values (u ̂_j,v ̂_j)
                interpolated_displacement = ndimage.map_coordinates(displacement, deformation_coordinates, order=3)
                # Add the displacement to the coordinates of the pixels in the image
                image_coordinates[i] = np.add(image_coordinates[i], interpolated_displacement)

            # Shifting each pixel intensity to its new position using bilinear interpolation
            transformed_image = ndimage.map_coordinates(image, image_coordinates, order=1, mode='reflect')
            # Return the transformed image and/or mask as a dictionary
            if self.has_mask:
                transformed_mask = ndimage.map_coordinates(mask, image_coordinates, order=1, mode='reflect')
                return {'image': transformed_image, 'mask': transformed_mask}
            else:
                return {'image': transformed_image}

        # If p is 0 or the random number is greater than p, return the original image and mask
        return {'image': image, 'mask': mask}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
