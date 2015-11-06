import numpy
import Image
from scipy import misc
from generate_file import generate_images

#const
SizeMnist = 28

def normalize_digit(x, digit_normalized_width, end_size):
	"""
	normalize digit image
	"""
    # build the matrix image
	x          = x.reshape(SizeMnist, SizeMnist)
    # non-zero col-sums
	width_diff = digit_normalized_width - sum(sum(x)!=0) 
	
	if width_diff:
            #dimension corresponding to new width
            new_dim          = SizeMnist + width_diff
            #generate new image to resize
            new_image        = misc.toimage(x)
            #resize image according to the new dim + end size
            normalized_image = new_image.resize((new_dim,end_size))
            normalized_array = numpy.array(normalized_image.getdata(), dtype=numpy.float32)
            # new x with new dimensions Ex: (26,28) to W18
            x   = normalized_array.reshape(end_size,new_dim) / 255

	return pad_image(x, end_size)

def pad_image(x, end_size):
    
    """
    padding to complete the image Size
    corresponding to SizeMnist ** 2 (28,28)
    """

    #get the length of padding
    #  difference between total image size 
    #  and the new normalized image with less width
    padding   = end_size - x.shape[1]

    #quantity for passing on left
    left_side = round(padding / 2)
    
    #quantity for passing on right
    right_side = padding - left_side

    #total padding
    pads = (left_side,right_side)

    # padding
    aa        = numpy.pad(x,((0.,0.),pads),mode='constant',constant_values=0)
    aareshape = aa.reshape(end_size**2)
    
    #generate images for validation
    generate_images(aareshape*255,28,28,'W'+str(20 - (padding)))
    return aareshape
        
    