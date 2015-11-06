import struct
from array import array
import Image
import math


def generate_images(obj, rows, columns, nameFile):
	img    = Image.new( 'L', (rows,columns)) 
	pixels = img.load() 
	cont = 0
	
	for i in range(columns):    # for every pixel:
	    for j in range(rows):
	        pixels[j,i] = obj[cont]#(int(obj[cont]),int(obj[cont]),100)
	        cont += 1

	img.save('img/GenerationImage_'+nameFile+'.jpg')


