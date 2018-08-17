import csv
import sys
import os

from gdsii.library import Library
import gdspy
from gdsii.elements import *
from shapely.geometry import Polygon

'''
method to extract all the polygons which correspond to a particular label
'''
def get_polygon_string(key, label, polygon_array):
	list_of_polygons = []
	for polygon in polygon_array:
		a = polygon[0]
		b = polygon[1]
		c = polygon[2]
		d = polygon[3]
		temp = 'POLYGON(('+str(a[0])+' '+str(a[1])+','+str(b[0])+' '+str(b[1])+','+str(c[0])+' '+str(c[1])+','+str(d[0])+' '+str(d[1])+','+str(a[0])+' '+str(a[1])+'))'
		list_of_polygons.append(str(key[0])+';'+temp+';'+label)
	return list_of_polygons

gdsii = gdspy.GdsLibrary()
buf = gdsii.read_gds('data/write_buffer.gds')
#all the labels present in the gds file. Note: This labels might be changed if the input gds file is changed
labels = ['and2_1x', 'inv_4x', 'and3_1x', 'inv_1x', 'or2_1x', 'mux4_c_1x', 'a22o2_1x', '12TSRAM', 'endec', 'inv_6x', 'flopenr_c_1x',
		 'fulladder','SRAMbitsNOpg', 'decinvblk', 'and2_2x', 'combo_logic', 'or2_2x',
		'Incr2', 'decoderwb', 'decoder4out', 'fourmux', 'SRAMarray', 'control_logic', 'decincflops', 'writebuffer']

with open('data/gds_polygon_and_label_try.txt', "w") as writer:
	for label in labels:
		label_bounding_box = buf.extract(label).get_bounding_box()
		elements = buf.extract(label).elements
		#Iterates over each element in the gds file and writes to an output file 
		# The layer id, coordinates and the corresponding label for the label
		for element in elements:
			if type(element) is gdspy.CellReference:
				layer_dtype = element.get_polygons(by_spec=True)
				for key in layer_dtype.keys():
					write_to_file_list = get_polygon_string(key, label, layer_dtype[key])
					for list_ in write_to_file_list:
						writer.write(list_+'\n')
			else:
				points_ = element.points
				a = points_[0]
				b = points_[1]
				c = points_[2]
				d = points_[3]
				temp = 'POLYGON(('+str(a[0])+' '+str(a[1])+','+str(b[0])+' '+str(b[1])+','+str(c[0])+' '+str(c[1])+','+str(d[0])+' '+str(d[1])+','+str(a[0])+' '+str(a[1])+'))'
				writer.write(str(element.layer)+';'+temp+';'+label+'\n')
