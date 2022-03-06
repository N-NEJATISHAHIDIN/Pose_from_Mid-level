import sys
import copy
import json
import numpy  as np 
import pandas as pd    

df = pd.DataFrame()
test_dict = eval(open("Home_0_label.json",mode='r',encoding='utf-8').read())

for im_name in test_dict.keys():
	data_dict  = {}

	for obj in test_dict[im_name]: 
		print(obj)
		if(test_dict[im_name][obj]["azimuth"] != 'no_value'):
			az = test_dict[im_name][obj]["azimuth"]
			if(az < 0):
				az = az+360
			filename = obj+"/"+ im_name
			data_dict['filename'] = filename	
			data_dict['az'] = az
			data_dict['type'] = obj
			if obj == "tabel" :
				data_dict['model'] = "IKEA_BJORKUDDEN_3"
			if obj == "small_sofa" :
				data_dict['model'] = "IKEA_EKTORP_1"	
			if obj == "big_sofa" :
				data_dict['model'] = "IKEA_EKTORP_3"
			# print(data_dict)
			df = df.append(data_dict, ignore_index=True)
