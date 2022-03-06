from model import *
from AVD_dataloader import PoseDataset
import json
import pandas as pd
import torch
from utills import *
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from collections import Counter
import sys
import copy
from os import walk
import warnings
warnings.filterwarnings("ignore")

num_workers = 0
batch_size = 10
learning_rate = 0.001
num_bins_az = 9
num_bins_el = 5
in_channels = 16
step_size = 3
input_path = "../../label_AVD_Pose/"
mask_size = 128

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print ('device', device)
use_rgb = False



######################################################

MODEL_PATH = "./model_info/best_models/upsample_NoMask.pth"
model_D_mask_reduction = PoseEstimationModelUpsampel_V1_NoMask_MaskedFeatures(in_channels, num_bins_az, mask_size,num_bins_el)

# MODEL_PATH = "./model_info/best_models/Arslan_Paper_VGG.pth"
# model_D_mask_reduction = Arslan_Paper_VGG(in_channels, num_bins_az, mask_size,num_bins_el)
# use_rgb = True

# MODEL_PATH = "./model_info/best_models/ResNet_NoMask.pth"
# model_D_mask_reduction = ResNet_NoMask(in_channels, num_bins_az, mask_size,num_bins_el)


model_D_mask_reduction.load_state_dict(torch.load(MODEL_PATH))
model_D_mask_reduction.eval()
model_D_mask_reduction.to(device)

######################################################

homes = ["Home_001_1/","Home_001_2/","Home_002_1/","Home_003_1/","Home_003_2/","Home_004_1/","Home_004_2/","Home_006_1/","Home_007_1/","Home_008_1/","Home_010_1/","Home_011_1/","Home_013_1/"
,"Home_014_1/","Home_014_2/","Home_015_1/"]

#homes = ["Home_010_1/"]

for home in homes:
    print(home)
    input_path = "/home/negar/Documents/label_AVD_Pose/labels/"+home

    all_files = []
    for (dirpath, dirnames, filenames) in walk(input_path):
        for f in filenames:
            all_files.append(f+dirpath.split("/")[-1])


    eval_dict = eval(open(input_path+"Home_0_label.json",mode='r',encoding='utf-8').read())

    df = pd.DataFrame()
    # azimuth_to_center
    for im_name in eval_dict.keys():
    	data_dict  = {}
    	for obj in eval_dict[im_name]: 
    		if(eval_dict[im_name][obj]["azimuth_to_center"] != 'no_value' and ((im_name+obj) in all_files) and obj != "tabel_symmetry"):
    			az = eval_dict[im_name][obj]["azimuth_to_center"]
    			if(az < 0):
    				az = az+360
    			filename = obj+"/"+ im_name
    			data_dict['filename'] = filename	
    			data_dict['az'] = az
    			data_dict['type'] = obj
    			# if obj == "tabel" :
    			# 	data_dict['model'] = "table/IKEA_BJORKUDDEN_3/model.obj/"
    			# if obj == "small_sofa" :
    			# 	data_dict['model'] = "chair/IKEA_EKTORP_1/model.obj/"	
    			# if obj == "big_sofa" :
    			# 	data_dict['model'] = "sofa/IKEA_EKTORP_3/model.obj/"
    			# print(data_dict)
    			df = df.append(data_dict, ignore_index=True)


    dd = generate_label_AVD(df, num_bins_az)

    eval_im_list = list(dd[0].index)

    labels  = dd[0]["az"]
    overlap_label = dd[1]

    AVD_dataset = PoseDataset(input_path, eval_im_list, labels,overlap_label ,mask_size, dd[0],home)
    eval_dataloader = DataLoader(AVD_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    # MODEL_PATH = "./pahse_two.pth"
    # model = PoseEstimationModelUpsampel_V1_MaskAsChannel(in_channels, num_bins_az, mask_size, num_bins_el, 1)
    # model.load_state_dict(torch.load(MODEL_PATH))
    # # model_D_mask_reduction = nn.DataParallel(model_D_mask_reduction)
    # model.eval()
    # model.to(device)

    accuracy_top = 0
    top_num_train = 2

    total = 0  
    total_el = 0  
    correct = 0 
    correct2 = 0  
    correct_el = 0 
    correct2_el = 0    
    # model.eval()
    all_labels = []
    all_pred = []
    all_cls = []

    correct_2_bins =0


    # model.eval()
    for i, (inputs, labels,y_over,cls,ID,img_rgb) in enumerate(eval_dataloader):
	
        size_batch = len(labels)


        if not use_rgb:
            features = inputs[0].to(device)
        else:
            features = img_rgb.float().to(device)
            
        mask = inputs[1].to(device)

        azimuth= labels.to(device)
        ######################################First Phase#################################################

        y_hat_top = model_D_mask_reduction(features, mask)

        prob_azimuth, predicted = torch.max(y_hat_top[0].data, 1)
        prob_azimuth, predicted2 = torch.topk(y_hat_top[0].data, top_num_train, 1)
        _, predicted2_el = torch.topk(y_hat_top[1].data, 2, 1)
        
        # prev_labels = torch.zeros(top_num_train,len(labels))

        # for i in range(top_num_train):
        #     yhat = model(features,get_Dmask_AVD(predicted2[:,i],predicted2_el[:,0],IDS).to(device),mask.to(device))

        #     y_pred_tag = torch.round(torch.sigmoid(yhat))

        #     prev_labels[i] = y_pred_tag.flatten()
            
        # print(prev_labels)
        # predicted = predicted2[torch.arange(size_batch).reshape(size_batch,1),torch.argmin(prev_labels,0).reshape(size_batch,1)].flatten()

        azimuth_1 = copy.deepcopy(azimuth)
        azimuth_2 = copy.deepcopy(azimuth)

        for ind,i in enumerate(y_over):
                # print(i/2)
                if (i%2 == 1):
                        # print("yes")
                        # print(int(i/2),int(i/2)+1)
                        azimuth_1[ind] = int(i/2)
                        if (int(i/2)+1 == 9 ):
                                azimuth_2[ind] = 0
                        else:
                                azimuth_2[ind] = int(i/2)+1

        predicted = torch.where((predicted == azimuth_1) + (predicted == azimuth_2),azimuth,predicted)

        all_labels.extend(azimuth.cpu().tolist())
        all_pred.extend(predicted.cpu().tolist())
        all_cls.extend(list(cls))

        total += azimuth.size(0)
        correct += torch.sum(predicted == azimuth).item()
        correct2 += torch.sum(torch.eq(predicted2, azimuth.reshape(-1,1))).item()

    accuracy = 100 * correct / total
    accuracy2 = 100 * correct2 / total


    print("####################################### ** AVD EVAL " +home+ " ** #######################################")
    print (f'Val_Accuracy [{accuracy}], Val_Accuracy2 [{accuracy2}]')
    print(classification_report(all_labels, all_pred))

    d2 = (Counter(np.array(all_cls)[np.array(all_labels) ==  np.array(all_pred)]))
    d1 = (Counter(all_cls))
    print(d1) 
    d3 = dict((k, "%.2f" % ( (float(d2[k]) / d1[k])*100 )) for k in d2)
    print(d3)
