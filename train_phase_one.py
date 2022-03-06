from model import *
from data_loader import PoseDataset
import json
import pandas as pd
import torch
from utills import *
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from collections import Counter
from torch import nn
import sys
import copy


# get the model name from command line arg
model_input_name = sys.argv[1]


model_info = get_model_config(model_input_name)

#training info
num_workers = model_info["num_workers"]
num_epochs =  model_info["num_epochs"]
batch_size =  model_info["batch_size"]
learning_rate =  model_info["learning_rate"]
step_size =  model_info["step_size"]

#model_info
model_name = model_info["model_name"]
in_channels =  model_info["in_channels"]
num_bins_az =  model_info["num_bins_az"]
mask_size =  model_info["mask_size"]
num_bins_el =  model_info["num_bins_el"]
flag = model_info["flag"]

#data_loder_info
input_path =  model_info["input_path"]
features = model_info["added_feature"]
gt_D_mask_info_flag = model_info["gt_D_mask_info"]
accuracy_top = 0

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print ('device', device)



test_dict = eval(open("pix3d_s1_test.json",mode='r',encoding='utf-8').read())
test_im_list = pd.DataFrame(test_dict["images"])["file_name"]

train_dict = eval(open("pix3d_s1_train.json",mode='r',encoding='utf-8').read())
train_im_list = pd.DataFrame(train_dict["images"])["file_name"]
# print(len(train_im_list))
# train_im_list = train_im_list[:5654]
# print(len(train_im_list))


csv_file = open(input_path + "/Pix3D/Pix3D.txt")
data_f = pd.read_csv(csv_file)

# all info about all imgs in all categories
dict_pix3d = np.asarray(data_f)
raw_labels = pd.DataFrame(dict_pix3d[:,5:]) 

#D_mask_infor for each image(path to the right cad models)
gt_D_mask_info = raw_labels.set_index([0])
no_D_mask = None
#labels of all images

dd = generate_label(raw_labels, num_bins_az,num_bins_el)
labels  = dd[0]
overlap_label = dd[1]

train_dataset = PoseDataset(input_path, train_im_list, labels,overlap_label ,mask_size, features, eval(gt_D_mask_info_flag))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# feature, gt_D_mask_info
# train_dataset[0]

test_dataset = PoseDataset(input_path, test_im_list, labels,overlap_label, mask_size, features, eval(gt_D_mask_info_flag))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


model = eval(model_name+"(in_channels, num_bins_az, mask_size, num_bins_el, flag)")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
n_total_steps = len(train_dataloader)
scheduler = StepLR(optimizer, step_size=step_size)


for epoch in range(num_epochs):
        model.train()

        all_labels = []
        all_pred = []
        for i, (inputs, labels, cls,IDS,y_over,img_rgb, mask_real,_) in enumerate(train_dataloader):
                
                if model_name not in ["Arslan_Paper_VGG","ResNet_NoMask"]:
                    features = inputs[0].to(device)
                else:
                    features = img_rgb.float().to(device)

                mask = inputs[1].to(device)

                azimuth = labels[0].to(device)
                elevation = labels[1].to(device)
                
                model.train()

                optimizer.zero_grad()
                # compute the model output
                yhat = model(features,mask)
                _, predicted = torch.max(yhat[0].data, 1)

                all_labels.extend(azimuth.cpu().tolist())
                all_pred.extend(predicted.cpu().tolist())

                # calculate loss
                train_loss = criterion(yhat[0], azimuth) + criterion(yhat[1], elevation)
                # credit assignment
                train_loss.backward()
                # update model weights
                optimizer.step()

        print(classification_report(all_labels, all_pred))
        total = 0  
        total_el = 0  
        correct = 0 
        correct2 = 0  
        correct_el = 0 
        correct2_el = 0    
        model.eval()
        all_labels = []
        all_pred = []
        all_cls = []
        test_eq =0
        match = 0

        for i, (inputs, labels, cls,IDS,y_over,img_rgb, mask_real,_) in enumerate(test_dataloader):

                if model_name not in ["Arslan_Paper_VGG","ResNet_NoMask"] :
                    features = inputs[0].to(device)
                else:
                    features = img_rgb.float().to(device)

                mask = inputs[1].to(device)

                # Overlap labels 
                # azimuth = y_over.to(device)
                azimuth = labels[0].to(device)
                elevation = labels[1].to(device)

                # get top 3 of the model without the mask

                yhat = model(features, mask)

                
                optimizer.zero_grad()
                test_loss = criterion(yhat[0], azimuth) + criterion(yhat[1], elevation)
                
                _, predicted = torch.max(yhat[0].data, 1)

                azimuth_1 = copy.deepcopy(azimuth)
                azimuth_2 = copy.deepcopy(azimuth)
                for ind,i in enumerate(y_over):
                        if (i%2 == 1):
                                azimuth_1[ind] = int(i/2)
                                if (int(i/2)+1 == 9 ):
                                        azimuth_2[ind] = 0
                                else:
                                        azimuth_2[ind] = int(i/2)+1
                
                # correct2 += torch.sum((predicted == azimuth_1) + (predicted == azimuth_2)).item()

                predicted = torch.where((predicted == azimuth_1) + (predicted == azimuth_2),azimuth,predicted)
                
                _, predicted2 = torch.topk(yhat[0].data, 2, 1)
                
                _, predicted_el = torch.max(yhat[1].data, 1)
                _, predicted2_el = torch.topk(yhat[1].data, 2, 1)
                

                all_labels.extend(azimuth.cpu().tolist())
                all_pred.extend(predicted.cpu().tolist())
                all_cls.extend(list(cls))

                # Total number of labels
                total += azimuth.size(0)
                total_el += elevation.size(0)

                correct += torch.sum(predicted == azimuth).item()
                correct2 += torch.sum(torch.eq(predicted2, azimuth.reshape(-1,1))).item()

                correct_el += torch.sum(predicted_el == elevation).item()
                correct2_el += torch.sum(torch.eq(predicted2_el, elevation.reshape(-1,1))).item()
    
        accuracy = 100 * correct / total
        accuracy2 = 100 * correct2 / total

        accuracy_el = 100 * correct_el / total_el
        accuracy2_el = 100 * correct2_el / total_el
        
        print("############################################################################################")
        print (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Val_Accuracy_Overlap [{accuracy}], Val_Accuracy_top2_nonOverlap [{accuracy2}], Val_Accuracy_elevation [{accuracy_el}], Val_Accuracy_elevation_top2 [{accuracy2_el}]')
        # print (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val_Accuracy [{accuracy}], Val_Accuracy2 [{accuracy2}]')
        # print(test_eq/total*25)
        print(classification_report(all_labels, all_pred))
        
        d2 = (Counter(np.array(all_cls)[np.array(all_labels) ==  np.array(all_pred)]))
        d1 = (Counter(all_cls))
        print(d1) 
        d3 = dict((k, "%.2f" % ( (float(d2[k]) / d1[k])*100 )) for k in d2)
        print(d3)

        if (accuracy_top < accuracy) :
                accuracy_top = accuracy
                #save top model
                PATH = 'model_info/best_models/{}.pth'.format(model_input_name)
                
                if not os.path.exists('model_info/best_models/'):                  
                  # Create a new directory because it does not exist 
                  os.makedirs('model_info/best_models/')
                  os.makedirs('model_info/accuracy_info/')
                torch.save(model.state_dict(), PATH)

                #save top model accuracy info
                Top_model_accuracy_info = open("model_info/accuracy_info/"+model_input_name + ".txt","w")
                Top_model_accuracy_info.write("####################################### ** "+ model_input_name +" ** #######################################")
                Top_model_accuracy_info.write("\n")
                Top_model_accuracy_info.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Val_Accuracy_Overlap [{accuracy}], Val_Accuracy_top2_nonOverlap [{accuracy2}], Val_Accuracy_elevation [{accuracy_el}], Val_Accuracy_elevation_top2 [{accuracy2_el}]')
                Top_model_accuracy_info.write("\n")
                Top_model_accuracy_info.write(classification_report(all_labels, all_pred))
                Top_model_accuracy_info.write("\n")
                Top_model_accuracy_info.write(str(d1))
                Top_model_accuracy_info.write("\n")
                Top_model_accuracy_info.write(str(d3))
                Top_model_accuracy_info.close()

        scheduler.step()
        
print('Finished Training')











