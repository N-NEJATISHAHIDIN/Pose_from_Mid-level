from model import *
from data_loader import PoseDataset
import json
import pandas as pd
import torch
from utills import *
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from collections import Counter

num_workers = 6
num_epochs = 20
batch_size = 20
learning_rate = 0.001
num_bins_az = 8
num_bins_el = 5
in_channels = 24
step_size = 5
input_path = "../../Datasets/pix3d"
MODEL_PATH = "./model.pth"
mask_size =64

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print ('device', device)

model_D_mask_reduction = PoseEstimationModelUpsampel_V1_MaskedFeatures(in_channels, num_bins_az, mask_size,num_bins_el)
model_D_mask_reduction.load_state_dict(torch.load(MODEL_PATH))
model_D_mask_reduction.eval()
model_D_mask_reduction.to(device)



#features = ['autoencoding','depth_euclidean','jigsaw' ,'reshading','colorization',
#'edge_occlusion','keypoints2d','room_layout',
#'curvature'  ,'keypoints3d'  ,'segment_unsup2d'  ,
#'class_object' ,'egomotion' ,  'nonfixated_pose'   , 'segment_unsup25d',
#'class_scene',  'fixated_pose'  , 'segment_semantic',      
#'denoising' , 'inpainting'   ,'point_matching' ,   'vanishing_point'
#]

#features = ['jigsaw']



#for feature in features:

test_dict = eval(open("pix3d_s1_test.json",mode='r',encoding='utf-8').read())
test_im_list = pd.DataFrame(test_dict["images"])["file_name"]

train_dict = eval(open("pix3d_s1_train.json",mode='r',encoding='utf-8').read())
train_im_list = pd.DataFrame(train_dict["images"])["file_name"]

csv_file = open(input_path + "/Pix3D/Pix3D.txt")
data_f = pd.read_csv(csv_file)
# all infor about all imgs in all categories
dict_pix3d = np.asarray(data_f)
raw_labels = pd.DataFrame(dict_pix3d[:,5:]) 

gt_D_mask_info = raw_labels.set_index([0])
labels  = generate_label(raw_labels, num_bins_az,num_bins_el)

#print("The "+ feature + " has been added.")
train_dataset = PoseDataset(input_path, train_im_list, labels ,mask_size,"jigsaw",gt_D_mask_info)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# train_dataset[0]

test_dataset = PoseDataset(input_path,test_im_list, labels,mask_size,"jigsaw",gt_D_mask_info)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = PoseEstimationModelUpsampel_V1_MaskedFeatures(in_channels, num_bins_az, mask_size,num_bins_el)
# model = PoseEstimationModelUpsampel_V1_MaskAsChannel(in_channels, num_bins, mask_size)
# model = PoseEstimationModel_baseline(in_channels, num_bins)

model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
n_total_steps = len(train_dataloader)
scheduler = StepLR(optimizer, step_size=step_size)

for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels, _,_) in enumerate(train_dataloader):
                
                features = inputs[0].to(device)
                mask = inputs[1].to(device)

                azimuth = labels[0].to(device)
                elevation = labels[1].to(device)
                
                model.train()

                optimizer.zero_grad()
                # compute the model output
                yhat = model(features,mask,1)
                # calculate loss
                train_loss = criterion(yhat[0], azimuth) + criterion(yhat[1], elevation)
                # credit assignment
                train_loss.backward()
                # update model weights
                optimizer.step()

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
        

        for i, (inputs, labels, cls,IDS) in enumerate(test_dataloader):

                features = inputs[0].to(device)
                mask = inputs[1].to(device)

                azimuth = labels[0].to(device)
                elevation = labels[1].to(device)


                # get top 3 of the model without the mask

                y_hat_top = model_D_mask_reduction(features, mask,0)

                _, predicted2 = torch.topk(y_hat_top[0].data, 4, 1)
                _, predicted2_el = torch.topk(y_hat_top[1].data, 2, 1)

                print(predicted2.cpu().numpy().shape, predicted2_el.cpu().numpy().shape)
                
                optimizer.zero_grad()
                
                for az_prob in range(predicted2.cpu().numpy().shape[1]):
                        for el_prob in range(predicted2_el.cpu().numpy().shape[1]):
                                yhat = model(features, get_Dmask(predicted2[:,az_prob],predicted2_el[:,el_prob],IDS,gt_D_mask_info).to(device), 1)
                                
                                print("gt_label: ", labels[0])
                                print("yhat[gt_label]: ", yhat[0].gather( 1,labels[0].to(device).reshape(labels[0].shape[0],1)))

                                print("D_mask_label: ", predicted2[:,az_prob])
                                print("yhat[D_mask_label]: ", yhat[0].gather( 1,predicted2[:,az_prob].to(device).reshape(predicted2[:,az_prob].shape[0],1)))

                                print("predicted_label :", torch.max(yhat[0].data, 1)[1])
                                print("yhat[predicted_label]:", torch.max(yhat[0].data, 1)[0])

                                # print("epoch: ", epoch ,"labels: ",labels[0])
                                # print("preds d-mask: ",predicted2[:,az_prob])
                                # print(yhat[0].get_device(),labels[0].to(device).get_device())
                                # print("yhat yhat: ", yhat[0].gather( 1,labels[0].to(device).reshape(labels[0].shape[0],1)))
                                # print("yhat d-mask: ", yhat[0].gather( 1,predicted2[:,az_prob].to(device).reshape(predicted2[:,az_prob].shape[0],1)))
                                # print("yhat top: ", torch.max(yhat[0].data, 1)[0])

                test_loss = criterion(yhat[0], azimuth) + criterion(yhat[1], elevation)
                
                _, predicted = torch.max(yhat[0].data, 1)
                _, predicted2 = torch.topk(yhat[0].data, 4, 1)
                
                _, predicted_el = torch.max(yhat[1].data, 1)
                _, predicted2_el = torch.topk(yhat[1].data, 2, 1)
                
                print("label",azimuth.cpu().tolist())
                print("preds", predicted.cpu().tolist() )

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
        print (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Val_Accuracy [{accuracy}], Val_Accuracy2 [{accuracy2}], Val_Accuracy [{accuracy_el}], Val_Accuracy2 [{accuracy2_el}]')
        print(classification_report(all_labels, all_pred))
        
        d2 = (Counter(np.array(all_cls)[np.array(all_labels) ==  np.array(all_pred)]))
        d1 = (Counter(all_cls))
        print(d1) 
        d3 = dict((k, "%.2f" % ( (float(d2[k]) / d1[k])*100 )) for k in d2)
        print(d3)

        scheduler.step()
        
print('Finished Training')
PATH = './model.pth'
#torch.save(model.state_dict(), PATH)
