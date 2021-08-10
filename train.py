from model import *
from data_loader import PoseDataset
import json
import pandas as pd
import torch
from utills import *
from torch.optim.lr_scheduler import StepLR

num_workers = 6
num_epochs = 20
batch_size = 20
learning_rate = 0.001
num_bins = 8
in_channels = 16
step_size = 2
input_path = "../../Datasets/pix3d"
mask_size =64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('device', device)


test_dict = eval(open("pix3d_s1_test.json",mode='r',encoding='utf-8').read())
test_im_list = pd.DataFrame(test_dict["images"])["file_name"]

train_dict = eval(open("pix3d_s1_train.json",mode='r',encoding='utf-8').read())
train_im_list = pd.DataFrame(train_dict["images"])["file_name"]

csv_file = open(input_path + "/Pix3D/Pix3D.txt")
data_f = pd.read_csv(csv_file)
#all infor about all imgs in all categories
dict_pix3d = np.asarray(data_f)
raw_labels = pd.DataFrame(dict_pix3d[:,5:])

labels  = generate_label(raw_labels, num_bins)


train_dataset = PoseDataset(input_path, train_im_list, labels ,mask_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
train_dataset[0]

test_dataset = PoseDataset(input_path,test_im_list, labels,mask_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = PoseEstimationModelUpsampel_V1_MaskAsChannel(in_channels, num_bins, mask_size)
# model = PoseEstimationModel_baseline(in_channels, num_bins)

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
n_total_steps = len(train_dataloader)
scheduler = StepLR(optimizer, step_size=step_size)

for epoch in range(num_epochs):
	model.train()
	for i, (inputs, labels) in enumerate(train_dataloader):
		
		features = inputs[0].to(device)
		mask = inputs[1].to(device)

		labels = labels.to(device)
		model.train()

		optimizer.zero_grad()
		# compute the model output
		yhat = model(features,mask,1)
		# calculate loss
		train_loss = criterion(yhat, labels)
		# credit assignment
		train_loss.backward()
		# update model weights
		optimizer.step()

	total = 0  
	correct = 0 
	correct2 = 0 	
	model.eval() 
	for i, (inputs, labels) in enumerate(test_dataloader):
		features = inputs[0].to(device)
		mask = inputs[1].to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		yhat = model(features, mask, 1)
		#print("yhat : ",yhat)
		#print("labels : ", labels)
		test_loss = criterion(yhat, labels.long())
		_, predicted = torch.max(yhat.data, 1)
		_, predicted2 = torch.topk(yhat.data, 2, 1)
		
        # Total number of labels
		total += labels.size(0)
		

		correct += torch.sum(predicted == labels).item()
		correct2 += torch.sum(torch.eq(predicted2, labels.reshape(-1,1))).item()
    
	accuracy = 100 * correct / total
	accuracy2 = 100 * correct2 / total

	print (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Val_Accuracy [{accuracy}], Val_Accuracy2 [{accuracy2}]')
		
	scheduler.step()
	
print('Finished Training')
PATH = './LSTM.pth'
torch.save(model.state_dict(), PATH)