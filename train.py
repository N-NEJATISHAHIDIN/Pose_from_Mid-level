from model import *
from data_loader import PoseDataset
import json
import pandas as pd
import torch
from utills import *

num_epochs = 500
batch_size = 20
learning_rate = 0.00001
num_bins = 8
in_channels = 64
input_path = "../../Datasets/pix3d"

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


train_dataset = PoseDataset(input_path, train_dict, labels )
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

test_dataset = PoseDataset(input_path,test_dict, labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

model = PoseEstimationModel(in_channels, num_bins)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
n_total_steps = len(train_loader)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

for epoch in range(num_epochs):
	for i, (inputs, labels) in enumerate(train_dataloader):
		
		print(inputs.shape())
		inputs = inputs.to(device)
		labels = labels.to(device)

		model.train()

		optimizer.zero_grad()
		# compute the model output
		yhat = model(inputs)
		# calculate loss
		loss = criterion(yhat, labels)
		# credit assignment
		loss.backward()
		# update model weights
		optimizer.step()

	if (i+1) % 5 == 0:
	    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
	    
	for i, (inputs, labels) in enumerate(test_dataloader):
		inputs = inputs.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		yhat = model(inputs)
		loss = criterion(yhat, labels.long())
		_, predicted = torch.max(yhat.data, 1)

        	# Total number of labels
		total += labels.size(0)

		if torch.cuda.is_available():
			correct += (predicted.cpu() == labels.cpu()).sum()
		else:
			correct += (predicted == labels).sum()
    
	accuracy = 100 * correct / total
	print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss.item():.4f}, Val_Accuracy [{accuracy}]')
		
	scheduler.step()
	
print('Finished Training')
PATH = './LSTM.pth'
torch.save(model.state_dict(), PATH)



