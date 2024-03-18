import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
func = lambda x: 2 * np.cos(0.5 * x) + 1.5 * np.sin(3 * x)
x = np.linspace(-5, 5, 900)
x_train, x_test = x[::2], x[1::2]
y_train, y_test = func(x_train), func(x_test)

#x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)


def convert_to_tensor_data(np_inputs):
    output = [torch.unsqueeze(torch.tensor([item]), dim = 0).to(dtype = torch.float) 
              for item in np_inputs]
    return torch.cat(output)
  
#.to(dtype = torch.float) 

x_train, x_test = convert_to_tensor_data(x_train), convert_to_tensor_data(x_test)
#y_train, y_test = convert_to_tensor_data(x_train), convert_to_tensor_data(y_test)
x_train, y_train = x_train.to(device = device, dtype = torch.float), y_train.to(device = device, dtype = torch.float)
#print(x_train)

func_max, func_min = max(y_train), min(y_train)


def gpu_info():
    print(f'Cuda availability: {torch.cuda.is_available()}')
    print(f'Number of devices: {torch.cuda.device_count()}')
    current_device = torch.cuda.current_device()
    print(f'Current device: {current_device}')
    print(f'Current device name: {torch.cuda.get_device_name(current_device)}')
    print(f'Device: {device}')



class SimpleNet(nn.Module):
    '''
    Simple fully connected neural net for one variable function approximation! 
    '''
    
    def __init__(self):
        super().__init__()  #SimpleNet, self
        self.fc1 = nn.Linear(1, 50)    #1, 15
        self.fc2 = nn.Linear(50, 25)   #15, 10
        self.fc3 = nn.Linear(25, 1)    #10, 1
        #self.fc4 = nn.Linear(10, 1)
        #self.fc3 = nn.Linear(5, 1)
        self.act_func1 = nn.Tanh()
        self.act_func2 = nn.Sigmoid()
        self.act_func3 = nn.PReLU(init = 0.5)
        

    def forward(self, x):
        x = self.act_func3(self.fc1(x))
        x = self.act_func1(self.fc2(x))
        x = self.act_func3(self.fc3(x))
        #x = self.act_func1(self.fc4(x))
        #x = self.act_func1(self.fc3(x))
        return x        



net = SimpleNet()
loss_func = nn.MSELoss()

lr, momentum = 0.007, 0.9
#opt = torch.optim.SGD(net.parameters(), lr = lr, momentum = momentum, weight_decay = 0.7)
#def train(model, lr):
opt = torch.optim.Adam(net.parameters(), lr = lr, betas = (0.9, 0.99), weight_decay = 0.00003)
#opt = torch.optim.SGD(net.parameters(), lr = lr, momentum = momentum, weight_decay = 0.005)
EPOCH_NUM = 500
test_loss_history = []



def train(model, loss_func, opt, epoch_num):
    model.to(device = device)
    epochs = []
    for epoch in range(epoch_num):
        opt.zero_grad()
        y_pred = model(x_train)
        loss = loss_func(y_pred, torch.unsqueeze(y_train, dim = 1))
        print(f'Epoch: {epoch}/{epoch_num},  Loss: {loss}')
        test_loss_history.append(loss.item())
        loss.backward()
        opt.step()
        epochs.append(epoch)

    #print(f'Validation loss: {loss_func(y_test.to(device = device), model(x_test.to(device = device)))}')
    print(f'Learning rate: {lr}, Momentum: {momentum}')
    print(f'Optimizer: {type(opt).__name__}')
    plt.plot(epochs, test_loss_history)
    plt.show()
    


if __name__ == "__main__":
    print(f'Max func value: {func_max}, min func value: {func_min}')
    gpu_info()
    train(net, loss_func, opt, EPOCH_NUM)
    last_loss_value = test_loss_history[-1]
    if last_loss_value <= 0.0075:
        torch.save(net, 'model.pth')



#plt.plot(x_train.cpu(), func(x_train.cpu()))
#plt.plot(x_train.cpu(), net(x_train.cpu()))

    #torch.no_grad()
#y_net_predicted = net(x_test.to(device=device))
#print(y_net_predicted)
#print(y_test)
#plt.plot(x_test.numpy(), y_test.numpy(), x_test.numpy(), net(x_test))
#plt.show()


#x1 = x_train.cpu().numpy() 
#x2 = x_test.cpu().numpy()  #torch.squeeze(tensor)
#net.eval()
#y1 = y_test.cpu().numpy()
#y2 = torch.squeeze(net(x_test.to(device = device)).cpu()).numpy()  #.cpu().numpy()
#x2 = torch.squeeze(x_test).cpu().numpy()
#plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), x_test.cpu().numpy(), net(x_test.to(device = device)))
#plt.show()
#print(x2)
#print(y1, y2)


