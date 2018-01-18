from __future__ import print_function
import torch
print ("pytorch version {}".format(torch.__version__))
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

epochs = 8000
d_steps = 1
g_steps = 1
bs = 16
in_size = 28*28
test_mu = np.random.random_sample() * 5
test_std = np.random.random_sample()
d_hid = 500
g_hid = 500

def normal_sampler(mu, sigma, bs, n):
    return torch.Tensor(np.random.normal(mu, sigma, (bs, n)))

def status(x):
    return [np.mean(x.data.numpy()), np.std(x.data.numpy())]

class G(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(G, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def data_sampler(self, bs, n):
        return torch.Tensor(np.random.randn(bs, n))

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(D, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


gen = G(in_size, g_hid, in_size)
dis = D(in_size, d_hid, 1)
if torch.cuda.is_available():
    gen = gen.cuda()
    dis = dis.cuda()
print(gen)
print(dis)


bce = nn.BCELoss()
g_optim = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)
d_optim = optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)


for epoch in range(epochs):
    for _ in range(d_steps):
        dis.zero_grad()        
        # train D on real
        d_real = Variable(normal_sampler(test_mu,test_std,bs,in_size))        
        d_real_loss = bce(dis(d_real), Variable(torch.ones(bs,1)))
        d_real_loss.backward()

        # train D on fake
        d_in = Variable(gen.data_sampler(bs,in_size))
        d_fake = gen(d_in)
        d_fake_loss = bce(dis(d_fake), Variable(torch.zeros(bs,1)))
        d_fake_loss.backward()

        d_optim.step()

    for _ in range(g_steps):
        gen.zero_grad()
        g_in = Variable(gen.data_sampler(bs,in_size))
        g_fake = gen(g_in)
        g_fake_loss = bce(dis(g_fake), Variable(torch.ones(bs,1)))
        g_fake_loss.backward()
        g_optim.step()

    print("epoch {}/{} D real {:.5f} fake {:.5f} G {:.5f} (Real {} Fake {})".format(
            epoch, epochs, d_real_loss.data[0], d_fake_loss.data[0],g_fake_loss.data[0],
            status(d_real), status(d_fake)))
