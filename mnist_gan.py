from __future__ import print_function
import torch
print ("pytorch version {}".format(torch.__version__))
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical


input_dim = 100
num_cls = 10
batch_idx = 64
num_epochs = 20
d_steps = 1
g_steps = 1


(tr_x, tr_y), (te_x, te_y) = mnist.load_data()
print("train data x {} y {}".format(tr_x.shape, tr_y.shape))

class G(nn.Module):
    def __init__(self, input_dim, output_w_h):
        super(G, self).__init__()
        self.input_dim = input_dim
        self.output_w_h = output_w_h
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128 ,256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, self.output_w_h * self.output_w_h)

    def forward(self, x, label):
        x = torch.cat([x, label], dim=-1) # embed label into input in form of concat
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(self.output_w_h, -1)
        return x


class D(nn.Module):
    def __init__(self, input_shape, num_cls, num_feature):
        super(D, self).__init__()
        self.input_shape = input_shape
        self.num_cls = num_cls
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        n_size = self._get_conv_output(self.input_shape)

        self.fc1_1 = nn.Linear(n_size, num_feature) # for input
        self.fc1_2 = nn.Linear(num_cls, num_feature) # for label
        self.fc2 = nn.Linear(num_feature * 2, 1)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.p1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.p2(x)
        return x

    def _get_conv_output(self, input_shape):
        bs = 1
        dummy_in = Variable(torch.rand(bs, *input_shape))
        dummy_conv_out = self._forward_conv(dummy_in)
        n_size = dummy_conv_out.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x, label):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        xf = F.relu(self.fc1_1(x))
        yf = F.relu(self.fc1_2(label))
        x = torch.cat([xf, yf], dim=-1) # embbed label into input
        x = F.sigmoid(self.fc2(x))
        return x

w, h = tr_x.shape[1:]
gen = G(input_dim=input_dim, output_w_h=w)
dis = D(input_shape=(1,w,h), num_cls=num_cls, num_feature=25)
if torch.cuda.is_available():
    gen = gen.cuda()
    dis = dis.cuda()

print(gen, dis)

bce = nn.BCELoss()
g_optim = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)
d_optim = optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1, num_epochs+1)[:1]:
    for _ in range(d_steps)[:1]:
        dis.zero_grad()
        # train D on real
        d_real_x = torch.from_numpy(np.expand_dims(tr_x, axis=1)).float()
        d_real_y = torch.from_numpy(to_categorical(tr_y, num_cls)).float()
        print (type(d_real_x), type(d_real_y))
        d_real_x, d_real_y = Variable(d_real_x), Variable(d_real_y)
        if torch.cuda.is_available():
            d_real_x = d_real_x.cuda()
            d_real_y = d_real_y.cuda()
        print(d_real_x.size(), d_real_y.size())
        res = dis(d_real_x, d_real_y)
        print (type(res), res.data.size(), res.data.type())
        d_real_loss = bce(res, Variable(torch.ones(tr_x.shape[0], 1)))
        # d_real_loss = ce(dis(d_real), d_real_label)
        print ("epoch {}/{} loss {}".format(epoch, num_epochs, d_real_loss.data[0]))
