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
batch_size = 64
num_epochs = 20
d_steps = 1
g_steps = 1


(tr_x, tr_y), (te_x, te_y) = mnist.load_data()
print("train data x {} y {}".format(tr_x.shape, tr_y.shape))

class G(nn.Module):
    def __init__(self, input_dim, num_cls, output_w_h):
        super(G, self).__init__()
        self.input_dim = input_dim
        self.output_w_h = output_w_h
        self.num_cls = num_cls
        self.fc1 = nn.Linear(self.input_dim + self.num_cls, 128)
        self.fc2 = nn.Linear(128 ,256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, self.output_w_h * self.output_w_h)

    def forward(self, x, label):
        x = torch.cat([x, label], dim=-1) # embed label into input in form of concat
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(x.data.size()[0], 1, self.output_w_h, self.output_w_h)
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
gen = G(input_dim=input_dim, num_cls=num_cls, output_w_h=w)
dis = D(input_shape=(1,w,h), num_cls=num_cls, num_feature=25)

# if torch.cuda.is_available():
#     gen = gen.cuda()
#     dis = dis.cuda()

print(gen, dis)

bce = nn.BCELoss()
g_optim = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)
d_optim = optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1, num_epochs+1):
    total_samples = len(tr_x)
    total_idx = np.random.permutation(np.arange(total_samples))
    i = 0
    while i < total_samples:
        j = min(i + batch_size, total_samples)
        current_tr_x = tr_x[total_idx[i:j]]
        current_tr_y = tr_y[total_idx[i:j]]
        i = j
        d_real_loss, d_fake_loss, g_fake_loss = 0,0,0
        for _ in range(d_steps):
            dis.zero_grad()

            # train D on real
            current_d_real_x = torch.from_numpy(np.expand_dims(current_tr_x, axis=1)).float()
            current_d_real_y = torch.from_numpy(to_categorical(current_tr_y, num_cls)).float()
            current_d_real_x, current_d_real_y = Variable(current_d_real_x), Variable(current_d_real_y)
            current_d_real_target = Variable(torch.ones(len(current_tr_x), 1))
            current_d_real_pred = dis(current_d_real_x, current_d_real_y)
            d_real_loss = bce(current_d_real_pred, current_d_real_target)
            d_real_loss.backward()

            # train D on fake
            current_tr_fake_x = np.random.normal(0, 1, (len(current_tr_x), input_dim))
            current_tr_fake_y = np.random.randint(0, num_cls, (len(current_tr_x), 1))
            current_d_fake_x = torch.from_numpy(current_tr_fake_x).float()
            current_d_fake_y = torch.from_numpy(to_categorical(current_tr_fake_y, num_cls)).float()
            current_d_fake_x, current_d_fake_y = Variable(current_d_fake_x), Variable(current_d_fake_y)
            current_d_gen_out = gen(current_d_fake_x, current_d_fake_y)
            current_d_fake_target = Variable(torch.zeros(len(current_tr_fake_x), 1))
            current_d_fake_pred = dis(current_d_gen_out, current_d_fake_y)
            d_fake_loss = bce(current_d_fake_pred, current_d_fake_target)
            d_fake_loss.backward()

            d_optim.step()

        for _ in range(g_steps):
            # train G on fake
            current_tr_fake_x = np.random.normal(0, 1, (len(current_tr_x), input_dim))
            current_tr_fake_y = np.random.randint(0, num_cls, (len(current_tr_x), 1))
            current_g_fake_x = torch.from_numpy(current_tr_fake_x).float()
            current_g_fake_y = torch.from_numpy(to_categorical(current_tr_fake_y, num_cls)).float()
            current_g_fake_x, current_g_fake_y = Variable(current_g_fake_x), Variable(current_g_fake_y)
            current_g_gen_out = gen(current_g_fake_x, current_g_fake_y)
            current_g_fake_target = Variable(torch.ones(len(current_tr_fake_x), 1))
            current_g_fake_pred = dis(current_g_gen_out, current_g_fake_y)
            g_fake_loss = bce(current_g_fake_pred, current_g_fake_target)
            g_fake_loss.backward()

            g_optim.step()

        print ("epoch {} {}/{} D real {} fake {} G {}".format(epoch, i/batch_size, total_samples/batch_size,
                                                    d_real_loss.data[0], d_fake_loss.data[0], g_fake_loss.data[0]))


