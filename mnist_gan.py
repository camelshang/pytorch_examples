from __future__ import print_function
import os
os.environ['DISPLAY'] = "10.92.6.138:19.0"
import torch
print ("pytorch version {}".format(torch.__version__))
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
plt.ioff()

torch.cuda.set_device(1)

input_dim = 100
num_cls = 10
batch_size = 100
num_epochs = 50
d_steps = 1
g_steps = 1
lr = 0.002 # important parameters


(tr_x, tr_y), (te_x, te_y) = mnist.load_data()
tr_x = tr_x / 255.0 # norm tp 0-1

fixed_z = np.random.rand(10,100)
fixed_y = np.arange(num_cls)
print("train data x {}(min {} max {}) y {}".format(tr_x.shape, np.amin(tr_x), np.amax(tr_x), tr_y.shape))
print ("fixed gen data x {} y {}({})".format(fixed_z.shape, fixed_y.shape, fixed_y))


def generate_images(epoch, z, y, save_name='mnist'):
    gen.eval()
    fixed_z = Variable(torch.from_numpy(z)).float()
    fixed_y = Variable(torch.from_numpy(to_categorical(y, num_cls))).float()
    if torch.cuda.is_available():
        fixed_z = fixed_z.cuda()
        fixed_y = fixed_y.cuda()
    gen_images = gen(fixed_z, fixed_y).data.cpu().numpy() # NCHW
    gen_images = np.transpose(gen_images, (0,2,3,1))
    fig = plt.figure()
    row = int(np.ceil(np.sqrt(len(z))))
    for i in range(len(z)):
        a = fig.add_subplot(row, row, i+1)
        a.imshow(gen_images[i].squeeze(), cmap='gray')
        a.set_title(str(y[i]))
        a.axis('off')
    fig.tight_layout()
    fig.savefig("{}_{}.png".format(save_name, epoch))

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        # torch.nn.init.xavier_normal(m.weight.data)
        # torch.nn.init.constant(m.bias.data, 0.1)
        torch.nn.init.normal(m.weight.data, 0, 0.02)
        torch.nn.init.normal(m.bias.data, 0, 0.02)
    elif isinstance(m, nn.Linear):
        # torch.nn.init.xavier_normal(m.weight.data)
        # torch.nn.init.constant(m.bias.data, 0.1)
        torch.nn.init.normal(m.weight.data, 0, 0.02)
        torch.nn.init.normal(m.bias.data, 0, 0.02)
        # m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()

class G(nn.Module):
    def __init__(self, input_dim, num_cls, output_w_h):
        super(G, self).__init__()
        self.input_dim = input_dim
        self.output_w_h = output_w_h
        self.num_cls = num_cls
        self.fc1_1 = nn.Linear(self.input_dim, 128)
        self.fc1_2 = nn.Linear(self.num_cls, 128)
        self.fc2 = nn.Linear(256 ,512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.output_w_h * self.output_w_h)

    def forward(self, x, label):
        x = F.relu(self.fc1_1(x))
        y = F.relu(self.fc1_2(label))
        x = torch.cat([x,y], dim=-1) # embed label into input in form of concat
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(x.data.size()[0], 1, self.output_w_h, self.output_w_h)
        return x


class D(nn.Module):
    def __init__(self, input_shape, num_cls):
        super(D, self).__init__()
        self.input_shape = input_shape
        self.num_cls = num_cls
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        n_size = self._get_conv_output(self.input_shape)
        print ("n_size", n_size)

        self.fc1_1 = nn.Linear(n_size, 512) # for input
        self.fc1_2 = nn.Linear(num_cls, 512) # for label
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256,1)

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
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

w, h = tr_x.shape[1:]
gen = G(input_dim=input_dim, num_cls=num_cls, output_w_h=w)
gen.apply(weight_init)
dis = D(input_shape=(1,w,h), num_cls=num_cls)
dis.apply(weight_init)

if torch.cuda.is_available():
    gen = gen.cuda()
    dis = dis.cuda()

print(gen, dis)

bce = nn.BCELoss()
g_optim = optim.SGD(gen.parameters(), lr=lr, momentum=0.9)
# g_optim = optim.Adam(gen.parameters(), lr=lr)
d_optim = optim.SGD(dis.parameters(), lr=lr, momentum=0.9)
# d_optim = optim.Adam(dis.parameters(), lr=lr)

for epoch in range(1, num_epochs+1):
    total_samples = len(tr_x)
    total_idx = np.random.permutation(np.arange(total_samples))
    i = 0
    while i < total_samples:
        j = min(i + batch_size, total_samples)
        current_tr_x = tr_x[total_idx[i:j]]
        current_tr_y = tr_y[total_idx[i:j]]
        real_target = Variable(torch.ones(len(current_tr_x)))
        fake_target = Variable(torch.zeros(len(current_tr_x)))
        if torch.cuda.is_available():
            real_target = real_target.cuda()
            fake_target = fake_target.cuda()

        i = j
        d_real_loss, d_fake_loss, g_fake_loss = 0,0,0
        d_real_acc, d_fake_acc, g_fake_acc = 0,0,0
        current_d_real_pred, current_d_fake_pred, current_g_fake_pred = 0,0,0
        for _ in range(d_steps):
            dis.zero_grad()

            # train D on real
            current_d_real_x = torch.from_numpy(np.expand_dims(current_tr_x, axis=1)).float()
            current_d_real_y = torch.from_numpy(to_categorical(current_tr_y, num_cls)).float()
            current_d_real_x, current_d_real_y = Variable(current_d_real_x), Variable(current_d_real_y)
            if torch.cuda.is_available():
                current_d_real_x = current_d_real_x.cuda()
                current_d_real_y = current_d_real_y.cuda()

            current_d_real_pred = dis(current_d_real_x, current_d_real_y).squeeze()
            # print ("D real pred", current_d_real_pred.data.cpu().numpy())
            d_real_loss = bce(current_d_real_pred, real_target)
            d_real_acc = np.mean(current_d_real_pred.data.cpu().numpy() >= 0.5)
            d_real_loss.backward()

            # train D on fake
            current_tr_fake_x = np.random.rand(len(current_tr_x), input_dim)
            current_tr_fake_y = np.random.randint(0, num_cls, (len(current_tr_x), 1))
            current_d_fake_x = torch.from_numpy(current_tr_fake_x).float()
            current_d_fake_y = torch.from_numpy(to_categorical(current_tr_fake_y, num_cls)).float()
            current_d_fake_x, current_d_fake_y = Variable(current_d_fake_x), Variable(current_d_fake_y)
            if torch.cuda.is_available():
                current_d_fake_x = current_d_fake_x.cuda()
                current_d_fake_y = current_d_fake_y.cuda()

            current_d_gen_out = gen(current_d_fake_x, current_d_fake_y)
            current_d_fake_pred = dis(current_d_gen_out, current_d_fake_y).squeeze()
            # print ("D fake pred", current_d_fake_pred.data.cpu().numpy())
            d_fake_loss = bce(current_d_fake_pred, fake_target)
            d_fake_acc = np.mean(current_d_fake_pred.data.cpu().numpy() < 0.5)

            d_fake_loss.backward()
            # d_loss = d_real_loss + d_fake_loss
            # d_loss.backward()

            d_optim.step()

        for _ in range(g_steps):
            gen.zero_grad()
            # train G on fake
            current_tr_fake_x = np.random.rand(len(current_tr_x), input_dim)
            current_tr_fake_y = np.random.randint(0, num_cls, (len(current_tr_x), 1))
            current_g_fake_x = torch.from_numpy(current_tr_fake_x).float()
            current_g_fake_y = torch.from_numpy(to_categorical(current_tr_fake_y, num_cls)).float()
            current_g_fake_x, current_g_fake_y = Variable(current_g_fake_x), Variable(current_g_fake_y)
            if torch.cuda.is_available():
                current_g_fake_x = current_g_fake_x.cuda()
                current_g_fake_y = current_g_fake_y.cuda()

            current_g_gen_out = gen(current_g_fake_x, current_g_fake_y)
            current_g_fake_pred = dis(current_g_gen_out, current_g_fake_y).squeeze()
            # print ("G fake pred", current_g_fake_pred.data.cpu().numpy())
            g_fake_loss = bce(current_g_fake_pred, real_target)
            g_fake_acc = np.mean(current_g_fake_pred.data.cpu().numpy() >= 0.5)

            g_fake_loss.backward()
            g_optim.step()

        if i % 200 == 0:
            print ("epoch {} {}/{} LOSS D real {:.5f} fake {:.5f} G {:.5f} ACC D real {:.3f} fake {:.3f} G {:.3f}".format(
                epoch, i/batch_size,total_samples/batch_size,
                d_real_loss.data[0], d_fake_loss.data[0], g_fake_loss.data[0],
                d_real_acc, d_fake_acc, g_fake_acc))
    generate_images(epoch, fixed_z, fixed_y)
            # print("D real pred", current_d_real_pred.data.cpu().numpy())
            # print("D fake pred", current_d_fake_pred.data.cpu().numpy())
            # print("G fake pred", current_g_fake_pred.data.cpu().numpy())


