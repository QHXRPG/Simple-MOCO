import torch
import torch.nn as nn
import numpy
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch.utils.data as Data
train_data = MNIST(root="./",train=True,transform=transforms.ToTensor(),download=True)
train_loader = Data.DataLoader(dataset=train_data,batch_size=100,shuffle=True,num_workers=0)
test_data = MNIST(root="./",train=False,transform=transforms.ToTensor(),download=True)
test_loader = Data.DataLoader(dataset=test_data,batch_size=10000,shuffle=False,num_workers=0)

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(1,3,3,2,1), #torch.Size([64, 3, 14, 14])
                                nn.ReLU(),
                                nn.Conv2d(3,10,2,2,0) #torch.Size([64, 9, 7, 7])
                                )
    def forward(self,x):
        B , C , H , W = x.shape
        return (self.l1(x)).view(B,-1) #(64,441)
f_k = encoder()
f_q = encoder()

def img_trans1(a):
    return a
def img_trans2(a):
    return a

"""冲量更新函数"""
def moco_updata(k,q):
    m = 0.999
    for i, j in zip(k.parameters(), q.parameters()):
        i.data = i.data * m + j.data * (1. - m)

"""更新队列"""
class Queue:
    def __init__(self):
        self.length = 0
        self.queue = torch.rand(0)
    def updata(self,img):
        k_img = f_k(img)
        K, C = k_img.shape
        self.length = self.length + K
        self.queue = torch.cat([k_img,self.queue])
        if self.length >3000:
            self.queue = self.queue[:3000]
        self.que = self.queue.transpose(1,0)
queue = Queue()


T = 4
opt = torch.optim.Adam(f_q.parameters(),lr=0.005)
CrossEntropyloss = torch.nn.CrossEntropyLoss()

"""训练"""
for i,(img,y) in enumerate(train_loader):
    queue.updata(img)
    img_q = img_trans1(img)
    img_k = img_trans2(img)
    q = f_q(img_q)  #(64,441)
    k = f_k(img_k) #(64,441)
    k = k.detach()
    B,C = q.shape
    l_neg = q @ queue.que #(B,K)
    l_pos = (q.view(B,1,C) @ k.view(B,C,1)).view(B,-1) #(B,1)
    logits = torch.cat([l_pos,l_neg],dim=1)
    labels = torch.zeros(B,dtype=torch.long)
    l = CrossEntropyloss(logits/T,labels)
    opt.zero_grad()
    l.backward(retain_graph=True) #更新f_q
    opt.step()
    moco_updata(f_k,f_q)  #动量更新f_k

for i,(x_t,y_t) in enumerate(test_loader):
    if i>0:
        break

for i,(x,y) in enumerate(train_loader):
    if i>0:
        break


class Linearhead(nn.Module):
    def __init__(self):
        super(Linearhead, self).__init__()
        self.l = nn.Sequential(nn.Linear(490,200),
                               nn.ReLU(),
                               nn.Linear(200,100),
                               nn.ReLU(),
                               nn.Linear(100,10))
    def forward(self,x):
        return self.l(x)
linearhead = Linearhead()
Opt = torch.optim.Adam(linearhead.parameters(),lr=0.001)
Loss = nn.CrossEntropyLoss()
f_q.eval()
for j in range(25):
    for i,(img,y)in enumerate(train_loader):
        q = f_q(img)
        y_p = linearhead(q)
        L = Loss(y_p,y)
        Opt.zero_grad()
        L.backward()
        Opt.step()

def pretect():
    linearhead.eval()
    q = f_q(x_t)
    y_p = linearhead(q)
    c = torch.argmax(y_p,1)
    v = c-y_t
    scount = (10000 - v.count_nonzero())/10000
    return scount

pretect()