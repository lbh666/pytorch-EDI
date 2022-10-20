from random import randrange
import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import torch.nn.functional as F
import matplotlib.pyplot as plt
path = 'processed.npy'
x = np.load(path, allow_pickle=True).item()

event = x['e1']
img = x['blur1']
cv.imwrite('blur1.png', img)
event = torch.from_numpy(event).unsqueeze(0)
img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

class sobel_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel_x = torch.tensor([[-1., -2., -1.],
                                      [ 0., 0. , 0. ],
                                      [ 1., 2., 1.]
                                      ]).unsqueeze(0).unsqueeze(0).double()
        self.kernel_y = torch.tensor([[-1., 0., 1.],
                                      [-2., 0., 2.],
                                      [-1., 0., 1.]
                                      ]).unsqueeze(0).unsqueeze(0).double()
    def forward(self, x, event):
        event = event.sum(dim=1, keepdim=True)
        B,C,H,W = x.shape
        sobel_ev = torch.sqrt((F.conv2d(event, self.kernel_x,padding=1))**2 + (F.conv2d(event, self.kernel_y,padding=1))**2+1e-5)
        sobel_x = torch.sqrt((F.conv2d(x, self.kernel_x,padding=1))**2 + (F.conv2d(x, self.kernel_y,padding=1))**2+1e-5)

        return torch.mul(sobel_ev , sobel_x).sum() / (B*C*H*W)

class TV_loss(nn.Module):
    def forward(self, x):
        B,C,H,W = x.shape
        return (( ( x[:,:,1:,:] - x[:,:,:-1,:] )**2).sum() + (( x[:,:,:,1:] - x[:,:,:,:-1] )**2 ).sum()) / ( B*C*H*W )

class total_loss(nn.Module):
    def __init__(self,weight=-1) -> None:
        super().__init__()
        self.l1 = sobel_loss()
        self.l2 = TV_loss()
        self.weight = weight
    
    def forward(self, x ,event):
        loss = 0
        loss += self.l1(x, event)*self.weight
        loss += self.l2(x)
        
        return loss

class mynet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.c = torch.tensor(0.22)
        
    def forward(self, event, img):
        E = []
        for j in range(12):
            if j <= 5:
                E.append( torch.exp(self.c* ( -event[:,j:6,...].sum(dim=1))))
            else:
                E.append( torch.exp(self.c* ( event[:,6:j+1,...].sum(dim=1))))
        E = torch.stack(E,dim=1).sum(dim=1).unsqueeze(1)/12.
        return img / E

net = mynet()
t_loss = total_loss(-5)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
l_loss = []
for i in range(1000):
    net.zero_grad()
    y = net(event, img)
    loss = t_loss(y,event[:,5:7,...])
    loss.backward()
    optimizer.step()
    if i %10 == 0:
        l_loss.append(loss.item())
        print(loss.item())
    if i % 100 == 0:
        tmp = y[0,0].detach().numpy()
        tmp = np.clip(tmp,0.,255.)
        tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        cv.imshow('1', tmp)
        cv.waitKey(0)
        cv.destroyAllWindows()


