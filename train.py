import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import torch.nn.functional as F

class sobel_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel_x = torch.tensor([[-1., -2., -1.],
                                      [ 0., 0. , 0. ],
                                      [ 1., 2., 1.]
                                      ]).unsqueeze(0).unsqueeze(0).double().cuda()
        self.kernel_y = torch.tensor([[-1., 0., 1.],
                                      [-2., 0., 2.],
                                      [-1., 0., 1.]
                                      ]).unsqueeze(0).unsqueeze(0).double().cuda()
    def forward(self, x, event):
        event = event.abs().sum(dim=1, keepdim=True)
        B,C,H,W = x.shape
        sobel_ev = torch.sqrt((F.conv2d(event, self.kernel_x,padding=1))**2 + (F.conv2d(event, self.kernel_y,padding=1))**2+1e-5)
        sobel_x = torch.sqrt((F.conv2d(x, self.kernel_x,padding=1))**2 + (F.conv2d(x, self.kernel_y,padding=1))**2+1e-5)

        return torch.mul(sobel_ev , sobel_x).mean()

class TV_loss(nn.Module):
    def forward(self, x):
        return ( ( x[:,:,1:,:] - x[:,:,:-1,:] )**2).mean() + (( x[:,:,:,1:] - x[:,:,:,:-1] )**2 ).mean()

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

class EDI(nn.Module):
    def __init__(self, bins=12, c=0., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c = nn.Parameter(torch.tensor(c), requires_grad=True)
        self.bins = bins
    
    def forward(self, img, event, eps=1e-8):
        '''
        input:
            x: [B, C, H, W]
            event: [B, T, H, W]
        '''
        left = -torch.cumsum(event[:, :self.bins//2].flip(dims=[1]), 1) * self.c
        right = torch.cumsum(event[:, self.bins//2:], 1) * self.c
        E = torch.cat([torch.exp(left), torch.exp(right)], dim=1).sum(dim=1, keepdim=True) / self.bins

        return img / (E + eps)

# load data
path = 'processed.npy'
x = np.load(path, allow_pickle=True).item()
event = x['e1']
img = x['blur1']
cv.imwrite('blur1.png', img)
event = torch.from_numpy(event).unsqueeze(0).cuda()
img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()


bins = 12
net = EDI().cuda()
t_loss = total_loss(-5)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
l_loss = []
for i in range(1000):
    net.zero_grad()
    y = net(img, event)
    loss = t_loss(y,event[:, bins//2-1:bins//2+1, ...])
    loss.backward()
    optimizer.step()
    if i %10 == 0:
        l_loss.append(loss.item())
        print(loss.item())
    if i % 100 == 0:
        print(f'current c: {net.c.item()}')
        tmp = y[0,0].detach().cpu().numpy()
        tmp = np.clip(tmp,0.,255.)
        tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        cv.imshow('1', tmp)
        cv.waitKey(0)
        cv.destroyAllWindows()


