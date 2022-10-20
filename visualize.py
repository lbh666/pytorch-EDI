import numpy as np
from pip import main
import torch
import torch.nn as nn
import cv2 as cv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio

class testnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.tensor(0.3383), requires_grad=True)
        
    def forward(self, event, img,idx = 6):
        E = []
        
        for j in range(event.shape[1]+1):
            E.append( torch.exp(self.c* ( np.sign(j-idx) * event[:,min(j,idx):max(j,idx),...].sum(dim=1))))
        E = torch.stack(E,dim=1).sum(dim=1).unsqueeze(1)/(event.shape[1]+1)
        
        return img / E


if __name__ == "__main__":
    path = 'processed.npy'
    x = np.load(path, allow_pickle=True).item()
    im_list = []
    
    for event, img in [(x['e1'], x['blur1']), (x['e2'], x['blur2'])]:
        event = torch.from_numpy(event).unsqueeze(0)
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        net = testnet()
        
        for i in range(event.shape[1]+1):
            with torch.no_grad():
                y = net(event, img, i)
            tmp = y[0,0].detach().numpy()
            tmp = np.clip(tmp,0.,255.)
            tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            im_list.append(tmp)
            cv.imshow('1', tmp)
            cv.waitKey(10)
    imageio.mimsave('cube.gif', im_list, 'GIF', duration=0.1)
