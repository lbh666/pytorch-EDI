import numpy as np
from pip import main
import torch
import torch.nn as nn
import cv2 as cv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio

class test_EDI(nn.Module):
    def __init__(self, bins=12, c=0.3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c = nn.Parameter(torch.tensor(c), requires_grad=True)
        self.bins = bins
    
    def forward(self, img, event, idx=None, eps=1e-8):
        '''
        input:
            x: [B, C, H, W]
            event: [B, T, H, W]
        '''
        left = -torch.cumsum(event[:, :idx].flip(dims=[1]), 1) * self.c
        right = torch.cumsum(event[:, idx:], 1) * self.c
        E = torch.cat([torch.exp(left), torch.exp(right)], dim=1).sum(dim=1, keepdim=True) / self.bins

        return img / (E + eps)


if __name__ == "__main__":
    path = 'processed.npy'
    x = np.load(path, allow_pickle=True).item()
    im_list = []
    
    for event, img in [(x['e1'], x['blur1']), (x['e2'], x['blur2'])]:
        event = torch.from_numpy(event).unsqueeze(0)
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        # for reconstruction
        # img = np.zeros_like(img) + 125.
        net = test_EDI()
        
        for i in range(event.shape[1]+1):
            with torch.no_grad():
                y = net(img, event, i)
            tmp = y[0,0].detach().numpy()
            tmp = np.clip(tmp,0.,255.).astype(np.uint8)
            # tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            im_list.append(tmp)
            cv.imshow('1', tmp)
            cv.waitKey(10)
    imageio.mimsave('rec.gif', im_list, 'GIF', duration=0.2)
