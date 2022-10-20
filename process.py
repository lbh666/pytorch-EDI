import numpy as np
import cv2 as cv
path = 'RBE.npz'
T = 12
x = np.load(path, allow_pickle=True)

event = x['events'].item()

e1 = {}
e2 = {}

e1['t'] = event['t'][(x['exp_start1']<=event['t']) & (event['t']<=x['exp_end1'])]
e1['y'] = event['y'][(x['exp_start1']<=event['t']) & (event['t']<=x['exp_end1'])]
e1['x'] = event['x'][(x['exp_start1']<=event['t']) & (event['t']<=x['exp_end1'])]
e1['p'] = event['p'][(x['exp_start1']<=event['t']) & (event['t']<=x['exp_end1'])]

e2['t'] = event['t'][(x['exp_start2']<=event['t']) & (event['t']<=x['exp_end2'])]
e2['y'] = event['y'][(x['exp_start2']<=event['t']) & (event['t']<=x['exp_end2'])]
e2['x'] = event['x'][(x['exp_start2']<=event['t']) & (event['t']<=x['exp_end2'])]
e2['p'] = event['p'][(x['exp_start2']<=event['t']) & (event['t']<=x['exp_end2'])]

data = {}

data['blur1'] = x['blur1']
data['blur2'] = x['blur2']

# data['e1'] = e1
# data['e2'] = e2

np_ev = [np.zeros((T, 260, 346)), np.zeros((T, 260, 346))]

for i, ev_data in enumerate([e1, e2]):
    t_start = ev_data['t'][0]
    t_end = ev_data['t'][-1]
    ev_data['t'] = (ev_data['t'] - t_start) / (t_end + 1 - t_start) * T
    np_ev[i] = np_ev[i].flatten()
    np.add.at(np_ev[i], (ev_data['x'] + ev_data['y']*346 + np.floor(ev_data['t'])*346*260).astype(np.int32), ev_data['p'])
    np_ev[i] = np_ev[i].reshape((T, 260, 346))
    for j in range(T):
        cv.imshow('1', np_ev[i][j])
        cv.waitKey(0)
        cv.destroyAllWindows()

data['e1'] = np_ev[0]
data['e2'] = np_ev[1]

np.save('processed.npy', data)

