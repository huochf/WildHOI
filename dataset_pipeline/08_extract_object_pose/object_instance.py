import os
import trimesh
import numpy as np
import torch
import torch.nn as nn

from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


class ObjectInstance(nn.Module):

    def __init__(self, object_name, R6d_init, trans_init):
        super().__init__()

        if object_name == 'cello':
            object_mesh = trimesh.load('../data/objects/{}_body.ply'.format(object_name), process=False)
        else:
            object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
        object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32)
        object_v = object_v - object_v.mean(0).reshape(1, -1)
        object_f = torch.tensor(np.array(object_mesh.faces, dtype=np.int64))

        self.register_buffer('object_v', object_v) # [n, 3]
        self.register_buffer('object_f', object_f) # [n, 3]

        self.R6d = nn.Parameter(R6d_init) # [b, 6]
        self.trans = nn.Parameter(trans_init) # [b, 3]


    def get_optimizer(self, lr=1e-3):
        optimizer = torch.optim.Adam([self.R6d, self.trans], lr=lr, betas=(0.9, 0.999))
        return optimizer


    def forward(self, batch_idx, batch_size):
        rotmat = rotation_6d_to_matrix(self.R6d[batch_idx:batch_idx+batch_size])
        trans = self.trans[batch_idx:batch_idx+batch_size]
        object_v = self.object_v.unsqueeze(0) @ rotmat.transpose(2, 1) + trans.reshape(-1, 1, 3)
        return {
            'rotmat': rotmat,
            'R6d': self.R6d[batch_idx:batch_idx+batch_size],
            'trans': trans,
            'object_v': object_v,
        }
