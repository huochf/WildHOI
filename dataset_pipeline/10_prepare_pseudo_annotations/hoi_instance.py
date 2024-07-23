import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


def load_J_regressor(path):
    data = np.loadtxt(path)

    with open(path, 'r') as f:
        shape = f.readline().split()[1:]
    J_regressor = np.zeros((int(shape[0]), int(shape[1])), dtype=np.float32)
    for i, j, v in data:
        J_regressor[int(i), int(j)] = v
    return J_regressor


class HOIInstance(nn.Module):

    def __init__(self, smpl, object_kps, object_v, smpl_betas=None, smpl_body_pose=None, lhand_pose=None, rhand_pose=None,
        obj_rel_trans=None, obj_rel_rotmat=None, hoi_trans=None, hoi_rot6d=None, object_scale=None):
        super(HOIInstance, self).__init__()

        self.smpl = smpl
        npose = 21 # SMPLX

        self.register_buffer('object_v', object_v) # [bs, n, 3]
        self.register_buffer('object_kps', object_kps) # [bs, n, 3]

        batch_size = self.object_v.shape[0]

        if smpl_betas is not None:
            self.smpl_betas = nn.Parameter(smpl_betas.reshape(batch_size, 10))
        else:
            self.smpl_betas = nn.Parameter(torch.zeros(batch_size, 10, dtype=torch.float32))

        if smpl_body_pose is not None:
            self.smpl_body_pose = nn.Parameter(smpl_body_pose.reshape(batch_size, npose, 3))
        else:
            self.smpl_body_pose = nn.Parameter(torch.zeros(batch_size, npose, 3))

        if lhand_pose is not None:
            self.lhand_pose = nn.Parameter(lhand_pose.reshape(batch_size, 15, 3))
        else:
            self.lhand_pose = nn.Parameter(torch.zeros(batch_size, 15, 3))

        if rhand_pose is not None:
            self.rhand_pose = nn.Parameter(rhand_pose.reshape(batch_size, 15, 3))
        else:
            self.rhand_pose = nn.Parameter(torch.zeros(batch_size, 15, 3))

        if obj_rel_trans is not None:
            self.obj_rel_trans = nn.Parameter(obj_rel_trans.reshape(batch_size, 3))
        else:
            self.obj_rel_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        if obj_rel_rotmat is not None:
            self.obj_rel_rot6d = nn.Parameter(matrix_to_rotation_6d(obj_rel_rotmat.reshape(batch_size, 3, 3)))
        else:
            self.obj_rel_rot6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(batch_size, 1, 1)))

        if hoi_trans is not None:
            self.hoi_trans = nn.Parameter(hoi_trans.reshape(batch_size, 3))
        else:
            self.hoi_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        if hoi_rot6d is not None:
            self.hoi_rot6d = nn.Parameter(hoi_rot6d.reshape(batch_size, 6))
        else:
            self.hoi_rot6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(batch_size, 1, 1)))

        if object_scale is not None:
            self.object_scale = nn.Parameter(object_scale.reshape(batch_size, 1, 1).float())
        else:
            self.object_scale = nn.Parameter(torch.ones(1).reshape(1, 1, 1).repeat(batch_size, 1, 1).float())

        wholebody_regressor = np.load('../data/smpl/smplx_wholebody_regressor.npz')
        self.register_buffer('wholebody_regressor', torch.tensor(wholebody_regressor['wholebody_regressor']).float())


    def forward(self, ):
        b = self.smpl_betas.shape[0]

        global_orient = torch.zeros((b, 3), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)
        jaw_pose = torch.zeros((b, 3), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)
        leye_pose = torch.zeros((b, 3), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)
        reye_pose = torch.zeros((b, 3), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)
        expression = torch.zeros((b, 10), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)

        smplx_out = self.smpl(betas=self.smpl_betas,
                               body_pose=self.smpl_body_pose,
                               left_hand_pose=self.lhand_pose,
                               right_hand_pose=self.rhand_pose,
                               global_orient=global_orient,
                               leye_pose=leye_pose,
                               reye_pose=reye_pose,
                               jaw_pose=jaw_pose,
                               expression=expression)
        smplx_v = smplx_out.vertices
        smplx_J = smplx_out.joints
        smplx_v_centered = smplx_v - smplx_J[:, :1]
        smplx_J_centered = smplx_J - smplx_J[:, :1]

        hoi_rotmat = rotation_6d_to_matrix(self.hoi_rot6d)
        smplx_v = smplx_v_centered @ hoi_rotmat.transpose(2, 1) + self.hoi_trans.reshape(b, 1, 3)
        smplx_J = smplx_J_centered @ hoi_rotmat.transpose(2, 1) + self.hoi_trans.reshape(b, 1, 3)

        scale = self.object_scale.reshape(b, 1, 1)
        object_v_org = self.object_v * scale
        object_kps_org = self.object_kps * scale

        object_rel_rotmat = rotation_6d_to_matrix(self.obj_rel_rot6d)
        object_rotmat = hoi_rotmat.detach() @ object_rel_rotmat
        object_trans = (hoi_rotmat.detach() @ self.obj_rel_trans.reshape(b, 3, 1)).squeeze(-1) + self.hoi_trans.detach()
        object_v = object_v_org @ object_rotmat.transpose(2, 1) + object_trans.reshape(b, 1, 3)
        object_kps = object_kps_org @ object_rotmat.transpose(2, 1) + object_trans.reshape(b, 1, 3)
        object_v_centered = object_v_org @ object_rel_rotmat.transpose(2, 1) + self.obj_rel_trans.reshape(b, 1, 3)
        object_kps_centered = object_kps_org @ object_rel_rotmat.transpose(2, 1) + self.obj_rel_trans.reshape(b, 1, 3)

        wholebody_kps = self.wholebody_regressor.unsqueeze(0) @ smplx_v # [b, 65, 3]

        results = {
            'betas': self.smpl_betas,
            'body_pose': self.smpl_body_pose,
            'lhand_pose': self.lhand_pose,
            'rhand_pose': self.rhand_pose,
            'smplx_v': smplx_v,
            'wholebody_kps': wholebody_kps,
            'smplx_J': smplx_J,
            'smplx_J_centered': smplx_J_centered,
            'smplx_v_centered': smplx_v_centered,

            'hoi_rot6d': self.hoi_rot6d,
            'hoi_rotmat': hoi_rotmat,
            'hoi_trans': self.hoi_trans,

            'object_scale': self.object_scale,
            'object_rel_rotmat': object_rel_rotmat,
            'object_rel_trans': self.obj_rel_trans,
            'object_rotmat': object_rotmat,
            'object_trans': object_trans,
            'object_v': object_v,
            'object_kps': object_kps,
            'object_v_centered': object_v_centered,
            'object_kps_centered': object_kps_centered,
        }
        return results
