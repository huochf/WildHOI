import numpy as np
import torch
import torch.nn as nn


class SMPLKPSLoss(nn.Module):

    def __init__(self, wholebody_kps):
        super().__init__()
        self.register_buffer('wholebody_kps', wholebody_kps)


    def project(self, points3d):
        u = points3d[..., 0] / points3d[..., 2]
        v = points3d[..., 1] / points3d[..., 2]
        return torch.stack([u, v], dim=-1)


    def forward(self, hoi_dict):
        wholebody_kps = hoi_dict['wholebody_kps']
        wholebody_kps = self.project(wholebody_kps)

        loss_body_kps2d = ((wholebody_kps[:, :23] - self.wholebody_kps[:, :23, :2]) * 1000) ** 2
        loss_body_kps2d = (loss_body_kps2d * self.wholebody_kps[:, :23, 2:]).mean()

        loss_lhand_kps2d = ((wholebody_kps[:, 23:44] - self.wholebody_kps[:, 91:112, :2]) * 1000) ** 2
        loss_lhand_kps2d = (loss_lhand_kps2d * self.wholebody_kps[:, 91:112, 2:]).mean()

        loss_rhand_kps2d = ((wholebody_kps[:, 44:65] - self.wholebody_kps[:, 112:133, :2]) * 1000) ** 2
        loss_rhand_kps2d = (loss_rhand_kps2d * self.wholebody_kps[:, 112:133, 2:]).mean()

        return {
            'body_kps2d': loss_body_kps2d,
            'lhand_kps2d': loss_lhand_kps2d,
            'rhand_kps2d': loss_rhand_kps2d,
        }


class SMPLDecayLoss(nn.Module):

    def __init__(self, betas_init, body_pose_init, lhand_pose_init, rhand_pose_init):
        super().__init__()
        self.register_buffer('betas_init', betas_init)
        self.register_buffer('body_pose_init', body_pose_init)
        self.register_buffer('lhand_pose_init', lhand_pose_init)
        self.register_buffer('rhand_pose_init', rhand_pose_init)


    def forward(self, hoi_dict):
        betas = hoi_dict['betas']
        body_pose = hoi_dict['body_pose']
        lhand_pose = hoi_dict['lhand_pose']
        rhand_pose = hoi_dict['rhand_pose']

        betas_decay_loss = ((betas - self.betas_init) ** 2).mean()
        body_pose_decay_loss = ((body_pose - self.body_pose_init) ** 2).mean()
        lhand_pose_decay_loss = ((lhand_pose - self.lhand_pose_init) ** 2).mean()
        rhand_pose_decay_loss = ((rhand_pose - self.rhand_pose_init) ** 2).mean()

        body_pose_norm_loss = (body_pose ** 2).mean()
        lhand_pose_norm_loss = (lhand_pose ** 2).mean()
        rhand_pose_norm_loss = (rhand_pose ** 2).mean()

        return {
            'betas_decay': betas_decay_loss,
            'body_decay': body_pose_decay_loss,
            'lhand_decay': lhand_pose_decay_loss,
            'rhand_decay': rhand_pose_decay_loss,
            'body_norm': body_pose_norm_loss,
            'lhand_norm': lhand_pose_norm_loss,
            'rhand_norm': rhand_pose_norm_loss,
        }


class CamDecayLoss(nn.Module):

    def __init__(self, cam_R6d, cam_trans):
        super().__init__()
        self.register_buffer('cam_R6d_init', cam_R6d)
        self.register_buffer('cam_trans_init', cam_trans)


    def forward(self, hoi_dict):
        cam_rot6d = hoi_dict['hoi_rot6d']
        cam_trans = hoi_dict['hoi_trans']

        rot_decay_loss = ((cam_rot6d - self.cam_R6d_init) ** 2).mean()
        trans_decay_loss = ((cam_trans - self.cam_trans_init) ** 2).mean()

        return {
            'cam_R_decay': rot_decay_loss,
            'cam_T_decay': trans_decay_loss,
        }


class ObjectCoorLoss(nn.Module):

    def __init__(self, object_name, coor_x2d, coor_x3d, coor_mask, f=5000):
        super().__init__()
        self.object_name = object_name
        coor_x2d = torch.tensor(coor_x2d).float() # [b, n, 2]
        coor_x3d = torch.tensor(coor_x3d).float() # [b, n, 3]
        coor_mask = torch.tensor(coor_mask).float() # [b, n, 1]
        coor_x3d_sym = coor_x3d.clone()
        if self.object_name == 'skateboard':
            coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
            coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]
        elif self.object_name == 'tennis':
            coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
            coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]
        elif self.object_name == 'baseball':
            coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
            coor_x3d_sym[:, :, 1] = - coor_x3d_sym[:, :, 1]
        elif self.object_name == 'barbell':
            coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
            coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]
        coor_x3d = torch.stack([coor_x3d, coor_x3d_sym], dim=1) # [b, 2, n, 3]

        self.f = f

        self.register_buffer('coor_x2d', coor_x2d)
        self.register_buffer('coor_x3d', coor_x3d)
        self.register_buffer('coor_mask', coor_mask)


    def forward(self, hoi_dict):
        rotmat = hoi_dict['object_rotmat']
        trans = hoi_dict['object_trans']
        object_scale = hoi_dict['object_scale']
        f = self.f
        b = trans.shape[0]

        coor_x3d = self.coor_x3d * object_scale.reshape(b, 1, 1, 1)
        coor_x2d = self.coor_x2d
        coor_mask = self.coor_mask

        coor_x3d = coor_x3d @ rotmat.unsqueeze(1).transpose(-1, -2) + trans.reshape(b, 1, 1, 3)
        u = coor_x3d[:, :, :, 0] / (coor_x3d[:, :, :, 2] + 1e-8) * f
        v = coor_x3d[:, :, :, 1] / (coor_x3d[:, :, :, 2] + 1e-8) * f
        coor_x2d_reproj = torch.stack([u, v], dim=-1)

        loss_coor = ((coor_x2d_reproj - coor_x2d.unsqueeze(1)) ** 2).sum(-1)
        loss_coor, indices = torch.sort(loss_coor, dim=2)

        loss_coor = (loss_coor * coor_mask.reshape(b, 1, -1)).mean(-1)
        loss_coor, _ = loss_coor.min(1)

        return {
            'obj_corr': loss_coor.mean(),
        }


class ObjectScaleLoss(nn.Module):

    def __init__(self, ):
        super().__init__()


    def forward(self, hoi_dict):
        object_scale = hoi_dict['object_scale']
        scale_loss = (object_scale - 1) ** 2
        return {
            'obj_scale': scale_loss.mean(),
        }


class ContactLoss(nn.Module):

    def __init__(self, contact_labels, smplx_part_indices, object_part_indices):
        super().__init__()
        self.contact_labels = contact_labels
        self.smplx_part_indices = smplx_part_indices
        self.object_part_indices = object_part_indices


    def forward(self, hoi_dict):
        smpl_v = hoi_dict['smplx_v_centered']
        object_v = hoi_dict['object_v_centered']

        loss_contact_all = []
        for b_idx in range(smpl_v.shape[0]):
            if len(self.contact_labels[b_idx]) == 0:
                continue

            contact_parts = self.contact_labels[b_idx]
            for parts in contact_parts:
                smpl_contact_v = smpl_v[b_idx, self.smplx_part_indices[str(parts[0])]]
                object_contact_v = object_v[b_idx, self.object_part_indices[str(parts[1])]]
                loss_contact = torch.norm(smpl_contact_v.unsqueeze(1) - object_contact_v.unsqueeze(0), dim=-1) # [n_smpl, n_obj]
                loss_contact = loss_contact.min(-1)[0].mean()
                # if str(parts[0]) in ['7']: # bicycle
                #     loss_contact *= 0.1
                loss_contact_all.append(loss_contact)
        if len(loss_contact_all) == 0:
            loss_contact = torch.zeros(1).float().to(smpl_v.device)
        else:
            loss_contact = torch.stack(loss_contact_all).mean()

        return {
            'contact': loss_contact
        }


from sdf import SDF
class HOCollisionLoss(nn.Module):
# adapted from multiperson (links, multiperson.sdf.sdf_loss.py)

    def __init__(self, smpl_faces, grid_size=32, robustifier=None,):
        super().__init__()
        self.sdf = SDF()
        self.register_buffer('faces', torch.tensor(smpl_faces.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier


    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        # vertices: (n, 3)
        boxes = torch.zeros(2, 3, device=vertices.device)
        boxes[0, :] = vertices.min(dim=0)[0]
        boxes[1, :] = vertices.max(dim=0)[0]
        return boxes


    @torch.no_grad()
    def check_overlap(self, bbox1, bbox2):
        # check x
        if bbox1[0,0] > bbox2[1,0] or bbox2[0,0] > bbox1[1,0]:
            return False
        #check y
        if bbox1[0,1] > bbox2[1,1] or bbox2[0,1] > bbox1[1,1]:
            return False
        #check z
        if bbox1[0,2] > bbox2[1,2] or bbox2[0,2] > bbox1[1,2]:
            return False
        return True


    def forward(self, hoi_dict, ):
        # assume one person and one object
        # person_vertices: (n, 3), object_vertices: (m, 3)
        person_vertices, object_vertices = hoi_dict['smplx_v_centered'], hoi_dict['object_v_centered']
        b = person_vertices.shape[0]
        scale_factor = 0.2
        loss = torch.zeros(1).float().to(object_vertices.device)

        for b_idx in range(b):
            person_bbox = self.get_bounding_boxes(person_vertices[b_idx])
            object_bbox = self.get_bounding_boxes(object_vertices[b_idx])
            if not self.check_overlap(person_bbox, object_bbox):
                return {'collision': loss.sum()}

            person_bbox_center = person_bbox.mean(dim=0).unsqueeze(0)
            person_bbox_scale = (1 + scale_factor) * 0.5 * (person_bbox[1] - person_bbox[0]).max()

            with torch.no_grad():
                person_vertices_centered = person_vertices[b_idx] - person_bbox_center
                person_vertices_centered = person_vertices_centered / person_bbox_scale
                assert(person_vertices_centered.min() >= -1)
                assert(person_vertices_centered.max() <= 1)
                phi = self.sdf(self.faces, person_vertices_centered.unsqueeze(0))
                assert(phi.min() >= 0)

            object_vertices_centered = (object_vertices[b_idx] - person_bbox_center) / person_bbox_scale
            object_vertices_grid = object_vertices_centered.view(1, -1, 1, 1, 3)
            phi_val = nn.functional.grid_sample(phi.unsqueeze(1), object_vertices_grid).view(-1)

            cur_loss = phi_val
            if self.robustifier:
                frac = (cur_loss / self.robustifier) ** 2
                cur_loss = frac / (frac + 1)

            loss += cur_loss.sum()

        return {'collision': loss.sum() / b}
