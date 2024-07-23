import numpy as np
import torch
import torch.nn as nn

import neural_renderer as nr


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

    def __init__(self, object_name, coor_x2d, coor_x3d, coor_mask):
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

        self.register_buffer('coor_x2d', coor_x2d)
        self.register_buffer('coor_x3d', coor_x3d)
        self.register_buffer('coor_mask', coor_mask)


    def forward(self, hoi_dict):
        rotmat = hoi_dict['object_rotmat']
        trans = hoi_dict['object_trans']
        object_scale = hoi_dict['object_scale']
        f = 5000
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


class MultiViewHOIKPSLoss(nn.Module):

    def __init__(self, model, visual_feats, n_views):
        super().__init__()
        self.model = model
        self.n_views = n_views
        batch_size = visual_feats.shape[0]
        visual_feats = visual_feats.unsqueeze(1).repeat(1, n_views, 1).reshape(batch_size * n_views, -1)
        hoi_kps_samples, _ = model.sampling(batch_size * n_views, visual_feats, z_std=1.)

        cam_pos = hoi_kps_samples.reshape(batch_size, n_views, -1)[..., -3:].detach()
        self.cam_pos = nn.Parameter(cam_pos) # [batch_size, n_views, 3]
        self.visual_feats = visual_feats


    def forward(self, hoi_dict):

        smpl_kps = hoi_dict['smplx_J_centered'][:, :22]
        object_kps = hoi_dict['object_kps_centered']
        hoi_kps = torch.cat([smpl_kps, object_kps], dim=1)
        b, n_kps, _ = hoi_kps.shape
        n_views = self.n_views
        hoi_kps = hoi_kps.reshape(b, 1, n_kps, 3).repeat(1, n_views, 1, 1)

        cam_pos = self.cam_pos.reshape(b, n_views, 1, 3)
        cam_directions = hoi_kps - cam_pos
        cam_directions = cam_directions / (torch.norm(cam_directions, dim=-1, keepdim=True) + 1e-8)
        x = torch.cat([cam_directions, cam_pos], dim=-2).reshape(b * n_views, -1)

        log_prob = self.model.log_prob(x, self.visual_feats)
        loss_kps_nll = - log_prob.mean()

        return {
            'kps_nll': loss_kps_nll,
        }


class ContactLoss(nn.Module):

    def __init__(self, smpl_occlusion_maps, object_occlusion_maps, smpl_mean_occlusion_map, object_mean_occlusion_map, threshold):
        super().__init__()

        self.smpl_occlusion_maps = smpl_occlusion_maps * smpl_mean_occlusion_map.unsqueeze(0)
        self.object_occlusion_maps = object_occlusion_maps * object_mean_occlusion_map.unsqueeze(0)
        self.threshold = threshold


    def forward(self, hoi_dict):
        smpl_v = hoi_dict['smplx_v']
        object_v = hoi_dict['object_v']

        batch_size = smpl_v.shape[0]
        loss_contact = torch.zeros(1).to(smpl_v.device)
        count = 0
        for b_idx in range(batch_size):
            smpl_occlusion_map = self.smpl_occlusion_maps[b_idx]
            smpl_v_contact = smpl_v[b_idx][smpl_occlusion_map > self.threshold]
            smpl_v_weights = smpl_occlusion_map[smpl_occlusion_map > self.threshold]

            object_occlusion_map = self.object_occlusion_maps[b_idx]
            object_v_contact = object_v[b_idx][object_occlusion_map > self.threshold]
            object_v_weights = object_occlusion_map[object_occlusion_map > self.threshold]

            if smpl_v_contact.shape[0] == 0 or object_v_contact.shape[0] == 0:
                continue

            ho_distance = torch.sqrt(((smpl_v_contact.unsqueeze(1) - object_v_contact.unsqueeze(0)) ** 2).sum(-1))
            weights = smpl_v_weights.unsqueeze(1) * object_v_weights.unsqueeze(0)
            ho_distance = ho_distance * weights
            h2o_dist, _ = ho_distance.min(1)
            o2h_dist, _ = ho_distance.min(0)
            ho_distance = h2o_dist.mean() + o2h_dist.mean()
            loss_contact += ho_distance
            count += 1

        if count > 0:
            loss_contact /= count

        return {
            'contact': loss_contact.mean(),
        }


class OrdinalDepthLoss(nn.Module):
    # from CHORE

    def __init__(self, person_masks, object_masks, smpl_f, object_f, bboxes, focal):
        super().__init__()

        batch_size = person_masks.shape[0]

        focal = focal / bboxes[:, 2:] * np.array([192, 256]).reshape(1, 2) # 5000

        cam_Ks = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float().to(person_masks.device)
        cam_Ks[:, 0, 0] = torch.from_numpy(focal[:, 0]).float().to(person_masks.device)
        cam_Ks[:, 1, 1] = torch.from_numpy(focal[:, 1]).float().to(person_masks.device)
        cam_Ks[:, 0, 2] = cam_Ks[:, 1, 2] = 128
        cam_Ks = cam_Ks / 256
        R = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float().to(person_masks.device)
        t = torch.zeros(batch_size, 1, 3).float().to(person_masks.device)
        self.renderer = nr.renderer.Renderer(
            image_size=256, K=cam_Ks, R=R, t=t, orig_size=1, anti_aliasing=False
        )

        self.person_masks = person_masks > 0.5
        self.object_masks = object_masks > 0.5
        self.smpl_f = smpl_f
        self.object_f = object_f

        self.smpl_text = torch.ones(batch_size, smpl_f.shape[1], 1, 1, 1, 3).float().to(person_masks.device)
        self.object_text = torch.ones(batch_size, object_f.shape[1], 1, 1, 1, 3).float().to(person_masks.device)


    def forward(self, hoi_dict):
        smpl_v = hoi_dict['smplx_v']
        object_v = hoi_dict['object_v']
        _, smpl_depths, smpl_sils = self.renderer.render(smpl_v, self.smpl_f, self.smpl_text)
        _, object_depths, object_sils = self.renderer.render(object_v, self.object_f, self.object_text)
        smpl_sils = (smpl_sils == 1).bool()
        object_sils = (object_sils == 1).bool()

        # import cv2
        # cv2.imwrite('./__debug__/smpl_sils.jpg', (smpl_sils.detach().cpu().numpy()[0] * 255).astype(np.uint8))
        # cv2.imwrite('./__debug__/object_sils.jpg', (object_sils.detach().cpu().numpy()[0] * 255).astype(np.uint8))
        # cv2.imwrite('./__debug__/person_masks.jpg', (self.person_masks.detach().cpu().numpy()[0] * 255).astype(np.uint8))
        # cv2.imwrite('./__debug__/object_masks.jpg', (self.object_masks.detach().cpu().numpy()[0] * 255).astype(np.uint8))
        # exit(0)

        loss_depth = torch.zeros(1).to(smpl_v.device)
        count = 0

        for b_idx in range(smpl_v.shape[0]):

            has_pred = smpl_sils[b_idx] & object_sils[b_idx]
            if has_pred.sum() == 0:
                continue

            person_front_gt = self.person_masks[b_idx] & (~self.object_masks[b_idx])
            person_front_pred = object_depths[b_idx] < smpl_depths[b_idx]
            m = person_front_gt & person_front_pred & has_pred
            if m.sum() != 0:
                dists = torch.clamp(smpl_depths[b_idx] - object_depths[b_idx], min=0.0, max=2.0)
                loss_depth += torch.sum(torch.log(1 + torch.exp(dists))[m])
                count += 1

            object_front_gt = (~self.person_masks[b_idx]) & self.object_masks[b_idx]
            object_front_pred = object_depths[b_idx] > smpl_depths[b_idx]
            m = object_front_gt & object_front_pred & has_pred
            if m.sum() != 0:
                dists = torch.clamp(object_depths[b_idx] - smpl_depths[b_idx], min=0.0, max=2.0)
                loss_depth += torch.sum(torch.log(1 + torch.exp(dists))[m])
                count += 1

        if count > 0:
            loss_depth = loss_depth / count

        return {
            "depth": loss_depth.mean(),
        }


class PHOSAInteractionLoss(nn.Module):

    def __init__(self, batch_size, smpl_part_indices, object_part_indices, bboxes, focal, expansion_parts=0.5, interaction_threshold=5):
        super().__init__()
        self.mse = nn.MSELoss()

        focal = focal / bboxes[:, 2:] * np.array([192, 256]).reshape(1, 2) # 5000

        cam_Ks = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        cam_Ks[:, 0, 0] = torch.from_numpy(focal[:, 0]).float()
        cam_Ks[:, 1, 1] = torch.from_numpy(focal[:, 1]).float()
        cam_Ks[:, 0, 2] = cam_Ks[:, 1, 2] = 128
        cam_Ks = cam_Ks / 256
        R = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        t = torch.zeros(batch_size, 1, 3).float()

        self.register_buffer('cam_Ks', cam_Ks)
        self.register_buffer('R', R)
        self.register_buffer('t', t)

        self.smpl_part_indices = smpl_part_indices
        self.object_part_indices = object_part_indices
        self.expansion_parts = expansion_parts
        self.z_thresh = interaction_threshold


    def project_bbox(self, vertices, part_indices, bbox_expansion=0.0):
        proj = nr.projection(
            vertices * torch.tensor([1, -1, 1.0]).reshape(1, 1, 3).float().to(vertices.device),
            K=self.cam_Ks,
            R=self.R,
            t=self.t,
            dist_coeffs=torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]),
            orig_size=1
        )
        proj = proj[:, :, :2]

        bbox_parts = {}
        for parts, indices in part_indices.items():
            bbox = torch.cat([proj[:, indices].min(1).values, proj[:, indices].max(1).values], dim=1) # [b, 4]
            if bbox_expansion:
                center = (bbox[:, :2] + bbox[:, 2:]) / 2
                b = (bbox[:, 2:] - bbox[:, :2]) / 2 * (1 + bbox_expansion)
                bbox = torch.cat([center - b, center + b], dim=1)
            bbox_parts[parts] = bbox

        return bbox_parts


    def check_overlap(self, bbox1, bbox2):
        # bbox: [x1, y1, x2, y2]

        if bbox1[0] > bbox2[2] or bbox2[0] > bbox1[2]:
            return False

        if bbox1[1] > bbox2[3] or bbox2[1] > bbox1[3]:
            return False

        return True


    def compute_dist_z(self, verts1, verts2):
        a = verts1[:, 2].min()
        b = verts1[:, 2].max()
        c = verts2[:, 2].min()
        d = verts2[:, 2].max()
        if d >= a and b >= c:
            return 0.0
        return torch.min(torch.abs(c - b), torch.abs(a - d))


    def assign_interaction_parts(self, smpl_v, object_v):

        with torch.no_grad():
            person_part_bboxes = self.project_bbox(smpl_v, self.smpl_part_indices, self.expansion_parts)
            object_part_bboxes = self.project_bbox(object_v, self.object_part_indices, self.expansion_parts)

        batch_size = smpl_v.shape[0]
        interaction_pairs_parts = []
        for b_idx in range(batch_size):
            for smpl_part_name in person_part_bboxes.keys():
                for object_part_name in object_part_bboxes.keys():
                    bbox_object = object_part_bboxes[object_part_name][b_idx]
                    bbox_person = person_part_bboxes[object_part_name][b_idx]
                    is_overlapping = self.check_overlap(bbox_object, bbox_person)
                    z_dist = self.compute_dist_z(smpl_v[b_idx][self.smpl_part_indices[smpl_part_name]],
                                                 object_v[b_idx][self.object_part_indices[object_part_name]])
                    if is_overlapping and z_dist < self.z_thresh:
                        interaction_pairs_parts.append((b_idx, smpl_part_name, object_part_name))
        return interaction_pairs_parts


    def forward(self, hoi_dict):
        smpl_v = hoi_dict['smplx_v']
        object_v = hoi_dict['object_v']

        loss_inter = self.mse(smpl_v.mean(1), object_v.mean(1)) / smpl_v.shape[0]

        loss_parts = torch.zeros(1).to(smpl_v.device)
        interaction_pairs_parts = self.assign_interaction_parts(smpl_v, object_v)
        for part_pair in interaction_pairs_parts:
            b_idx, smpl_part, object_part = part_pair
            loss_parts += self.mse(smpl_v[b_idx][self.smpl_part_indices[smpl_part]].mean(0),
                                   object_v[b_idx][self.object_part_indices[object_part]].mean(0))

        if len(interaction_pairs_parts) != 0:
            loss_parts = loss_parts / len(interaction_pairs_parts)

        return {
            'inter': loss_inter.mean(),
            'inter_part': loss_parts.mean()
        }


# from sdf import SDF
# class HOCollisionLoss(nn.Module):
# # adapted from multiperson (links, multiperson.sdf.sdf_loss.py)

#     def __init__(self, smpl_faces, grid_size=32, robustifier=None,):
#         super().__init__()
#         self.sdf = SDF()
#         self.register_buffer('faces', torch.tensor(smpl_faces.astype(np.int32)))
#         self.grid_size = grid_size
#         self.robustifier = robustifier


#     @torch.no_grad()
#     def get_bounding_boxes(self, vertices):
#         # vertices: (n, 3)
#         boxes = torch.zeros(2, 3, device=vertices.device)
#         boxes[0, :] = vertices.min(dim=0)[0]
#         boxes[1, :] = vertices.max(dim=0)[0]
#         return boxes


#     @torch.no_grad()
#     def check_overlap(self, bbox1, bbox2):
#         # check x
#         if bbox1[0,0] > bbox2[1,0] or bbox2[0,0] > bbox1[1,0]:
#             return False
#         #check y
#         if bbox1[0,1] > bbox2[1,1] or bbox2[0,1] > bbox1[1,1]:
#             return False
#         #check z
#         if bbox1[0,2] > bbox2[1,2] or bbox2[0,2] > bbox1[1,2]:
#             return False
#         return True


#     def forward(self, hoi_dict, ):
#         # assume one person and one object
#         # person_vertices: (n, 3), object_vertices: (m, 3)
#         person_vertices, object_vertices = hoi_dict['smplx_v_centered'], hoi_dict['object_v_centered']
#         b = person_vertices.shape[0]
#         scale_factor = 0.2
#         loss = torch.zeros(1).float().to(object_vertices.device)

#         for b_idx in range(b):
#             person_bbox = self.get_bounding_boxes(person_vertices[b_idx])
#             object_bbox = self.get_bounding_boxes(object_vertices[b_idx])
#             if not self.check_overlap(person_bbox, object_bbox):
#                 return {'collision': loss.sum()}

#             person_bbox_center = person_bbox.mean(dim=0).unsqueeze(0)
#             person_bbox_scale = (1 + scale_factor) * 0.5 * (person_bbox[1] - person_bbox[0]).max()

#             with torch.no_grad():
#                 person_vertices_centered = person_vertices[b_idx] - person_bbox_center
#                 person_vertices_centered = person_vertices_centered / person_bbox_scale
#                 assert(person_vertices_centered.min() >= -1)
#                 assert(person_vertices_centered.max() <= 1)
#                 phi = self.sdf(self.faces, person_vertices_centered.unsqueeze(0))
#                 assert(phi.min() >= 0)

#             object_vertices_centered = (object_vertices[b_idx] - person_bbox_center) / person_bbox_scale
#             object_vertices_grid = object_vertices_centered.view(1, -1, 1, 1, 3)
#             phi_val = nn.functional.grid_sample(phi.unsqueeze(1), object_vertices_grid).view(-1)

#             cur_loss = phi_val
#             if self.robustifier:
#                 frac = (cur_loss / self.robustifier) ** 2
#                 cur_loss = frac / (frac + 1)

#             loss += cur_loss.sum()

#         return {'collision': loss.sum() / b}
