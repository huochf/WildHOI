import os
import sys
sys.path.append('/inspurfs/group/wangjingya/huochf/Thesis/')
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
import trimesh
from scipy.spatial.transform import Rotation

from smplx import SMPLX
from hoi_recon.datasets.utils import load_pickle, save_pickle, save_json
from hoi_recon.utils.evaluator import ReconEvaluator


def load_object_kps_indices(object_name):
    if object_name == 'barbell':
        object_file = 'data/objects/barbell_keypoints_12.json'
    elif object_name == 'cello':
        object_file = 'data/objects/cello_keypoints_14.json'
    elif object_name == 'baseball':
        object_file = 'data/objects/baseball_keypoints.json'
    elif object_name == 'tennis':
        object_file = 'data/objects/tennis_keypoints_7.json'
    elif object_name == 'skateboard':
        object_file = 'data/objects/skateboard_keypoints_8.json'
    elif object_name == 'basketball':
        object_file = 'data/objects/basketball_keypoints.json'
    elif object_name == 'yogaball':
        object_file = 'data/objects/yogaball_keypoints.json'
    elif object_name == 'violin':
        object_file = 'data/objects/violin_body_keypoints.json'
    with open(object_file, 'r') as f:
        indices = json.load(f)

    if object_name == 'baseball':
        indices = {'1': indices['1'], '5': indices['5']}
    elif object_name == 'barbell':
        indices = {'1': indices['1'], '2': indices['2'], '3': indices['3'], '4': indices['4'],}

    return indices


def generate_object_rotmat_candidates(object_v, kps_indices, rotmat, object_name, ):

    if object_name in ['cello', 'violin', 'bicycle', 'basketball',]:
        return rotmat.reshape(1, 3, 3)

    sym_matrices = []
    if object_name == 'barbell':
        axis_vector = object_v[kps_indices['1']].mean(axis=0) - object_v[kps_indices['4']].mean(axis=0)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        rot_angle_epsilon = 1
        for rot_angle in range(0, 360, rot_angle_epsilon):
            rot_angle = rot_angle / 180 * np.pi
            sym_rotmat = Rotation.from_rotvec(axis_vector * rot_angle).as_matrix()
            sym_matrices.append(sym_rotmat @ rotmat)
    elif object_name == 'baseball':
        axis_vector = object_v[kps_indices['1']].mean(axis=0) - object_v[kps_indices['5']].mean(axis=0)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        rot_angle_epsilon = 1
        for rot_angle in range(0, 360, rot_angle_epsilon):
            rot_angle = rot_angle / 180 * np.pi
            sym_rotmat = Rotation.from_rotvec(axis_vector * rot_angle).as_matrix()
            sym_matrices.append(sym_rotmat @ rotmat)
    elif object_name == 'skateboard':
        v1 = object_v[kps_indices['2']].mean(axis=0) - object_v[kps_indices['4']].mean(axis=0)
        v2 = object_v[kps_indices['2']].mean(axis=0) - object_v[kps_indices['8']].mean(axis=0)
        axis_vector = np.cross(v1, v2)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        sym_matrix = Rotation.from_rotvec(axis_vector * np.pi).as_matrix()
        sym_matrices.append(rotmat)
        sym_matrices.append(sym_matrix @ rotmat)
    elif object_name == 'tennis':
        axis_vector = object_v[kps_indices['1']].mean(axis=0) - object_v[kps_indices['2']].mean(axis=0)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        sym_matrix = Rotation.from_rotvec(axis_vector * np.pi).as_matrix()
        sym_matrices.append(rotmat)
        sym_matrices.append(sym_matrix @ rotmat)

    return np.stack(sym_matrices, axis=0)


def evaluate(args):

    exp_name = args.exp
    object_name = args.object

    smplx = SMPLX('data/smpl/smplx/', gender='neutral', use_pca=False)
    smpl_f = np.array(smplx.faces).astype(np.int64)

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('data/objects/{}_body.ply'.format(object_name), process=False)
        object_v_org = np.array(object_mesh.vertices)
        object_f = np.array(object_mesh.faces).astype(np.int64)
    elif object_name == 'bicycle':
        bicycle_front = trimesh.load('data/objects/bicycle_front.ply', process=False)
        bicycle_back = trimesh.load('data/objects/bicycle_back.ply', process=False)
        bicycle_front_v = np.array(bicycle_front.vertices)
        bicycle_back_v = np.array(bicycle_back.vertices)
        object_v_org = np.concatenate([bicycle_front_v, bicycle_back_v])
        object_f = np.concatenate([np.array(bicycle_front.faces), np.array(bicycle_back.faces) + bicycle_front_v.shape[0]])
    else:
        object_mesh = trimesh.load('data/objects/{}.ply'.format(object_name), process=False)
        object_v_org = np.array(object_mesh.vertices)
        object_f = np.array(object_mesh.faces).astype(np.int64)

    if object_name == 'bicycle':
        object_kps_indices = None
    else:
        object_kps_indices = load_object_kps_indices(object_name)

    annotations = {}
    annotations_dir = 'data/annotation_hoi/{}/test'.format(object_name)
    for file in os.listdir(annotations_dir):
        img_id = file.split('.')[0]
        annotations[img_id] = load_pickle(os.path.join(annotations_dir, file))

    reconstruction_results = load_pickle('./outputs/{}/{}_test.pkl'.format(exp_name, object_name))
    reconstruction_results = {item['image_id']: item for item in reconstruction_results}

    evaluator = ReconEvaluator(align_mesh=False, smpl_only=False)

    image_ids = list(annotations.keys())
    evaluate_metrics = {}
    for img_id in tqdm(image_ids):
        if img_id not in reconstruction_results:
            print('missing item {} .'.format(img_id))
            continue

        beta_recon = reconstruction_results[img_id]['smplx_betas']
        body_pose_recon = reconstruction_results[img_id]['smplx_body_pose']
        lhand_pose_recon = reconstruction_results[img_id]['smplx_lhand_pose']
        rhand_pose_recon = reconstruction_results[img_id]['smplx_rhand_pose']
        object_rel_rotmat_recon = reconstruction_results[img_id]['obj_rel_rotmat']
        object_rel_trans_recon = reconstruction_results[img_id]['obj_rel_trans']
        object_scale_recon = reconstruction_results[img_id]['object_scale']

        beta_gt = annotations[img_id]['smplx_betas']
        body_pose_gt = annotations[img_id]['smplx_body_pose']
        lhand_pose_gt = annotations[img_id]['smplx_lhand_pose']
        rhand_pose_gt = annotations[img_id]['smplx_rhand_pose']
        object_rel_rotmat_gt = annotations[img_id]['obj_rel_rotmat']
        object_rel_trans_gt = annotations[img_id]['obj_rel_trans']
        object_scale_gt = annotations[img_id]['object_scale']

        smpl_out_recon = smplx(betas=torch.tensor(beta_recon).reshape(1, 10), 
                               body_pose=torch.tensor(body_pose_recon).reshape(1, 63),
                               left_hand_pose=torch.tensor(lhand_pose_recon).reshape(1, 45),
                               right_hand_pose=torch.tensor(rhand_pose_recon).reshape(1, 45))
        smpl_v_recon = smpl_out_recon.vertices.detach().cpu().numpy()[0]
        smpl_J_recon = smpl_out_recon.joints.detach().cpu().numpy()[0]
        smpl_v_recon = smpl_v_recon - smpl_J_recon[:1]

        smpl_out_gt = smplx(betas=torch.tensor(beta_gt).reshape(1, 10), 
                            body_pose=torch.tensor(body_pose_gt).reshape(1, 63),
                            left_hand_pose=torch.tensor(lhand_pose_gt).reshape(1, 45),
                            right_hand_pose=torch.tensor(rhand_pose_gt).reshape(1, 45))
        smpl_v_gt = smpl_out_gt.vertices.detach().cpu().numpy()[0]
        smpl_J_gt = smpl_out_gt.joints.detach().cpu().numpy()[0]
        smpl_v_gt = smpl_v_gt - smpl_J_gt[:1]

        object_v_recon = (object_v_org * object_scale_recon.reshape(1, 1)) @ object_rel_rotmat_recon.reshape(3, 3).T + object_rel_trans_recon.reshape(1, 3)
        object_v_gt = (object_v_org * object_scale_gt.reshape(1, 1)) @ object_rel_rotmat_gt.reshape(3, 3).T + object_rel_trans_gt.reshape(1, 3)

        object_rotmat_candidates = generate_object_rotmat_candidates(object_v_gt, object_kps_indices, object_rel_rotmat_gt, object_name)

        error_trans = np.sqrt(((object_rel_trans_recon - object_rel_trans_gt) ** 2).sum())
        error_rot = np.arccos(
            ((np.trace(object_rel_rotmat_recon.reshape(1, 3, 3) @ object_rotmat_candidates.transpose(0, 2, 1), axis1=1, axis2=2) - 1) / 2).clip(-1, 1)
        )
        error_rot = error_rot * 180 / np.pi
        error_rot = error_rot.min()

        if object_name == 'basketball':
            error_rot = 0

        smpl_mesh_recon = trimesh.Trimesh(smpl_v_recon, smpl_f)
        smpl_mesh_gt = trimesh.Trimesh(smpl_v_gt, smpl_f)
        object_mesh_recon = trimesh.Trimesh(object_v_recon, object_f)
        object_mesh_gt = trimesh.Trimesh(object_v_gt, object_f)
        chamfer_distances = evaluator.compute_errors([smpl_mesh_gt, object_mesh_gt], [smpl_mesh_recon, object_mesh_recon])
        evaluate_metrics[img_id] = {
            'error_rot': float(error_rot),
            'error_trans': float(error_trans),
            'chamfer_smpl': float(chamfer_distances[0]),
            'chamfer_object': float(chamfer_distances[1]),
        }

    avg_rot = np.array([item['error_rot'] for item in evaluate_metrics.values()]).mean()
    avg_trans = np.array([item['error_trans'] for item in evaluate_metrics.values()]).mean()
    avg_smpl = np.array([item['chamfer_smpl'] for item in evaluate_metrics.values()]).mean()
    avg_object = np.array([item['chamfer_object'] for item in evaluate_metrics.values()]).mean()
    evaluate_metrics['avg'] = {
        'error_rot': float(avg_rot),
        'error_trans': float(avg_trans),
        'chamfer_smpl': float(avg_smpl),
        'chamfer_object': float(avg_object),
    }
    print(evaluate_metrics['avg'])
    save_json(evaluate_metrics, './outputs/{}/{}_test_evaluate_metrics.json'.format(exp_name, object_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize with KPS flow')
    parser.add_argument('--exp')
    parser.add_argument('--object')
    args = parser.parse_args()
    evaluate(args)
