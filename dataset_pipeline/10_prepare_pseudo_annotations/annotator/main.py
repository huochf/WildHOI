import os
import cv2
import argparse
from time import time
import pyglet
import trimesh
import pickle
import pyrender
from smplx import SMPLX
import numpy as np
from PIL import Image
import torch
from scipy.spatial.transform import Rotation

import annotator.config as config


class Annotator:

    def __init__(self, args):

        print('loading data ...')
        self.image_dir = args.image_dir
        self.image_files = sorted(os.listdir(self.image_dir))
        self.params_all = self.load_hoi_params(args.params_dir)
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print('loaded {} images'.format(len(self.image_files)))
        self.object_name = args.object

        self.smplx = SMPLX('data/smplx/', gender='neutral', use_pca=False)
        if args.object in ['cello', 'violin']:
            self.object_mesh = trimesh.load('data/objects/{}_body.ply'.format(args.object))
        else:
            self.object_mesh = trimesh.load('data/objects/{}.ply'.format(args.object))

        self.window, self.label = self.create_window()
        self.window.push_handlers(on_key_press=self.handle_keydown, on_key_release=self.handle_keyup)

        self.render_rate = config.RENDER_RATE
        self.translation_step = config.TRANSLATION_STEPS[1]
        self.z_step = config.TRANSLATION_Z_STEPS[1]
        self.rotation_step = config.ROTATION_STEPS[1]
        self.scale_step = config.SCALE_STEPS[1]

        self.object_scale = 1.0

        self.last_render_time = 0
        self.current_idx = 0
        self.smpl_color = np.array([255,127,80, 255]) / 255
        self.object_color = np.array([80,127,255, 255]) / 255
        self.scene, self.smpl_node, self.object_node = self.init_scene()
        self.camera_img, self.camera_front, self.camera_head, self.camera_side, self.camera_corner = self.init_cameras()
        self.init_light()
        self.renderer = pyrender.OffscreenRenderer(viewport_height=256, viewport_width=256, point_size=1.0)
        self.load_image()
        self.render()
        self.window.on_draw = self.refresh_window

        self.executing_actions = set()
        self.continue_actions = [self.translate_up, self.translate_down, self.translate_left, self.translate_right, 
            self.translate_forward, self.translate_backward, self.rotate_plus_y, self.rotate_minus_y, 
            self.rotate_plus_x, self.rotate_minus_x, self.rotate_plus_z, self.rotate_minus_z, ]


    @classmethod
    def load_hoi_params(cls, dir_):
        params_all = {}
        for file in os.listdir(dir_):
            img_id = file.split('.')[0]
            with open(os.path.join(dir_, file), 'rb') as f:
                params = pickle.load(f)
            params_all[img_id] = params
        return params_all


    def init_scene(self, ):
        scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5, 1.0], bg_color=(255, 255, 255))

        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        smpl_out = self.smplx(betas=torch.from_numpy(params['betas']).reshape(1, -1).float(), 
                          body_pose=torch.from_numpy(params['body_pose']).reshape(1, -1).float(), 
                          left_hand_pose=torch.from_numpy(params['lhand_pose']).reshape(1, -1).float(),
                          right_hand_pose=torch.from_numpy(params['rhand_pose']).reshape(1, -1).float(),)
        smpl_v = smpl_out.vertices.detach().numpy()
        smpl_J = smpl_out.joints.detach().numpy()
        smpl_v = smpl_v[0, ] - smpl_J[0, :1]
        smpl_f = self.smplx.faces.astype(np.int64)
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=self.smpl_color)
        smpl_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(smpl_v, smpl_f), material=material)
        smpl_node = pyrender.Node(mesh=smpl_mesh)
        scene.add_node(smpl_node)

        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=self.object_color)
        object_mesh = pyrender.Mesh.from_trimesh(self.object_mesh, material=material)
        object_node = pyrender.Node(mesh=object_mesh)
        scene.add_node(object_node)
        obj_pose = np.eye(4)
        object_rel_rotmat = params['object_rel_rotmat']
        object_rel_trans = params['object_rel_trans']
        obj_pose[:3, :3] = object_rel_rotmat
        obj_pose[:3, 3] = object_rel_trans
        obj_pose[:3, :3] *= self.object_scale
        scene.set_pose(object_node, obj_pose)

        return scene, smpl_node, object_node


    def init_cameras(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        fx, fy = params['focal']
        cx, cy = params['princpt']
        fx, fy, cx, cy = fx / 4, fy / 4, cx / 4, cy / 4 # 1024 -> 512
        camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy,)

        cam_R_org = params['cam_R']
        cam_T_org = - params['cam_T'].reshape(-1)
        cam_R =  cam_R_org.T @ Rotation.from_euler('x', 180, degrees=True).as_matrix()
        cam_T = (cam_R_org.T @ cam_T_org.reshape(3, 1)).reshape(3, )
        rotation = Rotation.from_matrix(cam_R)
        camera_img = pyrender.Node(camera=camera, rotation=rotation.as_quat(), translation=cam_T)

        rotmat_front = np.eye(3)
        trans_front = Rotation.from_euler('x', -180, degrees=True).as_matrix() @ cam_T_org.reshape(3, )
        rotation_front = Rotation.from_matrix(rotmat_front)
        camera_front = pyrender.Node(camera=camera, rotation=rotation_front.as_quat(), translation=trans_front.reshape(-1))

        rotmat_head = Rotation.from_euler('x', -90, degrees=True).as_matrix()
        trans_head = Rotation.from_euler('x', -90, degrees=True).as_matrix() @ Rotation.from_euler('x', -180, degrees=True).as_matrix() @ cam_T_org.reshape(3, 1)
        rotation_head = Rotation.from_matrix(rotmat_head)
        camera_head = pyrender.Node(camera=camera, rotation=rotation_head.as_quat(), translation=trans_head.reshape(-1))

        rotmat_side = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        rotation_side = Rotation.from_matrix(rotmat_side)
        trans_side = Rotation.from_euler('y', 90, degrees=True).as_matrix() @ Rotation.from_euler('x', -180, degrees=True).as_matrix()  @ cam_T_org.reshape(3, 1)
        camera_side = pyrender.Node(camera=camera, rotation=rotation_side.as_quat(), translation=trans_side.reshape(-1))

        rotmat_corner = Rotation.from_euler('x', -45, degrees=True).as_matrix() @ \
                        Rotation.from_euler('y', 45, degrees=True).as_matrix() 
        rotation_corner = Rotation.from_matrix(rotmat_corner)
        trans_corner = Rotation.from_euler('x', -45, degrees=True).as_matrix() @ \
                        Rotation.from_euler('y', 45, degrees=True).as_matrix() @ Rotation.from_euler('x', -180, degrees=True).as_matrix() @ cam_T_org.reshape(3, 1)
        camera_corner = pyrender.Node(camera=camera, rotation=rotation_corner.as_quat(), translation=trans_corner.reshape(-1))

        return camera_img, camera_front, camera_head, camera_side, camera_corner


    def init_light(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]

        cam_R_org = params['cam_R']
        cam_T = - params['cam_T'].reshape(-1)
        cam_R =  cam_R_org.T @ Rotation.from_euler('x', 180, degrees=True).as_matrix()
        # cam_R = np.eye(3)
        cam_T = (cam_R_org.T @ cam_T.reshape(3, 1)).reshape(3, )
        pose = np.eye(4)
        pose[:3, :3] = cam_R
        pose[:3, 3] = cam_T
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.scene.add(light, pose=pose)

        rotmat_side = Rotation.from_euler('y', 90, degrees=True).as_matrix() @ cam_R
        rotation_side = Rotation.from_matrix(rotmat_side)
        trans_side = Rotation.from_euler('y', 90, degrees=True).as_matrix() @ cam_T.reshape(3, 1)
        pose = np.eye(4)
        pose[:3, :3] = rotation_side.as_matrix()
        pose[:3, 3] = trans_side.reshape(3, )
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.scene.add(light, pose=pose)

        rotmat_corner = Rotation.from_euler('y', 45, degrees=True).as_matrix() @ cam_R
        trans_corner = Rotation.from_euler('y', 45, degrees=True).as_matrix() @ cam_T.reshape(3, 1)
        rotation_corner = Rotation.from_matrix(rotmat_corner)
        pose = np.eye(4)
        pose[:3, :3] = rotation_corner.as_matrix()
        pose[:3, 3] = trans_corner.reshape(3, )
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.scene.add(light, pose=pose)


    def refresh_smpl(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        smpl_out = self.smplx(betas=torch.from_numpy(params['betas']).reshape(1, -1).float(), 
                          body_pose=torch.from_numpy(params['body_pose']).reshape(1, -1).float(), 
                          left_hand_pose=torch.from_numpy(params['lhand_pose']).reshape(1, -1).float(),
                          right_hand_pose=torch.from_numpy(params['rhand_pose']).reshape(1, -1).float(),)
        smpl_v = smpl_out.vertices.detach().numpy()
        smpl_J = smpl_out.joints.detach().numpy()
        smpl_v = smpl_v[0, ] - smpl_J[0, :1]
        smpl_f = self.smplx.faces.astype(np.int64)

        self.scene.remove_node(self.smpl_node)
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=self.smpl_color)
        smpl_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(smpl_v, smpl_f), material=material)
        self.smpl_node = pyrender.Node(mesh=smpl_mesh)
        self.scene.add_node(self.smpl_node) 


    def refresh_object(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        obj_pose = np.eye(4)
        object_rel_rotmat = params['object_rel_rotmat']
        object_rel_trans = params['object_rel_trans']
        obj_pose[:3, :3] = object_rel_rotmat
        obj_pose[:3, 3] = object_rel_trans
        obj_pose[:3, :3] *= params['object_scale'] # self.object_scale
        self.scene.set_pose(self.object_node, obj_pose)


    def refresh_cameras(self, ):
        self.camera_img, self.camera_front, self.camera_head, self.camera_side, self.camera_corner = self.init_cameras()


    def load_image(self, ):
        current_image_file = self.image_files[self.current_idx]
        image = np.array(Image.open(os.path.join(self.image_dir, current_image_file)))[..., :3]
        image = cv2.resize(image, dsize=(512, 512), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        self.current_image = image


    def render(self, ):
        current_time = time()
        if current_time - self.last_render_time <= self.render_rate:
            return
        self.last_render_time = current_time
        self.refresh_object()
        self.scene.add_node(self.camera_img)
        render_img, _ = self.renderer.render(self.scene, ) # flags=pyrender.RenderFlags.ALL_WIREFRAME)
        self.scene.remove_node(self.camera_img)

        self.scene.add_node(self.camera_front)
        render_front, _ = self.renderer.render(self.scene, ) # flags=pyrender.RenderFlags.ALL_WIREFRAME)
        self.scene.remove_node(self.camera_front)

        self.scene.add_node(self.camera_head)
        render_head, _ = self.renderer.render(self.scene, ) # flags=pyrender.RenderFlags.ALL_WIREFRAME)
        self.scene.remove_node(self.camera_head)

        self.scene.add_node(self.camera_side)
        render_side, _ = self.renderer.render(self.scene, ) # flags=pyrender.RenderFlags.ALL_WIREFRAME)
        self.scene.remove_node(self.camera_side)

        self.scene.add_node(self.camera_corner)
        render_corner, _ = self.renderer.render(self.scene,) # flags=pyrender.RenderFlags.ALL_WIREFRAME)
        self.scene.remove_node(self.camera_corner)

        render_img = cv2.resize(render_img, dsize=(512, 512), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        mask = np.all(render_img == 255, axis=-1, keepdims=True)
        final_image = np.where(mask, self.current_image, (1 - config.OVERLAY_OPACITY) * self.current_image + config.OVERLAY_OPACITY * render_img)
        image_show = np.concatenate([np.concatenate([render_front, render_head], axis=0),
            final_image,
            np.concatenate([render_side, render_corner], axis=0)], axis=1)
        self.current_image_show = image_show.astype(np.uint8)


    def refresh_window(self, ):
        self.window.clear()
        self.label.text = '{}  [{}/{}]'.format(self.image_files[self.current_idx].split('.')[0], self.current_idx, len(self.image_files))
        self.label.anchor_x = 'center'
        self.label.x = self.window.width // 2
        self.label.anchor_y = 'top'
        self.label.y = self.window.height
        self.label.draw()

        self.window.switch_to()
        image_show = self.current_image_show
        h, w, _ = image_show.shape
        resize_factor = min(self.window.width / w, self.window.height / h)
        resize_width, resize_height = int(w * resize_factor), int(h * resize_factor)
        image_data = pyglet.image.ImageData(w, h, 'RGB', image_show.tobytes(), pitch=-3*w)
        image_data.anchor_x = resize_width // 2
        image_data.anchor_y = resize_height
        image_data.blit(self.window.width // 2, self.window.height - self.label.content_height, width=resize_width, height=resize_height)


    @classmethod
    def create_window(cls):
        window = pyglet.window.Window(width=512 + 256 + 256, height=512, resizable=True)
        label = pyglet.text.Label(anchor_x='center')
        return window, label


    def do_action(self, action=None):
        # for action in self.executing_actions:
        if action not in self.executing_actions:
            return
        action()
        if action in self.continue_actions:
            pyglet.clock.schedule_once(lambda _: self.do_action(action), config.ACTION_DELAY)
        self.render()


    def handle_keydown(self, symbol, modifiers):
        action_name = config.KEYBINDINGS.get(symbol)
        if not action_name:
            return

        action = getattr(self, action_name, None)
        if not action:
            return
        self.executing_actions.add(action)
        self.do_action(action)


    def handle_keyup(self, symbol, modifiers):
        print(symbol)
        action_name = config.KEYBINDINGS.get(symbol)
        if not action_name:
            return

        action = getattr(self, action_name, None)
        if not action:
            return
        self.executing_actions.remove(action)


    def next_image(self, ):
        self.write_pose()
        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        self.load_image()
        self.refresh_smpl()
        self.refresh_object()
        self.refresh_cameras()


    def previous_image(self, ):
        self.write_pose()
        self.current_idx = (self.current_idx - 1) % len(self.image_files)
        self.load_image()
        self.refresh_smpl()
        self.refresh_object()
        self.refresh_cameras()


    def write_pose(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        with open(os.path.join(self.output_dir, '{}.pkl'.format(img_id)), 'wb') as f:
            pickle.dump(params, f)
            print('save annotation to {}'.format(os.path.join(self.output_dir, '{}.pkl'.format(img_id))))


    def translate_up(self):
        print('translate_up')
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        params['object_rel_trans'][1] += self.translation_step


    def translate_down(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        params['object_rel_trans'][1] -= self.translation_step


    def translate_right(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        params['object_rel_trans'][0] += self.translation_step


    def translate_left(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        params['object_rel_trans'][0] -= self.translation_step


    def translate_backward(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        params['object_rel_trans'][2] -= self.z_step


    def translate_forward(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        params['object_rel_trans'][2] += self.z_step


    def rotate_plus_x(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        rotation = Rotation.from_matrix(params['object_rel_rotmat'])
        rotmat = rotation.as_matrix() @ Rotation.from_euler('zyx', [0, 0, self.rotation_step], degrees=True).as_matrix()
        params['object_rel_rotmat'] = rotmat


    def rotate_minus_x(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        rotation = Rotation.from_matrix(params['object_rel_rotmat'])
        rotmat = rotation.as_matrix() @ Rotation.from_euler('zyx', [0, 0, - self.rotation_step], degrees=True).as_matrix()
        params['object_rel_rotmat'] = rotmat


    def rotate_plus_y(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        rotation = Rotation.from_matrix(params['object_rel_rotmat'])
        rotmat = rotation.as_matrix() @ Rotation.from_euler('zyx', [0, self.rotation_step, 0], degrees=True).as_matrix()
        params['object_rel_rotmat'] = rotmat


    def rotate_minus_y(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        rotation = Rotation.from_matrix(params['object_rel_rotmat'])
        rotmat = rotation.as_matrix() @ Rotation.from_euler('zyx', [0, -self.rotation_step, 0], degrees=True).as_matrix()
        params['object_rel_rotmat'] = rotmat


    def rotate_plus_z(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        rotation = Rotation.from_matrix(params['object_rel_rotmat'])
        rotmat = rotation.as_matrix() @ Rotation.from_euler('zyx', [self.rotation_step, 0, 0], degrees=True).as_matrix()
        params['object_rel_rotmat'] = rotmat


    def rotate_minus_z(self):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        rotation = Rotation.from_matrix(params['object_rel_rotmat'])
        rotmat = rotation.as_matrix() @ Rotation.from_euler('zyx', [- self.rotation_step, 0, 0], degrees=True).as_matrix()
        params['object_rel_rotmat'] = rotmat


    def scale_up(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        params['object_scale'] += self.scale_step
        # self.object_scale += self.scale_step


    def scale_down(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        params['object_scale'] -= self.scale_step
        # self.object_scale -= self.scale_step


    def step_size1(self, ):
        self.translation_step = config.TRANSLATION_STEPS[0]
        self.z_step = config.TRANSLATION_Z_STEPS[0]
        self.rotation_step = config.ROTATION_STEPS[0]
        self.scale_step = config.SCALE_STEPS[0]


    def step_size2(self, ):
        self.translation_step = config.TRANSLATION_STEPS[1]
        self.z_step = config.TRANSLATION_Z_STEPS[1]
        self.rotation_step = config.ROTATION_STEPS[1]
        self.scale_step = config.SCALE_STEPS[1]


    def step_size3(self, ):
        self.translation_step = config.TRANSLATION_STEPS[2]
        self.z_step = config.TRANSLATION_Z_STEPS[2]
        self.rotation_step = config.ROTATION_STEPS[2]
        self.scale_step = config.SCALE_STEPS[2]


    def reset(self, ):
        img_id = self.image_files[self.current_idx].split('.')[0]
        params = self.params_all[img_id]
        # params['object_rel_rotmat'] = np.eye(3)
        params['object_rel_trans'] = np.zeros(3)
        # params['object_scale'] = 1.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/images')
    parser.add_argument('--params_dir', type=str, default='data/params')
    parser.add_argument('--output_dir', type=str, default='data/annotations')
    parser.add_argument('--object', type=str, default='barbell')
    args = parser.parse_args()

    annotator = Annotator(args)
    pyglet.app.run()
