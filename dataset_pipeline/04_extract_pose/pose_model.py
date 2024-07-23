# borrow from PHALP_plus
import os
import numpy as np
import cv2

from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result


class PoseModel:

    MODEL_DICT = {
        'ViTPose-H-WholeBody': {
            'config': 'ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py',
            'weights': 'pretrained_models/vitpose_h_wholebody.pth',
        }
    }

    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self._load_model(model_name)


    def _load_model(self, name):
        dic = self.MODEL_DICT[name]
        weights_path = dic['weights']
        model = init_pose_model(dic['config'], weights_path, device=self.device)
        return model


    def predict_pose(self, 
                     image: np.ndarray, 
                     det_results: list[dict[str, np.ndarray]],
                     box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1] # RGB -> BGR
        out, _ = inference_top_down_pose_model(self.model, 
                                               image, 
                                               person_results=det_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out


    def visualize_pose_results(self, 
                               image: np.ndarray, 
                               pose_results: list[dict[str, np.ndarray]],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1] # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1] # BGR -> RGB
