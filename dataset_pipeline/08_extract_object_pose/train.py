import os
import numpy as np
import torch
import cv2
from models import Model
from datasets.object_corr_dataset import ObjCorrDataset


def visualize(outputs, batch):
    images, masks, coor, _ = batch
    corr_maps_pred = outputs
    bs, _, h, w = images.shape
    images = images.numpy()
    coor = coor.numpy()
    std = np.array((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    mean = np.array((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    images = (images * std + mean) * 255
    images = images.clip(0, 255).astype(np.uint8)

    image_show = np.ones((h * 4, w * 4, 3), dtype=np.uint8) * 255
    for b_idx in range(min(bs, 4)):
        image = images[b_idx].transpose(1, 2, 0)
        image_corr_gt = coor[b_idx].transpose(1, 2, 0)
        image_corr_gt = ((image_corr_gt + 1) / 2 * 255.).clip(0, 255).astype(np.uint8)
        image_corr_gt = cv2.resize(image_corr_gt, dsize=(w, h))

        image_corr_pred = corr_maps_pred[b_idx, :3].transpose(1, 2, 0)
        image_corr_pred = ((image_corr_pred + 1) / 2 * 255.).clip(0, 255).astype(np.uint8)
        image_corr_pred = cv2.resize(image_corr_pred, dsize=(w, h))

        mask_pred = corr_maps_pred[b_idx, 3:].transpose(1, 2, 0)
        mask_pred = np.concatenate([mask_pred, mask_pred, mask_pred], axis=2)
        mask_pred = (mask_pred * 255).clip(0, 255).astype(np.uint8)
        mask_pred = cv2.resize(mask_pred, dsize=(w, h))

        image_show[h * b_idx: h*b_idx + h, 0:w] = image
        image_show[h * b_idx: h*b_idx + h, w:w*2] = image_corr_gt
        image_show[h * b_idx: h*b_idx + h, w*2:w*3] = image_corr_pred
        image_show[h * b_idx: h*b_idx + h, w*3:w*4] = mask_pred
    return image_show


def train():
    device = torch.device('cuda')
    object_name = 'barbell'
    os.makedirs('./weights', exist_ok=True)
    os.makedirs('./__debug__/{}'.format(object_name), exist_ok=True)

    model = Model(num_kps=12).to(device)
    dataset = ObjCorrDataset(root_dir='/storage/data/huochf/HOIYouTube/{}'.format(object_name), 
        corr_dir='/storage/data/huochf/HOIYouTube/{}/object_annotations/corr'.format(object_name),
        out_res=224, coor_res=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8, shuffle=True)

    dataset_test = ObjCorrDataset(root_dir='/storage/data/huochf/HOIYouTube-test/{}'.format(object_name), 
        corr_dir='/storage/data/huochf/HOIYouTube-test/{}/object_annotations/corr'.format(object_name),
        out_res=224, coor_res=64)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, num_workers=8, shuffle=True)

    if os.path.exists('./weights/model_{}_stage1_test.pth'.format(object_name)):
        begin_epoch = model.load_checkpoint('./weights/model_{}_stage1_test.pth'.format(object_name))
    else:
        begin_epoch = 0
    for epoch in range(begin_epoch, 100000):

        model.train()
        for idx, data in enumerate(dataloader):
            images, masks, coor, coor_sym = data
            images = images.to(device)
            masks = masks.to(device)
            coor = coor.to(device)
            coor_sym = coor_sym.to(device)

            loss_dict = model.train_step(images, (coor, coor_sym, masks))

            if idx % 10 == 0:
                print('[{} / {}], loss: {:.4f}, loss_corr: {:.4f}, loss_mask: {:.4f}'.format(
                    epoch, idx, loss_dict['loss'].item(), loss_dict['loss_corr'].item(), loss_dict['loss_mask'].item()))
        for idx, data in enumerate(dataloader_test):
            images, masks, coor, coor_sym = data
            images = images.to(device)
            masks = masks.to(device)
            coor = coor.to(device)
            coor_sym = coor_sym.to(device)

            loss_dict = model.train_step(images, (coor, coor_sym, masks))

            if idx % 10 == 0:
                print('[{} / {}], loss: {:.4f}, loss_corr: {:.4f}, loss_mask: {:.4f}'.format(
                    epoch, idx, loss_dict['loss'].item(), loss_dict['loss_corr'].item(), loss_dict['loss_mask'].item()))

        if epoch % 10 == 0:
            model.eval()
            for idx, data in enumerate(dataloader_test):
                if idx > 10:
                    break
                images, masks, coor, coor_sym = data
                images = images.to(device)

                outputs = model.inference_step(images)
                image = visualize(outputs, data)
                cv2.imwrite('./__debug__/{}/{}_{}.jpg'.format(object_name, epoch, idx), image[:, :, ::-1])

            model.save_checkpoint(epoch, './weights/model_{}_stage1_test.pth'.format(object_name))


if __name__ == '__main__':
    train()
