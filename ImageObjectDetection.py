import torch
import torchvision
import torchvision.transforms.functional as TF
import os
import numpy as np
import cv2


class Detector():
    def __init__(self,score_thres=0.7,resize_flag=False):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,box_score_thresh=score_thres)
        self.is_cuda = torch.cuda.is_available()
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.score_thres = score_thres
        self.resize_flag = resize_flag

    def __call__(self, image):
        orig_image = image
        if self.resize_flag and (image.shape[0] > 512 or image.shape[1] > 512):
            from PIL import Image
            image = np.asarray(Image.fromarray(image).resize((512,512)))
        else:
            image = orig_image
        x = TF.to_tensor(image).cpu()
        if self.is_cuda:
            x = TF.to_tensor(image).cuda()

        with torch.no_grad():
            predictions = self.model([x])
        return get_bbs_info(predictions[0], orig_image, image)


def get_bbs_info(predictions, orig_image, resized_image, label=1):
    bbs_xywh, bbs_images, keypoints = [], [], []
    width_ratio = orig_image.shape[1] / resized_image.shape[1]
    height_ratio = orig_image.shape[0] / resized_image.shape[0]
    bbs = predictions['boxes']
    for i in range(len(bbs)):
        if predictions['labels'][i] == label:
            x1, x2 = int(bbs[i][0] * width_ratio), int(bbs[i][2] * width_ratio)
            y1, y2 = int(bbs[i][1] * height_ratio), int(bbs[i][3] * height_ratio)
            bbs_xywh.append((x1, y1, x2-x1, y2-y1))
            bbs_images.append(orig_image[y1:y2, x1:x2, :])
            kp = predictions['keypoints'][i].detach().squeeze().cpu().numpy()
            keypoints.append(kp)
    return bbs_images, bbs_xywh, keypoints


def do_detection(img_path):
    predic = Predictor(score_thres=0.6, resize_flag=False)
    image = cv2.imread(img_path)
    return predic(image)


if __name__ == '__main__':
    image_id = "game2"
    img_path = "utils" + os.path.sep + "lsd_1.6" + os.path.sep + "data" + os.path.sep + image_id + ".jpg"
    #img_path = "misc" + os.path.sep + "clip_18" + os.path.sep + "10.png"
    image = cv2.imread(img_path)
    pred = Predictor(score_thres=0.6,resize_flag=True)

    bbs_images, bbs_info = pred(image)

    resized = cv2.resize(image, (1000, 700))
    cv2.imshow('original', resized)
    for i in range(len(bbs_images)):
        img = bbs_images[i]
        scale_percent = 200  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim)
        cv2.imshow('bb' + str(i),resized)
        k = cv2.waitKey(0)
        cv2.destroyWindow('bb' + str(i))
    cv2.destroyAllWindows()
