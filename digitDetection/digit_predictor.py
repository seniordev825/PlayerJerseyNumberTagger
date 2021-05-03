import cv2, time
import torch, os
import torchvision
import torchvision.transforms.functional as TF

# from utils.visualizations import *
from shapely.geometry import Polygon

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np

IOU_DIGIT_TORSO_THRES = 0.7


def get_keypoints(kp_predictions):
    return np.stack(kp_predictions.detach().squeeze().cpu().numpy().astype(int))


def get_blocking_rectangle(p1, p2, p3, p4):
    topleft = min(p1[0], p2[0], p3[0], p4[0]), min(p1[1], p2[1], p3[1], p4[1])
    bottomright = max(p1[0], p2[0], p3[0], p4[0]), max(p1[1], p2[1], p3[1], p4[1])
    return topleft, bottomright


def _colorize_mask(mask, color=None):
    b = mask * np.random.randint(0, 255) if not color else mask * color[0]
    g = mask * np.random.randint(0, 255) if not color else mask * color[1]
    r = mask * np.random.randint(0, 255) if not color else mask * color[2]
    return (b, g, r)


def get_torso_keypoints(keypoints):
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy().astype(int)
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    if left_shoulder[0] > right_shoulder[0]:
        left_shoulder, right_shoulder = right_shoulder, left_shoulder
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    if left_hip[0] > right_hip[0]:
        left_hip, right_hip = right_hip, left_hip
    return left_shoulder, right_shoulder, left_hip, right_hip


def get_torso_blocking_bb(l_shoulder, r_shoulder, l_hip, r_hip, h, w):
    l_shoulder, r_shoulder, l_hip, r_hip = get_torso_with_margin(l_shoulder, r_shoulder, l_hip, r_hip, h, w)
    topleft, bottomright = get_blocking_rectangle(l_shoulder, r_shoulder, l_hip, r_hip)
    return topleft[0], topleft[1], bottomright[1] - topleft[1], bottomright[0] - topleft[0]


def filter_digits_by_torso_iou(predictions, torso_pol):
    ious_indexes = []
    boxes, scores, labels = predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']
    for i, (x1, y1, x2, y2) in enumerate(boxes.detach().cpu().numpy()):
        digit_pol = Polygon([(int(x1), int(y1)), (int(x1), int(y2)), (int(x2), int(y2)), (int(x2), int(y1))])
        try:
            iou = torso_pol.intersection(digit_pol).area / digit_pol.area
        except:
            iou = 0
        if iou >= IOU_DIGIT_TORSO_THRES:
            ious_indexes.append(i)
    predictions = [dict(boxes=boxes[ious_indexes], scores=scores[ious_indexes], labels=labels[ious_indexes])]
    return predictions


def filter_bad_scale_location_digits(predictions):
    boxes = predictions[0]['boxes']
    idxs = []
    top_digit_coords = boxes.tolist()[0]
    polygons = [Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)]) for \
                x1, y1, x2, y2 in boxes.tolist()]
    top_digit_poly = polygons[0]
    for i, (pol, coords) in enumerate(zip(polygons[1:], boxes[1:].tolist())):
        if min(top_digit_poly.area / pol.area, pol.area / top_digit_poly.area) > 0.5:
            idxs.append(i)
    return idxs


def get_top_digits(predictions):
    boxes, scores, labels = predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']

    if len(predictions[0]['boxes']) > 2:
        top_digits_idxs = [0]  # digit with the highest score is always chosen
        top_digit_coords = boxes.tolist()[0]
        idxs = filter_bad_scale_location_digits(predictions)
        boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]
        if len(boxes) > 1:
            polygons = [Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)]) for \
                        x1, y1, x2, y2 in boxes.tolist()]
            top_digit_poly = polygons[0]
            digits_similarity_scores = [
                0.5 * max(top_digit_poly.area / pol.area, pol.area / top_digit_poly.area) +  # area ratio
                pol.distance(top_digit_poly) +  # distance between rectangles
                abs(top_digit_coords[1] - coords[1]) +  # abs distance between top y coords
                abs(top_digit_coords[3] - coords[3])  # abs distance between bottom y coords
                for (pol, coords) in zip(polygons[1:], boxes.tolist()[1:])]
            top_digits_idxs.append(np.argmax(digits_similarity_scores) + 1)
        predictions = [
            dict(boxes=boxes[top_digits_idxs], scores=scores[top_digits_idxs], labels=labels[top_digits_idxs])]
    # TODO: filter one digit if it's scale or location within the torso is problematic
    # elif len(boxes) is 1:

    return predictions


def get_digits_orientation(predictions):
    labels = predictions[0]['labels'].cpu().numpy().tolist()
    assert len(labels) <= 2
    if len(labels) is 0:
        return []
    elif len(labels) is 1:
        return [labels[0]]
    else:  # 2 digits
        # check if left coordinate is smaller to determine orientation
        if (predictions[0]['boxes'][0][0] < predictions[0]['boxes'][1][0]).tolist():
            return labels
        else:
            return list(reversed(labels))


class DigitPredictor():
    def __init__(self, score_thresh=0.6):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device('cpu')
        # load a model pre-trained pre-trained on COCO
        keywords = {'box_score_thresh': score_thresh}
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **keywords)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 11  # 10 digits + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # move model to the right device
        model.to(device)
        model.load_state_dict(torch.load("digitDetection\\digitSVHNmodel.pth", map_location=device))
        self.model = model

        self.model.eval()

    def __call__(self, image, torso_pts):
        trf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])])
        l_shoulder, r_shoulder, l_hip, r_hip = torso_pts  # get_torso_keypoints(keypoints)
        im = trf(image)  # .cuda()

        with torch.no_grad():
            predictions = self.model([im])
        orig_predictions = predictions
        torso_pol = Polygon([l_shoulder, r_shoulder, r_hip, l_hip])
        # filter digits with low IOU with player torso
        predictions = filter_digits_by_torso_iou(predictions, torso_pol)
        # filter and keep up to 2 digits
        # predictions = get_top_digits(predictions)
        # get digits orientation (units and tens digits)
        # digits = get_digits_orientation(predictions)

        # print(digits)
        # visualisations

        return predictions
        for (x1, y1, x2, y2), score, label in zip(orig_predictions[0]['boxes'].detach().cpu().numpy(),
                                                  orig_predictions[0]['scores'],
                                                  orig_predictions[0]['labels']):
            digit_pol = Polygon([(int(x1), int(y1)), (int(x1), int(y2)), (int(x2), int(y2)), (int(x2), int(y1))])
            iou = torso_pol.intersection(digit_pol).area / digit_pol.area
            if iou < IOU_DIGIT_TORSO_THRES:
                continue
            bbox_color = _colorize_mask(1)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 1)

            cv2.putText(image, str(label.cpu().numpy()), (int(x2), int(y2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        bbox_color, 1)
            cv2.putText(image, str(round(score.cpu().__float__(), 4)) + "%", (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
            # outputImage = cv2.copyMakeBorder(dst,50,50,50,50,cv2.BORDER_CONSTANT,value=(255,255,255))
            # scale_percent = 300  # percent of original size
            # width = int(outputImage.shape[1] * scale_percent / 100)
            # height = int(outputImage.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # resized = cv2.resize(outputImage, dim)
            # pts = np.array([l_shoulder, r_shoulder, r_hip, l_hip])
            # rect = cv2.boundingRect(pts)
            # x, y, w, h = rect
            # cropped = image[y:y + h, x:x + w].copy()
            # pts = pts - pts.min(axis=0)
            # mask = np.zeros(cropped.shape[:2], np.uint8)
            # cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            # dst = cv2.bitwise_and(cropped, cropped, mask=mask)
            cv2.imshow('image', image)
            cv2.waitKey(0)
            cv2.destroyWindow('image')
        cv2.destroyWindow('orig image')
        return digits


if __name__ == "__main__":
    num_of_preds = []
    torso_pts = [((10, 41), (51, 41), (23, 100), (46, 99)),
                 ((20, 35), (57, 35), (30, 85), (52, 85)),
                 ((25, 32), (36, 30), (14, 70), (25, 68)),
                 ((44, 30), (73, 35), (36, 79), (59, 83))]
    for i in range(0, 4):
        img_path = str(i) + ".jpeg"
        from PIL import Image

        image = Image.open(img_path)
        digit_pred = DigitPredictor()
        res = digit_pred(image, torso_pts[i])
        image2 = cv2.imread(img_path)
        for i, (x1, y1, x2, y2) in enumerate(res[0]['boxes']):
            score, label = res[0]['scores'][i], res[0]['labels'][i]
            bbox_color = _colorize_mask(1)

            cv2.rectangle(image2, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 1)
            cv2.putText(image2, str(label.data.cpu().numpy()), (int(x1), int(y1) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image2, str(label.data.cpu().numpy()), (int(x1), int(y1) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        image2 = cv2.resize(image2, (0, 0), fx=3, fy=3)
        cv2.imshow('image ' + str(i), image2)
        cv2.waitKey()
        cv2.destroyWindow('image ' + str(i))
#    cv2.destroyAllWindows()
