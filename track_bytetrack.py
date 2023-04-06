import argparse

import torch.backends.cudnn as cudnn

import torchvision
from models.experimental import *
from utils.datasets import *
from utils.general import *
from utils.torch_utils import *
from utils.metrics import box_iou
# from utils.plots import Annotator, colors, save_one_box
from tracker.visualization import Annotator, colors
from models.common import DetectMultiBackend
from pathlib import Path
from tracker.byte_tracker import BYTETracker

from threading import Thread

down_stair = 'rtsp://user:remote456%2B@192.168.1.211/axis-media/media.amp?videocodec=h264'
front_door = 'rtsp://user:remote456%2B@192.168.1.212/axis-media/media.amp?videocodec=h264'
printer = 'rtsp://user:remote456%2B@192.168.1.213/axis-media/media.amp?videocodec=h264'
fabrication = 'rtsp://user:remote456%2B@192.168.1.214/axis-media/media.amp?videocodec=h264'
meeting_room = 'rtsp://user:remote456%2B@192.168.1.215/axis-media/media.amp?videocodec=h264'
living_room = 'rtsp://user:remote456%2B@192.168.1.216/axis-media/media.amp?videocodec=h264'
rd_department = 'rtsp://user:remote456%2B@192.168.1.217/axis-media/media.amp?videocodec=h264'


def detect(save_img=False):
    # Initialize

    device = select_device(0)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load models
    model_path = 'yolov5s.pt'
    model = DetectMultiBackend(model_path, device=device, dnn=False)

    # Initiate tracker
    tracker = BYTETracker()

    imgsz = check_img_size(1088, s=model.stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # video_path = '/home/user5/Downloads/plan feu stock dv.mp4'
    video_path = '/home/user5/Downloads/IJburgAmsterdam.mp4'
    # video_path = '/home/user5/WORKSPACE/DATASETS/eyeglass_detection/videos/glass_36.mp4'
    img_path = '/home/user5/Downloads/00000000-0000-0000-0000-00044becb9ba/24-Mar-2022-12h03m01s.jpg'

    cap = cv2.VideoCapture(living_room)
    # cap = cv2.VideoCapture(0)
    cv2.namedWindow('Vehicle tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 800)

    save_video = False
    visualize = False
    save_result = True
    results = []
    frame = 0

    tracklets = {}

    if save_video:
        fourcc = 'mp4v'  # output video codec

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        print('Video FPS: ', fps_video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter('out_.mp4', cv2.VideoWriter_fourcc(*fourcc), fps_video, (w, h))

    while cap.isOpened():
        ret, frame0 = cap.read()
        frame += 1
        # frame0 = cv2.resize(frame0, None, fx=0.125, fy=0.125)
        if not ret:
            break

        img = letterbox(frame0, new_shape=1088)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        visualize = Path('out_feature/') if visualize else False
        t1 = time_sync()
        pred = model(img, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred)
        t2 = time_sync()

        for i, det in enumerate(pred):
            annotator = Annotator(frame0)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame0.shape).round()

                output = tracker.update(det)

                for t in output:
                    xyxy = t.tlbr
                    tlwh = t.tlwh
                    c_x, c_y, _, _ = t.tlwh_to_xyah(tlwh)
                    track_id = int(t.track_id)
                    if track_id not in tracklets:
                        tracklets[track_id] = [(int(c_x), int(c_y))]
                    else:
                        tracklets[track_id].append((int(c_x), int(c_y)))
                    cls = int(t.cls)
                    # label = '%s %.2f' % (names[int(cls)], conf)
                    label = f'{names[int(cls)]} {track_id}'
                    annotator.draw_track(xyxy, label, color=colors(track_id, True), tracklet=tracklets[track_id])
                    results.append(
                        f"{frame},{track_id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{cls},-1,-1,-1\n")

            frame0 = annotator.result()
            fps = 1 / (t2 - t1)
            cv2.imshow('Vehicle tracking', frame0)
            if save_video:
                vid_writer.write(frame0)
        if cv2.waitKey(1) == 27:  # esc to quit
            break
    if save_result:
        res_file = 'result_40761.txt'
        with open(res_file, 'w') as f:
            f.writelines(results)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


if __name__ == '__main__':
    with torch.no_grad():
        detect()
