from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import torch



from utils import readb64, img2base64

model_int8 = YOLO('weights/best.torchscript', task='detect')

labels = {
    0: 'mask_weared_incorrect',
    1: 'with_mask',
    2: 'without_mask',
}


def inference_on_image(path):
    results = model_int8(path)

    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    for box in results[0].boxes:
        img = draw_bbox_prediction(img, box)
        
    cv2.imshow('Detected Image', img)
    cv2.waitKey(0)

    return results

def inference_on_video(path, vid_stride=10):
    results = model_int8(path, vid_stride=10, stream=True)

    cap = cv2.VideoCapture(path)
    ret, img = cap.read()

    frame_counter = 0
    while True:
        ret, img = cap.read()
        if ret:
            if frame_counter % 10 == 0: 
                result = next(results)
            for box in result.boxes:
                img = draw_bbox_prediction(img, box)
        else:
            cap.release()
            break
        
        cv2.imshow('Detected Image', img)
        frame_counter += 1

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

    return results

def draw_bbox_prediction(img, box):
    cls = box.cls.item()
    confidence = box.conf.item()
    label = labels[cls]

    x1, y1, x2, y2 = map(int, list(box.xyxy.numpy()[0]))
    scaler = (x2-x1)/(640/8)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 102, 255), int(2*scaler))
    img = cv2.rectangle(img, (x1, y1 - int(20*scaler)), (x1 + (x2 - x1)*3, y1), (0, 102, 255), -1)
    img = cv2.putText(img, "{}: {:.3f}".format(label, confidence), (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6*scaler,(255,255,255), int(1*scaler))
    return img


class ImagePipeline:
    def __init__(self, device='cpu', gpu_id=0, weights='weights/best.torchscript'):
        self.model = YOLO(weights, task='detect')

    def preprocess(self, data):
        image_base64 = data.pop("images", data)

        if not type(image_base64) == list:
            image_base64 = [image_base64]
        elif len(image_base64) > 1:
            raise Exception("ImagePipeline only accepts 1 image/frame")
        
        images = [readb64(image) for image in image_base64]
        return images

    def inference(self, images):
        results = self.model(images[0])
        return results

    def get_response(self, inference_result):
        response = []

        if not bool(set([0, 2]).intersection(inference_result[0].boxes.cls.numpy())):
        # if not set([0, 2]).issubset(inference_result[0].boxes.cls.numpy()):
            message = "Everyone is wearing mask correctly"
        else:
            message = "Someone is not wearing mask or incorrectly wearing mask"

        for i, result in enumerate(inference_result):
            for xywhn, cls, conf in zip(
                result.boxes.xywhn,
                result.boxes.cls,
                result.boxes.conf
            ):
                xywhn = list(xywhn.numpy())
                response.append({
                    'xywhn': {
                        'x': float(xywhn[0]),
                        'y': float(xywhn[1]),
                        'w': float(xywhn[2]),
                        'h': float(xywhn[3]),
                    },
                    'class': cls.item(),
                    'confidence': conf.item(),
                })

        return {'results': response,
                'message': message}

    def draw_bbox(self, images, inference_result):
        img = np.array(images[0])
        boxes = list(inference_result[0].boxes)
        boxes.reverse()


        for box in boxes:
            img = draw_bbox_prediction(img, box)
        
        return img
    
    def __call__(self, data, config_payload=None, draw_bbox=False):
        images = self.preprocess(data)
        inference_result = self.inference(images)
        response = self.get_response(inference_result)
        if draw_bbox:
            annotated_img = self.draw_bbox(images, inference_result)
            return response, annotated_img
        return response

class VideoPipeline:
    def __init__(self, device='cpu', gpu_id=0, weights='weights/best.torchscript'):
        self.model = YOLO(weights, task='detect')

    def preprocess(self, data):
        return data

    def inference(self, video_path, vid_stride=30):
        results = self.model(video_path, vid_stride=vid_stride)
        return results

    def get_response(self, inference_result):
        response = []

        
        # default message
        message = "Everyone is wearing mask correctly"

        for i, result in enumerate(inference_result):
            
            if set([0, 2]).issubset(inference_result[0].boxes.cls.numpy()):
                message = "Someone is not wearing mask or incorrectly wearing mask"

            for xywhn, cls, conf in zip(
                result.boxes.xywhn,
                result.boxes.cls,
                result.boxes.conf
            ):
                xywhn = list(xywhn.numpy())
                response.append({
                    'xywhn': {
                        'x': float(xywhn[0]),
                        'y': float(xywhn[1]),
                        'w': float(xywhn[2]),
                        'h': float(xywhn[3]),
                    },
                    'class': cls.item(),
                    'confidence': conf.item(),
                })

        return {'results': response,
                'message': message}
    
    def __call__(self, data, config_payload=None):
        data = self.preprocess(data)
        inference_result = self.inference(data)
        response = self.get_response(inference_result)
        return response


if __name__ == '__main__':
    import cv2
    import argparse
 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_type',
                    default='image',
                    const='image',
                    nargs='?',
                    choices=['image', 'video'],
                    help='type of input (default: %(default)s)')
    parser.add_argument("-p", "--path", help="filepath")
    args = parser.parse_args()

    if args.input_type=='image':
        results = inference_on_image(args.path)
    elif args.input_type == 'video':
        results = inference_on_video(args.path)
    
    
    print(results)

    
    # Examples
    # python pipelines.py --input_type image --path sample_files/image-1.jpeg
    # python pipelines.py --input_type video --path sample_files/video-1.mp4