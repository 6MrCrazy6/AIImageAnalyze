import requests
import torch
import os
import clip

from PIL import Image, ImageDraw
from ultralytics import YOLO
from torchvision import models, transforms
from download_assets import MODELS_DIR

class ObjectDetector:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "yolov10m.pt")
        self.model = YOLO(model_path)

    def detect(self, image_path):
        results = self.model(image_path, verbose=False)[0]
        names = results.names
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls.item())
            label = names[cls_id]
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            detections.append({"label": label, "coords": coords})

        return detections

    def draw_boxes(self, image_path, detections):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for det in detections:
            x1, y1, x2, y2 = det["coords"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 5, y1 + 5), det["label"], fill="red")
        return image

class Classifier:
    def __init__(self):
        self.api_user = "622806118"
        self.api_secret = "wmFnKWhBqe4gLrzgPRYiuAKtNc8VvHNk"
        self.url = "https://api.sightengine.com/1.0/check.json"

    def classify_type(self, type_dict):
        if not type_dict:
            print("Could not determine image type (empty dictionary).")
            return None

        most_likely_type = max(type_dict, key=type_dict.get)
        print("Image type:", most_likely_type)
        return most_likely_type

    def classify(self, image_path):
        with open(image_path, "rb") as image_file:
            files = {'media': image_file}
            params = {
                'models': 'type',
                'api_user': self.api_user,
                'api_secret': self.api_secret
            }

            response = requests.post(self.url, files=files, data=params)

        if response.status_code == 200:
            result = response.json()
            type_info = result.get("type")
            return self.classify_type(type_info)
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None


class Segmenter:
    def __init__(self):
        weight_path = os.path.join(MODELS_DIR, "deeplabv3_resnet101_coco-586e9e4e.pth")
        self.model = models.segmentation.deeplabv3_resnet101(aux_loss=True)
        state_dict = torch.load(weight_path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def segment(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)["out"][0]
        mask = output.argmax(0).byte().cpu().numpy()
        return Image.fromarray(mask)


class CLIPContextAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def analyze(self, image_path):
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        categories = ["person", "tree", "building", "nature", "car", "animal", "flower", "camera", "laptop", "phone"]
        import clip
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {category}") for category in categories]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)
        values, indices = similarity.cpu().topk(3)

        return [categories[idx] for idx in indices]


class ImageAnalyzer:
    def __init__(self):
        self.detector = ObjectDetector()
        self.classifier = Classifier()
        self.segmenter = Segmenter()
        self.clip_analyzer = CLIPContextAnalyzer()

    def perform_extended_analysis(self, image_path):
        detections = self.detector.detect(image_path)
        classification = self.classifier.classify(image_path)
        segmentation = self.segmenter.segment(image_path)
        clip_analysis = self.clip_analyzer.analyze(image_path)

        count_by_label = {}
        for det in detections:
            label = det["label"]
            count_by_label[label] = count_by_label.get(label, 0) + 1

        results = {
            "detected_objects": detections,
            "classification": classification,
            "segmentation": segmentation,
            "clip_analysis": clip_analysis,
            "object_counts": count_by_label
        }

        return results

    def draw_annotated_image(self, image_path):
        detections = self.detector.detect(image_path)
        return self.detector.draw_boxes(image_path, detections)
