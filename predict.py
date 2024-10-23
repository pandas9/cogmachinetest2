from Models import VisionModel
from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF
from cog import BasePredictor, Input, Path

THRESHOLD = 0.4

class Predictor(BasePredictor):
    def setup(self):
        path = './models'  # Change this to where you downloaded the model
        self.model = VisionModel.load_model(path)
        self.model.eval()
        self.model = self.model.to('cuda')

        with open(Path(path) / 'top_tags.txt', 'r') as f:
            self.top_tags = [line.strip() for line in f.readlines() if line.strip()]

    def prepare_image(self, image: Image.Image, target_size: int) -> torch.Tensor:
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        image_tensor = TVF.pil_to_tensor(padded_image) / 255.0
        image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        return image_tensor

    @torch.no_grad()
    def predict(self, image: Path = Input(description="Input image")) -> str:
        print('*******************')
        print(image)
        print('*******************')
        input_image = Image.open(image)
        image_tensor = self.prepare_image(input_image, self.model.image_size)
        batch = {'image': image_tensor.unsqueeze(0).to('cuda')}

        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            preds = self.model(batch)
            tag_preds = preds['tags'].sigmoid().cpu()

        scores = {self.top_tags[i]: tag_preds[0][i] for i in range(len(self.top_tags))}
        predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
        tag_string = ', '.join(predicted_tags)

        return tag_string
