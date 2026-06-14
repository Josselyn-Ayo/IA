import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import base64
import numpy as np


class Generator(nn.Module):
    def __init__(self, nc=4, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.network(input)
        return output


model = Generator()
weights_path = hf_hub_download('nateraw/cryptopunks-gan', 'generator.pth')
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

out = model(torch.randn(64, 100, 1, 1))
save_image(out, "punks.png", normalize=True)
print("Generated punks.png with 64 punks!")

img = mpimg.imread("punks.png")
plt.imshow(img)
plt.axis("off")
plt.title("Generated CryptoPunks")
plt.show()

print("Exporting model weights for web...")
weights = {}
for name, param in model.named_parameters():
    arr = param.detach().cpu().numpy()
    weights[name] = {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'data': base64.b64encode(arr.tobytes()).decode('ascii')
    }
for name, buf in model.named_buffers():
    arr = buf.detach().cpu().numpy()
    weights[name] = {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'data': base64.b64encode(arr.tobytes()).decode('ascii')
    }

with open('model_weights.json', 'w') as f:
    json.dump(weights, f)

print("Created model_weights.json - upload this file with your web page to Vercel!")