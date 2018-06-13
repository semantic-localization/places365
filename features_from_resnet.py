# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision import transforms as trn
from torch.nn import functional as F
import numpy as np
from PIL import Image

import os, sys
from glob import glob
import pickle


# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '{}/my-checkpoints/{}_places365.pth.tar'.format(os.environ['HOME'], arch)
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)


features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

features_names = ['avgpool'] # this is the last conv layer of the resnet
for name in features_names:
    model._modules.get(name).register_forward_hook(hook_feature)

# set model to eval mode - no training
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load images
img_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/grocery_data/Traderjoe/StPaul'
img_files = sorted(glob('{}/rectification/cylindrical/*.jpg'.format(img_dir)))
img_map = np.array([int(img_file.strip().split('/')[-1].split('.')[0][-7:]) for img_file in img_files])
n_files = len(img_files)
batch_size = 64
num_batches = int(n_files/batch_size)
for i in range(0,n_files,batch_size):
    input_imgs = [centre_crop(Image.open(img_file)).unsqueeze(0) for img_file in img_files[i:min(i+64,n_files)]]
    input_imgs = torch.cat(input_imgs)
    input_imgs = V(input_imgs)
    _ = model.forward(input_imgs)
    cur_batch = i//64 + 1
    if cur_batch % 10 == 0: print('{}/{} batches done'.format(cur_batch, num_batches))

features = np.concatenate(features_blobs)
with open('{}/features_places.pkl'.format(img_dir), 'wb') as f:
    pickle.dump({'img_map': img_map, 'features': features}, f)
