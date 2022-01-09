# this script is based on ./examples/cars segmentation (camvid).ipynb
# ========== loading data ==========
'''
For this example, we will use PASCAL2012 dataset. It is a set of:
- train images + instance segmentation masks
- validation images + instance segmentation masks
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Will use only the first GPU device

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.transform import resize
import albumentations as albu

DATA_DIR = './data/voc/VOC2012/'

x_dir = os.path.join(DATA_DIR, 'JPEGImages')
y_dir = os.path.join(DATA_DIR, 'SegmentationObject')

train_ids = os.path.join(DATA_DIR, 'ImageSets/Segmentation/train.txt')
valid_ids = os.path.join(DATA_DIR, 'ImageSets/Segmentation/val.txt')

# some useful constants
dim = (256, 256) # resize all images to dim=(256, 256)
background_dist_const = 256 # gt val to indicate the pixel is a background

# ========== data loader ==========
'''
Writing helper class for data extraction, tranformation and preprocessing
https://pytorch.org/docs/stable/data
'''
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


def viz_boundary(y, id, folder_name):
    """Visualize boundaries based on (l, t, r, b)

    Args:
        y: array of size [(4, H, W)], with the first dimension being (l, t, r, b) distances
    Return:
        4 plots visualization of the 'counts' of each pixel, where others think of it as a boundary
    
    """
    # read in ground truth, display as background in plots
    im = plt.imread(os.path.join(DATA_DIR, 'SegmentationObject', id[:-3]+'.png'))
    im = np.where(im == 0, 255, im) # convert the background pixels to white (for visualization)
    res_im = resize(im, dim)

    mask = np.array(Image.open(os.path.join(DATA_DIR, 'SegmentationObject', id[:-3]+'.png')).resize(dim, resample=Image.NEAREST), dtype=int)
    mask = np.where(mask == 255, 0, mask) # convert the void pixels to background

    plt.figure(figsize = (20, 20))
    l_viz, t_viz, r_viz, b_viz = np.zeros((y.shape[1], y.shape[2])), np.zeros((y.shape[1], y.shape[2])), np.zeros((y.shape[1], y.shape[2])), np.zeros((y.shape[1], y.shape[2]))
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):
            l_viz[i,max(0, int(j - y[0,i,j]))] += 1 if mask[i,j] != 0 else 0
            t_viz[max(0, int(i - y[1,i,j])),j] += 1 if mask[i,j] != 0 else 0
            r_viz[i,min(y.shape[2]-1, int(j + y[2,i,j]))] += 1 if mask[i,j] != 0 else 0
            b_viz[min(y.shape[1]-1, int(i + y[3,i,j])),j] += 1 if mask[i,j] != 0 else 0
    plt.subplot(221, title='left boundary visualization')
    plt.imshow(res_im, interpolation='nearest', alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(l_viz, cmap='gray_r', interpolation='nearest', norm=LogNorm(vmin=0, vmax=np.amax(l_viz)))
    plt.subplot(222, title='top boundary visualization')
    plt.imshow(res_im, interpolation='nearest', alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(t_viz, cmap='gray_r', interpolation='nearest', norm=LogNorm(vmin=0, vmax=np.amax(t_viz)))
    plt.subplot(223, title='right boundary visualization')
    plt.imshow(res_im, interpolation='nearest', alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(r_viz, cmap='gray_r', interpolation='nearest', norm=LogNorm(vmin=0, vmax=np.amax(r_viz)))
    plt.subplot(224, title='bottom boundary visualization')
    plt.imshow(res_im, interpolation='nearest', alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(b_viz, cmap='gray_r', interpolation='nearest', norm=LogNorm(vmin=0, vmax=np.amax(b_viz)))
    plt.savefig('./' + folder_name + '/' + id + '.png', dpi=300, bbox_inches='tight')
    plt.close()


class Dataset(BaseDataset):
    """PASCAL Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        img_ids (str): path to the file containing image ids
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            img_ids,
            augmentation=None, 
            preprocessing=None,
    ):
        with open(img_ids, 'r') as f:
            self.ids = [x.strip() for x in f.readlines()]
        print(img_ids + ': ' + str(len(self.ids)) + ' Images')
        self.images_fps = [os.path.join(images_dir, image_id + '.jpg') for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id + '.png') for image_id in self.ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = np.array(Image.open(self.images_fps[i]).resize(dim, resample=Image.NEAREST))
        mask = np.array(Image.open(self.masks_fps[i]).resize(dim, resample=Image.NEAREST), dtype=int)

        mask = np.where(mask == 255, 0, mask) # convert the void pixels to background
        # print(np.unique(mask))
        l, t, r, b = np.zeros(mask.shape), np.zeros(mask.shape), np.zeros(mask.shape), np.zeros(mask.shape)

        # get l, t, r, b distance info from mask
        # horizontal scanline
        for i in range(mask.shape[0]):
            cur_label = mask[i,0]
            last_pos = 0
            for j in range(1, mask.shape[1]):
                if mask[i,j] != cur_label:
                    cur_label = mask[i,j]
                    last_pos = j
                # update l
                l[i,j] = j - last_pos
                # update r
                r[i,last_pos:j] += 1    
        # vertical scanline
        for j in range(mask.shape[1]):
            cur_label = mask[0,j]
            last_pos = 0
            for i in range(1, mask.shape[0]):
                if mask[i,j] != cur_label:
                    cur_label = mask[i,j]
                    last_pos = i
                # update t
                t[i,j] = i - last_pos
                # update b
                b[last_pos:i,j] += 1
        l[mask == 0] = background_dist_const
        t[mask == 0] = background_dist_const
        r[mask == 0] = background_dist_const
        b[mask == 0] = background_dist_const
        mask = np.stack((l, t, r, b), axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
    
    def __len__(self):
        return len(self.ids)


# look at the data we have
# dataset = Dataset(x_dir, y_dir, train_ids)
# image, mask = dataset[14] # get some sample
# with open(train_ids, 'r') as f:
#     ids = [x.strip() for x in f.readlines()]
# print(ids[14])
# viz_boundary(mask, 'plot1', '.')


# ========== Preprocessing ==========

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# ========== create model and train ==========
import torch
import segmentation_models_pytorch as smp

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
pascal_class = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] # 20 classes (excluding background)
ACTIVATION = 'relu' # for 4 regression heads
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=4, # l, t, r, b distances (4 output channels)
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_dir, 
    y_dir, 
    train_ids,
    preprocessing=get_preprocessing(preprocessing_fn),
)

valid_dataset = Dataset(
    x_dir, 
    y_dir, 
    valid_ids,
    preprocessing=get_preprocessing(preprocessing_fn),
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.L1Loss(ignore_val=background_dist_const)
metrics = [
    smp.utils.metrics.L1Score_left_object(),
    smp.utils.metrics.L1Score_top_object(),
    smp.utils.metrics.L1Score_right_object(),
    smp.utils.metrics.L1Score_bot_object(),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# train model for 40 epochs

min_score = 300
'''plot the training and validation losses
   sanity check if they are decreasing over epochs
'''
train_loss = []
valid_loss = []
l1_left_object, l1_top_object, l1_right_object, l1_bot_object = [], [], [], []
epochs = range(0, 40)

for i in range(0, 40):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    train_loss.append(train_logs['l1_loss'])
    valid_loss.append(valid_logs['l1_loss'])
    l1_left_object.append(valid_logs['L1_left_object'])
    l1_top_object.append(valid_logs['L1_top_object'])
    l1_right_object.append(valid_logs['L1_right_object'])
    l1_bot_object.append(valid_logs['L1_bot_object'])
    
    # do something (save model, change lr, etc.)
    if valid_logs['l1_loss'] < min_score:
        min_score = valid_logs['l1_loss']
        torch.save(model, './best_model_new.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# save the plots of training and validation losses
plt.plot(epochs, train_loss, label='training_loss', color='red')
plt.plot(epochs, valid_loss, label='validation_loss', color='blue')
plt.title('loss visualization', fontsize=12)
plt.legend(loc='upper left')
plt.xlabel('epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.savefig('./loss.png', dpi=300, bbox_inches='tight')
plt.close()

# save the plots of l1-metrics on validation set
plt.plot(epochs, l1_left_object, label='l1_left_object', color='blue')
plt.plot(epochs, l1_top_object, label='l1_top_object', color='blue')
plt.plot(epochs, l1_right_object, label='l1_right_object', color='blue')
plt.plot(epochs, l1_bot_object, label='l1_bot_object', color='blue')
plt.title('metrics visualization', fontsize=12)
plt.legend(loc='upper right')
plt.xlabel('epochs', fontsize=12)
plt.ylabel('metrics', fontsize=12)
plt.savefig('./metrics.png', dpi=300, bbox_inches='tight')
plt.close()


# ========== visualize predictions ==========
# load best saved checkpoint
best_model = torch.load('./best_model_new.pth')\

with open(valid_ids, 'r') as f:
    ids = [x.strip() for x in f.readlines()]

for idx in range(10):
    i = idx # np.random.choice(len(valid_dataset))
    print(ids[i])
    image, gt_mask = valid_dataset[i]
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    viz_boundary(gt_mask, str(ids[i]) + '_gt', 'val_viz')
    viz_boundary(pr_mask, str(ids[i]) + '_pr', 'val_viz')

with open(train_ids, 'r') as f:
    ids = [x.strip() for x in f.readlines()]

for idx in [24, 121, 135, 431, 837, 871, 966, 1118, 1294, 1374]:
    i = idx # np.random.choice(len(train_dataset))
    print(ids[i])
    image, gt_mask = train_dataset[i]
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    viz_boundary(gt_mask, str(ids[i]) + '_gt', 'train_viz')
    viz_boundary(pr_mask, str(ids[i]) + '_pr', 'train_viz')
