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
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.transform import resize

DATA_DIR = './data/voc/VOC2012/'

x_dir = os.path.join(DATA_DIR, 'JPEGImages')
y_dir = os.path.join(DATA_DIR, 'SegmentationObject')

train_ids = os.path.join(DATA_DIR, 'ImageSets/Segmentation/train.txt')
valid_ids = os.path.join(DATA_DIR, 'ImageSets/Segmentation/val.txt')

# some useful constants
dim = (256, 256) # resize all images to dim=(256, 256)
background_dist_const = 300 # gt val to indicate the pixel is a background

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
    plt.figure(figsize = (20, 20))
    l_viz, t_viz, r_viz, b_viz = np.zeros((y.shape[1], y.shape[2])), np.zeros((y.shape[1], y.shape[2])), np.zeros((y.shape[1], y.shape[2])), np.zeros((y.shape[1], y.shape[2]))
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):
            l_viz[i,max(0, int(j - y[0,i,j]))] += 1 if y[0,i,j] < 256 else 0
            t_viz[max(0, int(i - y[1,i,j])),j] += 1 if y[1,i,j] < 256 else 0
            r_viz[i,min(y.shape[2]-1, int(j + y[2,i,j]))] += 1 if y[2,i,j] < 256 else 0
            b_viz[min(y.shape[1]-1, int(i + y[3,i,j])),j] += 1 if y[3,i,j] < 256 else 0
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

        # extract l, t, r, b distance info from mask
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
            
        return image.transpose(2, 0, 1).astype('float32'), mask.transpose(2, 0, 1).astype('float32')
    
    def __len__(self):
        return len(self.ids)


# look at the data we have
# dataset = Dataset(x_dir, y_dir, train_ids)
# image, mask = dataset[14] # get some sample
# with open(train_ids, 'r') as f:
#     ids = [x.strip() for x in f.readlines()]
# print(ids[14])
# viz_boundary(mask, 'plot1', '.')


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
)

valid_dataset = Dataset(
    x_dir, 
    y_dir, 
    valid_ids,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.L1Loss()
metrics = [
    smp.utils.metrics.L1Score_left_object(),
    smp.utils.metrics.L1Score_top_object(),
    smp.utils.metrics.L1Score_right_object(),
    smp.utils.metrics.L1Score_bot_object(),
    smp.utils.metrics.L1Score_left_background(),
    smp.utils.metrics.L1Score_top_background(),
    smp.utils.metrics.L1Score_right_background(),
    smp.utils.metrics.L1Score_bot_background(),
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

# min_score = 100000000
# '''plot the training and validation losses
#    sanity check if they are decreasing over epochs
# '''
# train_loss = []
# valid_loss = []
# epochs = range(0, 40)

# for i in range(0, 40):
    
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(valid_loader)
#     train_loss.append(train_logs['l1_loss'])
#     valid_loss.append(valid_logs['l1_loss'])
    
#     # do something (save model, change lr, etc.)
#     if valid_logs['l1_loss'] < min_score:
#         min_score = valid_logs['l1_loss']
#         torch.save(model, './best_model_new.pth')
#         print('Model saved!')
        
#     if i == 25:
#         optimizer.param_groups[0]['lr'] = 1e-5
#         print('Decrease decoder learning rate to 1e-5!')

# # save the plots of training and validation losses
# plt.plot(epochs, train_loss, label='training_loss', color='red')
# plt.plot(epochs, valid_loss, label='validation_loss', color='blue')
# plt.title('loss visualization', fontsize=12)
# plt.legend(loc='upper left')
# plt.xlabel('epochs', fontsize=12)
# plt.ylabel('loss', fontsize=12)
# plt.savefig('./loss.png', dpi=300, bbox_inches='tight')
# plt.close()

# l1_left_object = [29.56, 28.45, 27.11, 27.6, 27.64, 26.42, 29.37, 28.96, 30.87, 32.0, 34.72, 36.65, 35.9, 38.46, 40.62, 42.8, 45.59, 48.85, 51.02, 56.37, 57.88, 57.86, 61.91, 63.17, 71.96, 67.26, 71.92, 72.1, 73.13, 70.36, 70.52, 74.31, 70.57, 73.48, 73.84, 71.48, 75.07, 74.72, 73.3, 78.85]

# l1_top_object = [31.8, 30.37, 28.91, 29.09, 28.75, 27.79, 30.03, 29.22, 30.73, 31.99, 34.46, 35.83, 35.07, 37.46, 38.71, 42.11, 44.18, 46.91, 49.66, 53.83, 54.16, 54.84, 59.09, 60.33, 68.57, 64.22, 69.16, 69.34, 70.17, 68.08, 68.11, 71.91, 68.1, 71.06, 71.79, 69.23, 72.69, 72.23, 70.89, 76.52]

# l1_right_object = [31.06, 29.99, 28.87, 28.47, 28.13, 27.72, 29.81, 29.15, 30.69, 32.37, 33.94, 35.45, 35.36, 37.24, 38.82, 41.77, 44.38, 46.01, 49.26, 54.25, 54.31, 56.19, 59.17, 61.65, 69.83, 66.23, 70.9, 70.88, 71.78, 69.43, 69.36, 73.37, 69.63, 72.51, 72.84, 70.67, 74.03, 73.67, 72.37, 77.92]

# l1_bot_object = [35.02, 33.02, 31.51, 30.56, 30.0, 29.09, 30.5, 29.9, 31.03, 31.83, 32.92, 34.99, 34.76, 36.22, 38.14, 39.94, 43.02, 44.57, 46.8, 52.06, 51.57, 52.95, 56.16, 58.21, 65.79, 63.22, 67.72, 67.79, 68.82, 66.33, 66.47, 70.09, 66.71, 69.46, 70.06, 67.82, 71.29, 70.9, 69.8, 74.88]

# l1_left_background = [272.4, 265.6, 258.7, 250.9, 242.2, 234.4, 224.5, 215.3, 205.2, 194.8, 184.6, 173.1, 162.6, 151.3, 138.8, 126.3, 115.6, 102.5, 91.76, 79.25, 71.34, 63.46, 55.54, 48.27, 40.63, 40.69, 36.28, 35.47, 34.39, 35.63, 35.38, 32.43, 34.04, 32.75, 32.48, 33.03, 31.27, 30.56, 31.47, 29.44]

# l1_top_background = [278.1, 272.0, 265.6, 258.6, 251.1, 243.4, 234.3, 225.5, 216.2, 206.2, 196.2, 185.8, 175.2, 163.3, 151.7, 139.4, 127.7, 114.9, 103.2, 90.82, 81.21, 71.09, 61.94, 54.58, 45.89, 42.68, 39.26, 38.6, 37.32, 37.86, 37.62, 35.01, 36.7, 35.01, 34.62, 34.91, 33.27, 32.54, 32.86, 30.99]

# l1_right_background = [278.4, 272.7, 266.4, 259.2, 251.8, 244.5, 235.0, 226.5, 216.9, 207.1, 197.1, 187.0, 176.3, 164.8, 152.9, 140.3, 128.8, 116.0, 104.3, 92.03, 82.04, 72.0, 63.2, 55.41, 46.58, 43.82, 40.32, 39.59, 38.31, 39.0, 38.74, 35.92, 37.49, 35.89, 35.63, 35.94, 34.34, 33.47, 33.89, 32.03]

# l1_bot_background = [288.5, 282.4, 276.6, 270.0, 262.9, 255.9, 247.4, 238.9, 229.9, 220.8, 211.2, 201.2, 191.0, 179.7, 168.5, 156.3, 145.4, 132.6, 120.7, 108.7, 97.68, 85.45, 76.3, 68.6, 57.82, 49.72, 48.24, 47.75, 46.05, 46.06, 46.0, 43.56, 45.08, 42.96, 42.57, 42.34, 40.95, 39.77, 39.43, 37.64]

# # save the plots of l1-metrics on validation set
# plt.plot(epochs, l1_left_object, label='l1_left_object', color='red')
# plt.plot(epochs, l1_top_object, label='l1_top_object', color='red')
# plt.plot(epochs, l1_right_object, label='l1_right_object', color='red')
# plt.plot(epochs, l1_bot_object, label='l1_bot_object', color='red')
# plt.plot(epochs, l1_left_background, label='l1_left_background', color='blue')
# plt.plot(epochs, l1_top_background, label='l1_top_background', color='blue')
# plt.plot(epochs, l1_right_background, label='l1_right_background', color='blue')
# plt.plot(epochs, l1_bot_background, label='l1_bot_background', color='blue')
# plt.title('metrics visualization', fontsize=12)
# plt.legend(loc='upper right')
# plt.xlabel('epochs', fontsize=12)
# plt.ylabel('metrics', fontsize=12)
# plt.savefig('./metrics.png', dpi=300, bbox_inches='tight')
# plt.close()


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

    # print('gt min :' + str(np.amin(gt_mask)) + ' max : ' + str(np.amax(gt_mask)))
    # print('pred min :' + str(np.amin(pr_mask)) + ' max : ' + str(np.amax(pr_mask)))
    # print(pr_mask)

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
