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
from skimage.morphology import dilation, thin, binary_erosion
from scipy.ndimage import measurements
from tqdm import tqdm
import math

DATA_DIR = './data/voc/VOC2012/'

x_dir = os.path.join(DATA_DIR, 'JPEGImages')
y_dir = os.path.join(DATA_DIR, 'SegmentationObject')
cls_dir = os.path.join(DATA_DIR, 'SegmentationClass')

train_ids = os.path.join(DATA_DIR, 'ImageSets/Segmentation/train.txt')
valid_ids = os.path.join(DATA_DIR, 'ImageSets/Segmentation/val.txt')

# some useful constants
dim = (256, 256) # resize all images to dim=(256, 256)
background_centerness_const = 0 # gt val to indicate the pixel is a background


# ========== visualization helper ==========
def viz_centerness(y, id, folder_name):
    """Visualize centerness based on centerness scores

    Args:
        y: array of size [(1, H, W)], with the first dimension being centerness score
    Return:
        visualization plot
    
    """
    # read in ground truth, display as background in plots
    im = plt.imread(os.path.join(DATA_DIR, 'SegmentationObject', id[:-3]+'.png'))
    im = np.where(im == 0, 255, im) # convert the background pixels to white (for visualization)
    res_im = resize(im, dim)
    mask = np.array(Image.open(os.path.join(DATA_DIR, 'SegmentationObject', id[:-3]+'.png')).resize(dim, resample=Image.NEAREST), dtype=int)
    mask = np.where(mask == 255, 0, mask)
    y = y.squeeze()
    # y = np.where(mask == 0, 0, y)
    
    # save centerness image
    plt.figure(figsize = (20, 20))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y[:,:,None], cmap='gray_r', interpolation='nearest')
    plt.imshow(res_im, interpolation='nearest', alpha=0.4)
    plt.savefig('./' + folder_name + '/' + id + '.png', dpi=300, bbox_inches='tight')
    plt.close()

    # save modified mask
    modified_mask = np.zeros(mask.shape)
    for label in np.unique(mask):
        if label == 0:
            continue
        # perform morphological thinning and dilation
        cur_mask = np.array(mask == label, dtype=int)
        cur_mask = binary_erosion(cur_mask)
        cur_mask = dilation(cur_mask)
        modified_mask = modified_mask + cur_mask
    
    # save modified mask
    plt.figure(figsize = (20, 20))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(modified_mask[:,:,None], cmap='gray_r', interpolation='nearest')
    plt.imshow(res_im, interpolation='nearest', alpha=0.4)
    plt.savefig('./' + folder_name + '/' + id[:-3] + '.png', dpi=300, bbox_inches='tight')
    plt.close()


# ========== data loader ==========
'''
Writing helper class for data extraction, tranformation and preprocessing
https://pytorch.org/docs/stable/data
'''
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


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
        assert image.shape[:-1] == mask.shape

        # concat xy position info on image (RGBXY)
        # xx, yy = np.arange(image.shape[0]), np.arange(image.shape[1])
        # grid_x, grid_y = np.meshgrid(xx, yy, indexing='ij')
        # image = np.concatenate((image, grid_x[:,:,None], grid_y[:,:,None]), axis=-1).astype('float')

        mask = np.where(mask == 255, 0, mask) # convert the void pixels to background
        # print(np.unique(mask))
        centerness = np.zeros(mask.shape)
        weight = np.ones(mask.shape)
        eps = 0.0001

        for label in np.unique(mask):
            if label == 0:
                continue
            # perform morphological thinning and dilation
            cur_mask = (mask == label)
            cur_mask = binary_erosion(cur_mask)
            cur_mask = dilation(cur_mask)
            if np.count_nonzero(cur_mask) == 0: # avoid empty object after erosion
                cur_mask = (mask == label)
            # compute the center coordinate by average all coordinates in the current object
            xx, yy = np.arange(mask.shape[0]), np.arange(mask.shape[1])
            grid_x, grid_y = np.meshgrid(xx, yy, indexing='ij')
            grid_x = np.where(cur_mask == 1, grid_x, 0)
            grid_y = np.where(cur_mask == 1, grid_y, 0)
            center_x, center_y = np.sum(grid_x) / np.count_nonzero(grid_x), np.sum(grid_y) / np.count_nonzero(grid_y)
            # assign center-ness score (between 0 and 1) to each pixel of 'label' based on distance to center
            # const / distance
            x_sqr_diff = np.square(np.absolute(grid_x - center_x))
            y_sqr_diff = np.square(np.absolute(grid_y - center_y))
            dist = np.sqrt(x_sqr_diff + y_sqr_diff) + eps # prevent division by 0
            scores = np.minimum(1, np.divide(10, dist))
            scores = np.where(mask == label, scores, 0)
            # uniformly spread more weight to smaller objects (at least weight of a radius-10 circle)
            if np.sum(scores) >= 314:
                centerness = np.where(mask == label, scores, centerness)
            else:
                inc = (314 - np.sum(scores)) / np.sum(mask == label)
                centerness = np.where(mask == label, scores + inc, centerness)
            # update weight on pixels of current instance, used in loss computation
            # weight = np.where(mask == label, 10000 / np.sum(mask == label), weight)
            # weight = np.where(mask == label, (100 / np.sum(mask == label))**2, weight)
            weight = np.where(mask == label, 100 / math.sqrt(np.sum(mask == label)), weight)
        
        # sanity check
        assert np.all(centerness >= np.zeros(centerness.shape))
        assert np.all(centerness[mask == 0] == 0)
        assert np.all(centerness[mask != 0] > 0)
        centerness = centerness[:,:,None].astype('float')
        weight = weight[:,:,None].astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=centerness)
            image, centerness = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=centerness)
            image, centerness = sample['image'], sample['mask']
            
        return image, centerness, weight.transpose(2, 0, 1).astype('float32')
    
    def __len__(self):
        return len(self.ids)


# look at the data we have
# dataset = Dataset(x_dir, y_dir, train_ids)
# image, centerness = dataset[14] # get some sample
# with open(train_ids, 'r') as f:
#     ids = [x.strip() for x in f.readlines()]
# print(ids[14])
# viz_centerness(centerness, 'plot2', '.')


# ========== preprocess image ==========
import albumentations as albu

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
ACTIVATION = 'sigmoid' # for centerness head
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=1, # centerness (1 output channels)
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
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=6)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.Weighted_MSELoss(ignore_val=background_centerness_const) # smp.utils.losses.L1Loss(ignore_val=background_centerness_const) # total loss computed on object pixels
metrics = [
    smp.utils.metrics.L1_centerness_object(), # per pixel score
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

# # train model for 20 epochs

# min_score = 100000
# '''plot the training and validation losses
#    sanity check if they are decreasing over epochs
# '''
# train_loss, valid_loss = [], []
# l1_centerness = []
# epochs = range(0,20)

# for i in range(0,20):
    
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(valid_loader)
#     train_loss.append(train_logs['weighted_mse_loss'])
#     valid_loss.append(valid_logs['weighted_mse_loss'])
#     l1_centerness.append(valid_logs['L1_centerness_object'])
    
#     # do something (save model, change lr, etc.)
#     if valid_logs['L1_centerness_object'] < min_score:
#         min_score = valid_logs['L1_centerness_object']
#         torch.save(model, './best_model_centerness_weightedL2_invsqrt.pth')
#         print('Model saved!')

# # save the plots of training and validation losses
# plt.plot(epochs, train_loss, label='training_loss', color='red')
# plt.plot(epochs, valid_loss, label='validation_loss', color='blue')
# plt.title('loss visualization', fontsize=12)
# plt.legend(loc='upper left')
# plt.xlabel('epochs', fontsize=12)
# plt.ylabel('loss', fontsize=12)
# plt.savefig('./loss.png', dpi=300, bbox_inches='tight')
# plt.close()


# ========== visualize predictions ==========
# load best saved checkpoint
best_model = torch.load('./best_model_centerness_weightedL2_invsqrt.pth')\

# with open(valid_ids, 'r') as f:
#     ids = [x.strip() for x in f.readlines()]

# for idx in range(10):
#     i = idx # np.random.choice(len(valid_dataset))
#     print(ids[i])
#     image, gt_mask, weight = valid_dataset[i]
#     gt_mask = gt_mask.squeeze()
    
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     centerness = best_model.predict(x_tensor)
#     centerness = (centerness.squeeze().cpu().numpy().round())
#     # print(np.sum(centerness))
#     # print(np.amax(centerness))
#     # print(np.amin(centerness))

#     viz_centerness(gt_mask, str(ids[i]) + '_gt', 'centerness_val_viz')
#     viz_centerness(centerness, str(ids[i]) + '_pr', 'centerness_val_viz')

# with open(train_ids, 'r') as f:
#     ids = [x.strip() for x in f.readlines()]

# for idx in [24, 121, 135, 431, 837, 871, 966, 1118, 1294, 1374]:
#     i = idx # np.random.choice(len(train_dataset))
#     print(ids[i])
#     image, gt_mask, weight = train_dataset[i]
#     gt_mask = gt_mask.squeeze()
    
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     centerness = best_model.predict(x_tensor)
#     centerness = (centerness.squeeze().cpu().numpy().round())

#     viz_centerness(gt_mask, str(ids[i]) + '_gt', 'centerness_train_viz')
#     viz_centerness(centerness, str(ids[i]) + '_pr', 'centerness_train_viz')


# ========== other evaluations ==========
# percentage of correct centers detected (on validation set)
with open(valid_ids, 'r') as f:
    val_ids_arr = [x.strip() for x in f.readlines()]
masks_fps = [os.path.join(y_dir, image_id + '.png') for image_id in val_ids_arr]
semantic_fps = [os.path.join(cls_dir, image_id + '.png') for image_id in val_ids_arr]

total_count = 0
evals = dict()
for i in tqdm(range(len(val_ids_arr))):
    image, gt_mask, weight = valid_dataset[i]
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    centerness = best_model.predict(x_tensor)
    # 1) threshold centerness scores
    centerness = (centerness.squeeze().cpu().numpy().round())
    # 2) compute connected components
    labeled_array, num_features = measurements.label(centerness)
    
    obj_mask = np.array(Image.open(masks_fps[i]).resize(dim, resample=Image.NEAREST), dtype=int)
    cls_mask = np.array(Image.open(semantic_fps[i]).resize(dim, resample=Image.NEAREST), dtype=int)
    image = image.transpose(1, 2, 0)
    assert image.shape[:-1] == obj_mask.shape
    assert image.shape[:-1] == cls_mask.shape

    obj_mask = np.where(obj_mask == 255, 0, obj_mask) # convert the void pixels to background
    cls_mask = np.where(cls_mask == 255, 0, cls_mask)

    # 3) assign connected component to class it has the largest overlap with
    for component in np.unique(labeled_array):
        if component == 0: continue
        largest_overlap = 0
        cls_belong = 0
        for cur_cls in np.unique(cls_mask):
            if cur_cls == 0: continue
            overlap = np.count_nonzero(np.logical_and(labeled_array == component, cls_mask == cur_cls))
            if overlap > largest_overlap:
                largest_overlap = overlap
                cls_belong = cur_cls
        pos = np.logical_and(labeled_array == component, cls_mask == cls_belong)
        labeled_array[labeled_array == component] = 0
        labeled_array[pos] = component

    # 4) compute how many connected components fall inside each instance
    for label in np.unique(obj_mask):
        if label == 0:
            continue
        cur_centerness = np.where(obj_mask == label, labeled_array, 0)
        num_components = np.unique(cur_centerness)
        num_components = len(num_components[num_components != 0])
        if num_components in evals:
            evals[num_components] += 1
        else:
            evals[num_components] = 1
        total_count += 1

# 5) compute precentages of connected components per instance
print("====================")
print("total number of instances in val set: " + str(total_count))
for num_components in evals:
    percentage = evals[num_components] / total_count
    print(str(num_components) + ' connected component(s): ' + str(percentage))