import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Normalization parameters
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = (224, 224)



# train_transform = A.Compose([
#     A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
#     A.HorizontalFlip(),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=25, border_mode=cv2.BORDER_CONSTANT),
#     A.OneOf([
#             A.ImageCompression(),
#             A.MultiplicativeNoise(),
#             A.GaussNoise(),
#             A.ISONoise(),
#     ]),
#     A.OneOf([
#             A.RGBShift(),
#             A.RandomBrightnessContrast(),
#             A.RandomGamma(gamma_limit=(50, 150)),
#             A.HueSaturationValue(),
#             A.ChannelShuffle(),
#             A.CLAHE(),
#     ]),
#     A.OneOf([
#             A.Blur(),
#             A.MedianBlur(),
#             A.MotionBlur(),
#             A.GaussianBlur(),
#             A.GlassBlur()
#     ], p=0.5),
#     A.OneOf([
#         A.OneOf([
#                 A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT),
#                 A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
#         ], p=0.5),
#         A.CoarseDropout(
#                         max_holes=1,
#                         min_holes=1,
#                         min_height=int(IMG_SIZE[1] / 2.56),
#                         min_width=int(IMG_SIZE[0] / 2.56),
#                         max_height=int(IMG_SIZE[1] / 1.28),
#                         max_width=int(IMG_SIZE[0] / 1.28),
#                         fill_value=(0, 0, 0)),
#     ], p=0.5),
#     A.Normalize(mean=MEAN, std=STD),
#     ToTensorV2(),
# ])

# val_transform = A.Compose([
#     A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
#     A.Normalize(mean=MEAN, std=STD),
#     ToTensorV2(),
# ])


def train_transforms(mean=MEAN, std=STD, image_size=IMG_SIZE):
        
        train_transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
                A.OneOf([
                        A.ImageCompression(),
                        A.MultiplicativeNoise(),
                        A.GaussNoise(),
                        A.ISONoise(),
                ]),
                A.OneOf([
                        A.RGBShift(),
                        A.RandomBrightnessContrast(),
                        A.RandomGamma(gamma_limit=(50, 150)),
                        A.HueSaturationValue(),
                        A.ChannelShuffle(),
                        A.CLAHE(),
                ]),
                A.OneOf([
                        A.Blur(),
                        A.MedianBlur(),
                        A.MotionBlur(),
                        A.GaussianBlur(),
                        A.GlassBlur()
                ], p=0.5),
                A.OneOf([
                        A.OneOf([
                                A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT),
                                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
                        ], p=0.5),
                        A.CoarseDropout(
                                        max_holes=1,
                                        min_holes=1,
                                        min_height=int(image_size[1] / 2.56),
                                        min_width=int(image_size[0] / 2.56),
                                        max_height=int(image_size[1] / 1.28),
                                        max_width=int(image_size[0] / 1.28),
                                        fill_value=(0, 0, 0)),
                ], p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
                ])

        return train_transform


def val_transforms(mean=MEAN, std=STD, image_size=IMG_SIZE):

        val_transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
                ])
        
        return val_transform