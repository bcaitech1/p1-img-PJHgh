
from PIL import Image
from torchvision.transforms import *
from Argumentation_Policies import *

class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def set_argumentation(version):
    if version == 'v0':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 512, 384]')
        
    elif version == 'v1':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(224),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 224, 224]')
        
    elif version == 'v2':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v3':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([RandomHorizontalFlip(p=0.5)]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v4':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([RandomVerticalFlip(p=0.5)]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v5':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([RandomRotation(degrees=30)]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v6':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([RandomAffine(40)]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
    
    elif version == 'v7':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([ColorJitter(brightness=(1,1.1))]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v8':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([ColorJitter(contrast=(0.2, 3))]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v9':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([ColorJitter(saturation=(0.1, 3))]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v10':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([ColorJitter(hue=(-0.5, 0.5))]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
    
    elif version == 'v11':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([ColorJitter(0.1, 0.1, 0.1, 0.1)]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v12':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([RandomGrayscale(p=0.1)]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
    
    elif version == 'v13':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                         RandomChoice([AddGaussianNoise()])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
#     elif version == 'v14':
#         argumentation = [Resize((512, 384), Image.BILINEAR),
#                          CenterCrop(384),
#                          ToTensor(),
#                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                          RandomChoice([RandomPerspective(distortion_scale=0.5, p=0.5)])]
#         print(f'argumentation {version} input size : [N, 3, 384, 384]')
    
#     elif version == 'v15':
#         argumentation = [Resize((512, 384), Image.BILINEAR),
#                          CenterCrop(384),
#                          ToTensor(),
#                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                          RandomChoice([GaussianBlur(kernel_size, sigma=(0.1, 2.0))])]
#         print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'v16':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                         RandomChoice([RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
    
    elif version == 'v17':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomHorizontalFlip(),
                         ImageNetPolicy(),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
    
    elif version == 'version0':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomChoice([RandomHorizontalFlip(p=0.5),
                                       RandomVerticalFlip(p=0.5),
                                       RandomRotation(30)]),
                         RandomChoice([ColorJitter(brightness=(1,1.1)),
                                       ColorJitter(contrast=(0.2, 3)),
                                       ColorJitter(saturation=(0.1, 3)),
                                       ColorJitter(hue=(-0.5, 0.5)),
                                       ColorJitter(0.1, 0.1, 0.1, 0.1),
                                       RandomGrayscale(p=0.1)]),
                         ToTensor(),
                         Normalize(mean=[0.548, 0.504, 0.479], std=[0.237, 0.247, 0.246]),
                         RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
    
    elif version == 'version1':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(384),
                         RandomHorizontalFlip(),
                         ImageNetPolicy(),
                         ToTensor(),
                         Normalize(mean=[0.548, 0.504, 0.479], std=[0.237, 0.247, 0.246])]
        print(f'argumentation {version} input size : [N, 3, 384, 384]')
        
    elif version == 'version2':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(304),
                         RandomChoice([RandomHorizontalFlip(p=0.5),
                                       RandomVerticalFlip(p=0.5),
                                       RandomRotation(30)]),
                         RandomChoice([ColorJitter(brightness=(1,1.1)),
                                       ColorJitter(contrast=(0.2, 3)),
                                       ColorJitter(saturation=(0.1, 3)),
                                       ColorJitter(hue=(-0.5, 0.5)),
                                       ColorJitter(0.1, 0.1, 0.1, 0.1),
                                       RandomGrayscale(p=0.1)]),
                         ToTensor(),
                         Normalize(mean=[0.548, 0.504, 0.479], std=[0.237, 0.247, 0.246]),
                         RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)]
        print(f'argumentation {version} input size : [N, 3, 304, 304]')
        
    elif version == 'version3':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         RandomChoice([CenterCrop(304),
                                       RandomCrop(304)]),
                         RandomChoice([RandomRotation(30),
                                       RandomAffine(40)]),
                         RandomChoice([ColorJitter(brightness=(1,1.1)),
                                       ColorJitter(contrast=(0.2, 3)),
                                       ColorJitter(saturation=(0.1, 3)),
                                       ColorJitter(hue=(-0.5, 0.5))]),
                         ToTensor(),
                         Normalize(mean=[0.548, 0.504, 0.479], std=[0.237, 0.247, 0.246])]
        print(f'argumentation {version} input size : [N, 3, 304, 304]')
        
    elif version == 'version4':
        argumentation = [Resize((512, 384), Image.BILINEAR),
                         CenterCrop(304),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        print(f'argumentation {version} input size : [N, 3, 304, 304]')
    
    else:
        raise NameError(f'!!!!! argumentation version ERROR : {version} !!!!!')
    return argumentation

