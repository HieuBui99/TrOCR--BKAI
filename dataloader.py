from fastai.vision.all import *
from imgaug import augmenters as iaa
import imgaug as ia

class BeforeBatchTransform(Transform):
    """
    Resize image before create batch
    """

    def __init__(self, height=32, max_width=32 * 5, keep_ratio=True, min_ratio=5.):
        super(BeforeBatchTransform, self).__init__()
        self.height, self.width = height, max_width
        self.keep_ratio, self.min_ratio = keep_ratio, min_ratio

    def encodes(self, items):
        images, *labels = zip(*items)
        
        height, width = self.height, self.width
#         print(images)
        # tgt_input = []
        max_ratio = self.min_ratio
        for i, image in enumerate(images):
            w, h = image.size
#             h, w = image.shape[1], image.shape[2]
            max_ratio = max(max_ratio, w / h)

        width = int(np.floor(height * max_ratio))

        rs_tfm = Resize(size=(height, width), method=ResizeMethod.Pad, pad_mode=PadMode.Border)
        images = [rs_tfm(image) for image in images]
        return zip(images, *labels)

    
class CreateBatchTransform(Transform):
    """
    Create batch
    """

    def __init__(self):
        super(CreateBatchTransform, self).__init__()
        self.pipeline = Pipeline(funcs=[ToTensor, ])

    def encodes(self, items):
        # images, *labels = zip(*items)
        images, labels = zip(*items)
#         print(type(images[0]))
        # process images
        images = self.pipeline(images)
#         xs = torch.stack(images, dim=0)
        xs = TensorImage(torch.stack(images, dim=0))


        target_weights = []
        tgt_input = []
        max_label_len = max(len(label) for label in labels)
        
        for label in labels:
            label_len = len(label)
            label = label.numpy()
            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)
            one_mask_len = label_len - 1
            
            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len,dtype=np.float32))))
        
        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0

        mask = np.random.random(size=tgt_input.shape) < 0.05
        mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
        tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights)==0
        # # process labels
        ys = {
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask),
        }

        return xs, ys
    
class ImageAugmentation(ItemTransform):
    split_idx = 0
    
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        self.aug =  iaa.Sequential(iaa.SomeOf((1, 5), 
        [
        # blur

        sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)),
                            iaa.MotionBlur(k=3)])),
        
        # color
        sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
        sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)),
        sometimes(iaa.Invert(0.25, per_channel=0.5)),
        sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
        sometimes(iaa.Dropout2d(p=0.5)),
        sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
        sometimes(iaa.Add((-40, 40), per_channel=0.5)),

        sometimes(iaa.JpegCompression(compression=(5, 80))),
        
        # distort
        sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
        sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), 
#                            rotate=(-5, 5), shear=(-5, 5), 
                            order=[0, 1], cval=(0, 255), 
                            mode=ia.ALL)),
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
        sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))])),

    ],
        random_order=True),
    random_order=True)
    
    def encodes(self, x):
        img, label = x
        img = np.array(img)
        img = self.aug.augment_image(img)
#         x = PILImage.create(x)
#         print(img.shape)
        return PILImage.create(img), label