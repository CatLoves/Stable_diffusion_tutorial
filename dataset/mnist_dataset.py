import glob
import os
import pandas as pd
from PIL import Image
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset

def convert_csv_to_images(csv_path, output_dir):
    """ 
    将 csv 文件转为 0-9/*.jpg 的形式
    Args:
        csv_path: str
        output_dir: str
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path, header=0)

    # 获取标签列和像素值列
    labels = df.iloc[:, 0]
    pixels = df.iloc[:, 1:]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每一行数据
    for i in range(len(df)):
        # 获取标签和像素值
        label = labels.iloc[i]
        pixel_values = pixels.iloc[i].values

        # 将一维像素值数组转为二维数组 (28x28)
        image_array = pixel_values.reshape(28, 28)

        # 创建图像对象
        image = Image.fromarray(image_array.astype('uint8'))

        # 生成保存路径
        save_path = os.path.join(output_dir, f"{label}/{i}.jpg")

        # 确保标签子目录存在
        os.makedirs(os.path.join(output_dir, str(label)), exist_ok=True)

        # 保存图像
        image.save(save_path)
        print(f"=> saved to {save_path}")

class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """ 
    
    def __init__(self, split, im_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        
        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        
        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)
        
        # Whether to load images and call vae or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
        
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, d_name, '*.{}'.format('png')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpeg')))
            for fname in fnames:
                ims.append(fname)
                if 'class' in self.condition_types:
                    labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'class' in self.condition_types:
            cond_inputs['class'] = self.labels[index]
        #######################################
        
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.ToTensor()(im)
            
            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
            
if __name__ == "__main__":

    # """ convert test set """
    # csv_path = '/root/bigModelProjects/stable_diffusion_explainingAI/data/mnist_test.csv'
    # save_dir = '/root/bigModelProjects/stable_diffusion_explainingAI/data/mnist/test/images'
    # ds = convert_csv_to_images(csv_path, save_dir)
    
    """ convert train set """
    csv_path = '/data/mnist_train.csv'
    save_dir = '/data/mnist/train/images'
    ds = convert_csv_to_images(csv_path, save_dir)