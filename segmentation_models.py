import torch
import torch.nn as nn
from torchvision.transforms.functional import crop
from torchsummary import summary



class FullyConvSegNet(nn.Module):
    '''
    Implementation of Fully-Convolutional image segmentation network. 
    '''
    def __init__(self, conv_layers_num: int, img_size: tuple, segment_classes_num: int, 
                 kernel_size = 3, conv_dim = 2, include_activation = True, include_batchnorm = True):
        super().__init__()
        self.model = nn.Sequential()
        self.conv_layers_num = conv_layers_num
        self.img_size = img_size
        self.segment_classes_num = segment_classes_num
        self.conv_dim = conv_dim
        self.kernel_size = kernel_size
        self.include_activation = include_activation
        self.include_batchnorm = include_batchnorm
        self.act_func = nn.ReLU() if self.include_activation else None
        self.build_model()


    def build_model(self):
        '''
        Method builds a model: sequential(conv_1, conv_2, ... , conv_conv_layers_num)
        '''
        for i in range(self.conv_layers_num):
            out_channels = self.img_size[0] if i != self.conv_layers_num - 1 else self.segment_classes_num 
            if self.conv_dim == 2:
                conv_layer = nn.Conv2d(self.img_size[0], out_channels, self.kernel_size, padding = 'same')
                batchnorm = nn.BatchNorm2d(conv_layer.out_channels) if self.include_batchnorm else None

            elif self.conv_dim == 3:    
                conv_layer = nn.Conv3d(self.img_size[0], out_channels, self.kernel_size, padding = 'same')
                batchnorm = nn.BatchNorm3d(conv_layer.out_channels) if self.include_batchnorm else None

            else:
                conv_layer = nn.Conv1d(self.img_size[0], out_channels, self.kernel_size, padding = 'same')
                batchnorm = nn.BatchNorm1d(conv_layer.out_channels) if self.include_batchnorm else None
            

            self.model.append(conv_layer)
            
            if self.include_activation and i != self.conv_layers_num - 1:
                self.model.append(self.act_func)
            
            if self.include_batchnorm and i != self.conv_layers_num - 1:
                self.model.append(batchnorm)
            


    def forward(self, x):
        x = x.view(-1, *x.shape)
        return self.model(x)
    


class UNet(nn.Module):
    '''
    U-Net model implementation 
    '''
    def __init__(self, input_img_size, segment_class_num):
        super().__init__()
        self.input_img_size = input_img_size
        self.segment_class_num = segment_class_num
        self.calculate_image_sizes()
        self.crop_img_sizes = self.img_sizes_dict['expansive_path'][::3][::-1]
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = self.input_img_size[0], out_channels = 64, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3)
        self.conv8 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3)
        self.bottleneck_conv1 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3)
        self.bottleneck_conv2 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3)
        self.upconv1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        self.upconv2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        self.upconv3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        self.upconv4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        self.conv10 = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3)
        self.conv11 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3)
        self.conv12 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3)
        self.conv13 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3)
        self.conv14 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3)
        self.conv15 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.conv16 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.conv17 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3)
        self.conv18 = nn.Conv2d(in_channels = 64, out_channels = self.segment_class_num, kernel_size = 3, padding = 'same')
        self.concat = torch.cat
        self.pool = nn.MaxPool2d(kernel_size = 2)


    def central_crop(self, image, cropped_img_size):
        '''
        Method for cropping the central square of the given image
        '''
        if isinstance(cropped_img_size, int):
            height, width = cropped_img_size, cropped_img_size
        else:
            height, width = cropped_img_size
        
        
        img_height, img_width = image.shape[1:]
        top, left = (img_height - height) // 2, (img_width - width) // 2
        cropped_img = crop(image, top = top, left = left, height = height, width = width)
        return cropped_img


    def calculate_image_sizes(self):
        '''
        Method calculating image sizes through the all network
        '''
        img_size = self.input_img_size[1]
        self.img_sizes_dict = {'contracting_path': [], 'expansive_path': [], 'bottleneck': []}
        self.img_sizes_dict['contracting_path'].append(img_size)
        for _ in range(4):
            conv1_img_size = img_size - 2
            conv2_img_size = conv1_img_size - 2
            polled_img_size = conv2_img_size // 2
            sizes_list = [conv1_img_size, conv2_img_size, polled_img_size]
            self.img_sizes_dict['contracting_path'] += sizes_list
            img_size = polled_img_size


        for _ in range(2):
            img_size -= 2
            self.img_sizes_dict['bottleneck'].append(img_size)
        

        for _ in range(4):
            upconv_size = img_size * 2
            conv1_img_size = upconv_size - 2
            conv2_img_size = conv1_img_size - 2
            sizes_list = [upconv_size, conv1_img_size, conv2_img_size]
            self.img_sizes_dict['expansive_path'] += sizes_list
            img_size = conv2_img_size

        

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_cropped4 = self.central_crop(x, self.crop_img_sizes[0])
        x = self.relu(self.conv3(self.pool(x)))
        x = self.relu(self.conv4(x))
        x_cropped3 = self.central_crop(x, self.crop_img_sizes[1]) 
        x = self.relu(self.conv5(self.pool(x)))
        x = self.relu(self.conv6(x))
        x_cropped2 = self.central_crop(x, self.crop_img_sizes[2])
        x = self.relu(self.conv7(self.pool(x)))
        x = self.relu(self.conv8(x))
        x_cropped1 = self.central_crop(x, self.crop_img_sizes[3])
        x = self.relu(self.bottleneck_conv1(self.pool(x)))
        x = self.relu(self.bottleneck_conv2(x))
        x = self.concat((x_cropped1, self.upconv1(x)))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.concat((x_cropped2, self.upconv2(x)))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.concat((x_cropped3, self.upconv3(x)))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.concat((x_cropped4, self.upconv4(x)))
        x = self.relu(self.conv16(x))
        x = self.relu(self.conv17(x))
        x = self.conv18(x)
        return x





net = UNet((1, 508, 508), 2)
net.calculate_image_sizes()
print(net.img_sizes_dict)
rand_img = torch.rand((1, 508, 508))
print(net(rand_img).shape)


'''
fconv_segnet = FullyConvSegNet(5, (3, 5, 5), 6)
img = torch.rand((3, 500, 500))
segm_mask = fconv_segnet(img)
print(fconv_segnet)
print(segm_mask.shape)
'''

print(summary(net))