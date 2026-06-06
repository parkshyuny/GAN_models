import matplotlib.pyplot as plt

from torch import nn

def sample_images(generated_images, image_size=128, subplots=(2,5), figsize=(22,8), save=False):
    plt.figure(figsize=figsize)

    for i, image in enumerate(generated_images):
        image = image.detach().numpy()
        plt.subplot(subplots[0], subplots[1], i+1)
        plt.imshow(image.reshape(image_size, image_size), cmap='gray')
        
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def initialize_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.normal_(layer.weight.data, 0.0, 0.02)

    if isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)