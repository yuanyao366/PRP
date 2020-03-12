from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('AGG')#PDF/SVG/PS
import matplotlib.pyplot as plt
import os

i = 0;


def showPredict(data):
    print(data.shape)
    image = data[0,:,0,:,:]
    image = image.detach()
    image = image.cpu().numpy()
    image = image.transpose(1,2,0)
    image = image *255
    image = Image.fromarray(np.uint8(image));
    image.show();
def showoneofthem(data,root_path,epoch,step,tag):
    f_idx = 5
    image =  data[:,f_idx,:,:]
    image = image.detach()
    image = image.cpu().numpy()
    image = image.transpose(1, 2, 0)
   # image = image * 255
    plt.imshow(image)
   #  plt.show()
    img_path = os.path.join(root_path,'imgs','e{}'.format(epoch))
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    img_name = img_path + '/s{}_f{}_{}.jpg'.format(step,f_idx,tag)
    plt.savefig(img_name)
    
    f_idx = -1
    image2 =  data[:,-1,:,:]
    image2 = image2.detach()
    image2 = image2.cpu().numpy()
    image2 = image2.transpose(1, 2, 0)
   # image = image * 255
    plt.imshow(image2)
   #  plt.show()
    img_path = os.path.join(root_path,'imgs','e{}'.format(epoch))
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    img_name = img_path + '/s{}_f{}_{}.jpg'.format(step,f_idx,tag)
    plt.savefig(img_name)
