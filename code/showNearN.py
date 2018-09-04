import matplotlib.pyplot as plt
from PIL import Image

data_dir = '/home/ubuntu/assignments/hw1/data/VOCdevkit/VOC2007'

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

IMG_IND =[ 
	320, 
	44,
	425,
	583,
	402,
	559,
	345,
	567,
	401,
	752
]
Pool5_alex_IND = [4873,  526 ,2966, 2293 ,3232 ,4782, 1013, 2820, 1252, 4703]

fc7_alex_IND = [4873, 4819, 2966,   35, 3194, 4782, 1758, 4759, 1252, 4703]

Pool5_vgg16_IND = [3395, 3395, 1743, 2660, 3395, 3831, 1022, 2665, 1252,  265]

fc7_vgg16_IND =[4032, 4025, 4044, 4484, 2726, 4720,  818, 3501 ,1252, 1882] 

img_count = 0
IMGPATH = data_dir+'/JPEGImages'
IDXPATH = data_dir+'/ImageSets/Main'

img_dict = dict()
with open(IDXPATH+'/'+CLASS_NAMES[0]+'_test.txt', 'r') as f:
	for line in f.readlines():
	    img_count += 1
	    img_label = line.strip().split()
	    img_idx = img_label[0]
	    img_dict[img_count] = img_idx

fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 3
for i in range(1, 11):
    img = Image.open(IMGPATH+'/'+str(img_dict[Pool5_vgg16_IND[i-1]])+'.jpg')
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(img)
plt.savefig('Pool5_vgg16.jpg')
plt.show()

