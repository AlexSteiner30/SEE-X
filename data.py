import matplotlib.pyplot as plt
import nibabel as nib

from PIL import Image
import numpy as np

import glob as glob
import json
import os

pwd = "/Users/alex.steiner/Downloads/Dataset/"
subfolders = [ f.path for f in os.scandir(pwd) if f.is_dir() ]

count = 0
z = []
for i in subfolders:
    for j in glob.glob(i + '/*.txt' , recursive=True):
        x = True

        with open(j) as f:
            y = ({"index" : count, "x": f.readline() != ""}) 
            z.append(y)

            test_load = nib.load(i + '/orig/TOF.nii.gz').get_fdata()
            plt.axis('off')
            plt.imshow(test_load[:,:,50])

            plt.savefig("dataset/" + str(count), bbox_inches='tight',transparent=True, pad_inches=0)
            Image.open('dataset/' + str(count) + '.png').convert('L').save('dataset/' + str(count) + '.png')

            count = count + 1


with open('description.json', mode='w') as json_file:
    print(z)
    json_file.write(json.dumps(z, indent=2) + '\n')