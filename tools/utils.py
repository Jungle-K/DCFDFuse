import os
import cv2
import numpy as np
from skimage.io import imsave
def write_mes(msg, log_name=None, show=True, mode='a'):
    get_end = lambda line: '' if line.endswith('\n') else '\n'
    if show:
        if isinstance(msg, str):
            print(msg, end=get_end(msg))
        elif isinstance(msg, (list, tuple)):
            for line in msg:
                print(line, end=get_end(line))  # might be a different thing
        else:
            print(msg)

    if log_name is not None:
        with open(log_name, mode) as f:
            if mode == 'a':
                f.writelines('\n')
            f.writelines(msg)

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img