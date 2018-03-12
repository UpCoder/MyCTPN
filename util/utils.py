import numpy as np


def shuffle(*parameters):
    first_arr = parameters[0]
    indexs = range(len(first_arr))
    np.random.shuffle(indexs)
    res = []
    for i in range(len(parameters)):
        res.append(np.array(parameters[i])[indexs])
    return res


def split2_train_val(image_paths, gt_paths):
    [image_paths, gt_paths] = shuffle(image_paths, gt_paths)
    rate = 0.8
    boundary = int(len(image_paths) * rate)
    img_paths_train = image_paths[:boundary]
    img_paths_val = image_paths[boundary:]
    gt_paths_train = gt_paths[:boundary]
    gt_paths_val = gt_paths[boundary:]
    return img_paths_train, gt_paths_train, img_paths_val, gt_paths_val
