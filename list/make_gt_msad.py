import numpy as np
import pandas as pd
import cv2

clip_len = 16

# the dir of testing images
feature_list = 'MSAD_test.csv'
# the ground truth txt

gt_txt = 'anomaly_annotations_txt.txt'     ## the path of test annotations
gt_lines = list(open(gt_txt))
print(gt_lines[0].split()[3])
gt = []
lists = pd.read_csv(feature_list)
count = 0
count2 = 0

for idx in range(lists.shape[0]):
    name = lists.loc[idx]['path']
    # if '__0.npy' not in name:
    #     continue
    #feature = name.split('label_')[-1]
    fea = np.load(name)
    lens = (fea.shape[0] + 1) * clip_len
    name = name.split('/')[-1]
    name = name[:-4]
    # the number of testing images in this sub-dir

    gt_vec = np.zeros(lens).astype(np.float32)
    if 'normal' not in name:
        for gt_line in gt_lines:
            if name in gt_line:
                count += 1
                gt_content = gt_line.strip('\n').split(' ')
                print(gt_content)
                abnormal_fragment = [[int(gt_content[3]),int(gt_content[4])]]
                if len(abnormal_fragment) != 0:
                    abnormal_fragment = np.array(abnormal_fragment)
                    for frag in abnormal_fragment:
                        if frag[0] != -1 and frag[1] != -1:
                            gt_vec[frag[0]:frag[1]]=1.0
                break
    else:
        count2+=1
    gt.extend(gt_vec[:-clip_len])

print(count,count2)
np.save('gt_msad_new.npy', gt)