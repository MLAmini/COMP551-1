import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib 
from os import listdir
import os
import math

_smooth = 1
threshold = np.arange(0, 1.45, 0.01)

output_path = './pred_stats_output/'
#data_type = ['training_training_predictions/', 'training_validation_predictions/']
data_type = ['training_training_predictions/']
#unc_type = ['entropy/whole/', 'bald_uncertainty/whole/']
unc_type = ['bald_uncertainty/whole/']
pred = 'segmentation/'
unc = ''

truth = './Brats17TrainingData/All/'

#class 0 = binary classification
cl = [[0], [1], [2], [4], [1, 4]] 
class_names = ['binary', 'class 1', 'class 2', 'class 4', 'core']
#Tanya's code, how much too things overlap
def global_dice_bin(h, t):
  h[h > 1] = 1
  t[t > 1] = 1
  h = h.flatten()
  t = t.flatten()
  intersection = np.sum(h * t)
  union = np.sum(h) + np.sum(t)
  dice = (2. * intersection + _smooth) / (union + _smooth)
  return dice



#dice per class
def global_dice_class(h, t):
  dice = []
  for c in cl:
    h_temp = h.copy()
    t_temp = t.copy()
    if 0 not in c:
	mask = np.isin(h_temp, c, invert=True)
    	h_temp[mask] = 0
    	mask = np.isin(t_temp, c, invert=True)
    	t_temp[mask] = 0
    
    dice.append(global_dice_bin(h_temp, t_temp))
  return np.array([dice])



def main():
    for type in data_type:
        for unc in unc_type:
            pred_path = './exp_1/' + type + pred
	    if not os.path.exists(output_path + type):
                os.mkdir(output_path + type)
            if not os.path.exists(output_path + type + unc.split('/')[0]):
		os.mkdir(output_path + type + unc.split('/')[0])
	    if not os.path.exists(output_path + type + unc):
		os.mkdir(output_path + type + unc)
	    output_dir = output_path + type + unc
            unc = './exp_1/' + type + unc
	    ds = listdir(pred_path)
	    res_dice_class = [pd.DataFrame() for i in range(len(cl))]

            i=0
	    print(output_dir)
            for subj in ds:
                print(i + 0, " of ", len(ds))
                #model prediction
                h = np.asarray(nib.load(pred_path + subj).get_data())
                name = subj.split('.')[0]
                #ground truth
                t = np.asarray(nib.load(truth + name + '/' + name + '_seg.nii.gz').get_data())
                #uncertainty measure
                u = np.asarray(nib.load(unc  + subj).get_data())

                dice_bin = []
                dice_class = np.array([[] for c in range(len(cl))])
                
                for e in threshold:
	                 h_thres = h.copy()
        	         h_thres[u > e] = 0
                   	 dice_class = np.append(dice_class, global_dice_class(h_thres, t).T, axis=1)
                for c in range(len(cl)):
                    res_dice_class[c][name] = dice_class[c]
                i += 1
                
            mean_dice_class = [np.array(res_dice_class[c].mean(axis = 1)) for c in range(len(cl))]
	    mean_df = pd.DataFrame({'thres': threshold})

	    for i in range(len(cl)):
		mean_df[class_names[i]] = mean_dice_class[i]
#            print(mean_df)
	    mean_df.to_csv(output_dir + 'dice_class.csv', index=False)
	    	    

if __name__=='__main__':
    main()


