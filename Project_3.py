#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import svm
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import metrics

threshold = 70  # threshold for hypoglycemic events

def rmse(x, y):
    error = (x - y) * 400  # scale up the output
    squared_error = np.square(error)
    mean_suqred_error = np.mean(squared_error)
    root_mean_suqred_error = np.sqrt(mean_suqred_error)
    return root_mean_suqred_error

def load_data(csv_file):

    data = pd.read_csv(csv_file)
    vars_to_include = []
    cbg = data['cbg'].values
    cbg = np.round(cbg) / 400  # resize all samples, so that they lay in range 0.1 to 1, approximately
    vars_to_include.append(cbg)
    vars_to_include.append(data['missing_cbg'].values)
    dataset = np.stack(vars_to_include, axis=1)
    dataset[np.isnan(dataset)] = 0
    return dataset

def extract_valid_sequences(data, min_len=144):
    ValidData = []
    i = 0
    sequence = []
    while i < data.shape[0]:  # dataset.shape[0] = number of train/test samples for one patient
        if data[i, -1] == 1:  # if we have missing values in the cbg measurements
            if len(sequence) > 0:
                if len(sequence) >= min_len:
                     ValidData.append(np.stack(sequence))
                sequence = []
            i = i + 1
        else:
            sequence.append(data[i, :-1])  # do not add the "missing_cbg" column
            i = i + 1
    return ValidData

def prepare_data(sequences, lookback, prediction_horizon, threshold, validation_split=None):
    samples = []
    targets = []
    targets_hypo = []
    threshold = threshold / 400
    for seq in sequences:
        assert seq.shape[0] > lookback + prediction_horizon
        for i in range(seq.shape[0] - lookback - (prediction_horizon - 1)):
            samples.append(seq[i: i + lookback, :])
            target_val = seq[i + lookback + prediction_horizon - 1, 0]
            targets.append(target_val)
            targets_hypo.append((target_val < threshold) * 1)
    samples = np.squeeze(np.stack(samples, axis=0))
    targets = np.stack(targets, axis=0)
    targets_hypo = np.stack(targets_hypo, axis=0)

    if validation_split is not None:
        num_train = int(samples.shape[0] * (1 - validation_split))
        num_val = samples.shape[0] - num_train
    else:
        num_train = samples.shape[0]

    train_samples = samples[0: num_train, :]
    train_targets = targets[0: num_train]
    train_targets_hypo = targets_hypo[0: num_train]
    # calculate class weights
    num_hypo = np.sum(train_targets_hypo * 1)
    num_non_hype = len(targets_hypo) - num_hypo
    class_counts = np.asarray([num_non_hype, num_hypo])
    overall_ild_frequencies = class_counts / (np.sum(class_counts) + 1e-8)  # shape: (8,)
    class_weights = np.median(overall_ild_frequencies) / (overall_ild_frequencies + 1e-8)  # shape: (8,)
    if validation_split is not None:
        val_samples = samples[-num_val:, :]
        val_targets = targets[-num_val:]
        val_targets_hypo = targets_hypo[-num_val:]
        return train_samples, train_targets, train_targets_hypo, class_weights, val_samples, val_targets, val_targets_hypo
    else:
        return train_samples, train_targets, train_targets_hypo, class_weights

def get_hypo_event(segment,threshold): 
    segment=segment*400
    for i in range(0,len(segment)):
        if segment[i]<threshold:         
            segment[i]= 1
        else:
            segment[i]= 0
    return segment

def metrics(gt_events, pred_events):
    gt_events=2*gt_events
    difference=gt_events-pred_events
    tn=0
    tp=0
    fp=0
    fn=0
    for i in range(0,len(difference)):
        if difference[i]==-1:
            fp=fp+1
        if difference[i]==0:
            tn=tn+1 
        if difference[i]==1:
            tp=tp+1
        if difference[i]==2:
            fn=fn+1
    sensitivity = (tp)/(tp + fn)
    specificity = (tn)/(tn + fp)       
    F = tp/(tp+0.5*(fp+fn))
    return sensitivity, specificity, F


def trueHypoEvent(data):
    data=np.array(data, dtype=int)
    for i in range(0,len(data)-2):
        if  data[i]>0 and data[i+1]>0 and data[i+2]>0:
            data[i]=2
            data[i+1]=2
            data[i+2]=2
            if (data[i]>0 and data[i+1]==0 and data[i+2]>0) or (data[i]>0 and data[i+1]==0 and data[i+2]==0 and data[i+3]==0):
                data[i]=2
                data[i+1]=2
                data[i+2]=2
            
    for i in range(0,len(data)):
        if data[i]==1:
            data[i]=0
    data=np.array(data/2, dtype=int)
    return data

def countEvent(segment):
    segment=np.array(segment, dtype=int)
    counter =0
    for i in range(0,len(segment)-1):
        if segment[i]==0 and segment[i+1]==1:
            counter=counter+1
    return counter


if __name__ == "__main__":
    lookback = 12 # 12 past 1 hour
    prediction_horizon = 6  # 30 minutes ahead in time

    
    patient_root = "/Users/Riccardo Gabrieli/Desktop/Ohio Data/Ohio2020_processed/train/596-ws-training_processed.csv"  # TODO: path to csv file
    patient_root_test = "/Users/Riccardo Gabrieli/Desktop/Ohio Data/Ohio2020_processed/test/596-ws-testing_processed.csv"  # TODO: path to csv file

    # Load training and testing data
    train_dataset = load_data(csv_file=patient_root)
    test_dataset = load_data(csv_file=patient_root_test)
    # The sequences contain missing measurements at some time steps due to sensor and/or user errors or off-time.
    # We only select sequences without any interruption for at least half a day (144 5-minute steps = 12h)
    valid_train = extract_valid_sequences(train_dataset, min_len=144)
    valid_test = extract_valid_sequences(test_dataset, min_len=144)

    train_samples, train_targets, train_targets_hypo, class_weights, val_samples, val_targets, val_targets_hypo = prepare_data(valid_train, lookback=lookback, prediction_horizon=prediction_horizon, threshold=threshold, validation_split=0.3)
    test_samples, test_targets, test_targets_hypo, _ = prepare_data(valid_test, lookback=lookback, prediction_horizon=prediction_horizon, threshold=threshold)

    
    # Implementation of SVR for BG prediction

    # Implementation of linear for BG prediction
    regressor_linear=SVR(kernel='linear') 
    regressor_linear.fit(train_samples,train_targets)
    pred_seq_svr_linear =regressor_linear.predict(test_samples)
    binary_svr_linear=get_hypo_event(pred_seq_svr_linear,threshold)

    # Implementation of poly for BG prediction
    regressor_poly=SVR(kernel='poly', degree = 2) 
    regressor_poly.fit(train_samples,train_targets)
    pred_seq_svr_poly =regressor_poly.predict(test_samples)
    binary_svr_poly=get_hypo_event(pred_seq_svr_poly,threshold)    
    
    # Implementation of rbf for BG prediction
    regressor_rbf=SVR(kernel='rbf') 
    regressor_rbf.fit(train_samples,train_targets)
    pred_seq_svr_rbf =regressor_rbf.predict(test_samples)
    binary_svr_rbf=get_hypo_event(pred_seq_svr_rbf,threshold)    
    

    # TODO: implement a function that returns a binary sequence, indicating if we are in a hypo-event (1) or not (0)
    gt_event_masks = get_hypo_event(test_targets, threshold)

    # Implemention of SVM for hypo-event prediction
    y_train=get_hypo_event(train_targets,threshold)   
    y_test=get_hypo_event(test_targets, threshold)    

    # SVM with class_weights
    clf = svm.SVC(class_weight={0:class_weights[0] , 1:class_weights[1]})
    clf.fit(train_samples,train_targets_hypo)
    pred_events_svm=clf.predict(test_samples)
    #pred_events_mask_svm=1- get_hypo_event(pred_events, threshold)


# In[25]:


# check if we have a hypo event in the ground truth
if np.max(gt_event_masks) == 1:

# pred_event_mask_direct
    sensitivity, specificity,F = metrics(y_test, binary_svr_rbf)
    print('binary_svr_rb\nsensitivity: {}\nspecificity: {}'.format(sensitivity, specificity))
    print('F-score: {}'.format(F))
    print('rmse: {}'.format(rmse(binary_svr_rbf, y_test)))
    print('\n')

# binary_svr_linear
    sensitivity, specificity,F = metrics(y_test, pred_events_svm)
    print('pred_events_svm\nsensitivity: {}\nspecificity: {}'.format(sensitivity, specificity))
    print('F-score: {}'.format(F))
    print('rmse: {}'.format(rmse(pred_events_svm,y_test)))
    print('\n')

else:
    print('patient did not have any phase in GT below {}mg/dl'.format(threshold))
    


# In[26]:


plt.plot(regressor_linear.predict(test_samples)[1:200]*400, color = 'green')
plt.plot(regressor_poly.predict(test_samples)[1:200]*400, color = 'blue')
plt.plot(regressor_rbf.predict(test_samples)[1:200]*400, color = 'orange')
plt.plot(test_targets[1:200]*400, color = 'red')

plt.xlabel('Measurement number')
plt.ylabel('Blood glucose (mg/dL)')
plt.title('Blood glucose prediction with SVR')
plt.grid(True)
plt.legend(['regressor_linear', 'regressor_poly','regressor_rbf','test_targets']);
plt.show()

print('rmse for linear regressor: {}'.format(rmse(regressor_linear.predict(test_samples)[1:200],test_targets[1:200])))
print('rmse for polynomial regressor: {}'.format(rmse(regressor_poly.predict(test_samples)[1:200],test_targets[1:200])))
print('rmse for rbf regressor: {}'.format(rmse(regressor_rbf.predict(test_samples)[1:200],test_targets[1:200])))


# In[27]:


plt.plot(regressor_linear.predict(test_samples)*400, color = 'green')
plt.plot(test_targets*400, color = 'red')


plt.xlabel('Measurement number')
plt.ylabel('Blood glucose (mg/dL)')
plt.title('Blood glucose prediction with SVR')
plt.grid(True)
plt.legend(['regressor_linear', 'regressor_poly','regressor_rbf','test_targets']);
plt.show()



# In[28]:


# SVM
plt.plot(pred_events_svm[0:500]*0.975,color='brown',label='pred_events_mask_direct',linewidth=1)
plt.plot(binary_svr_linear[0:500]*0.925, color='green',label='binary_svr_linear',linewidth=1)
plt.plot(y_test[0:500]*0.9,color='red',label='y_test')

plt.grid(True)
plt.legend(['pred_events_svm','binary_svr_linear','y_test']);
plt.show()


