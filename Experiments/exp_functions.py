import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.callbacks import TerminateOnNaN
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import sys
sys.path.append("../")
from main import data_reduction

#Parameters
n_iter=10
epochs=50
learning_rate=0.01
threshold = np.arange(0.05, 1.00, 0.05)
perc=0.20
epsilon=None
perc_base=perc*0.5
initial_epsilon=5
max_iter=20
topological_radius=4
scoring_version='multiDim'
dimension=1
landmark_type='representative'
decomposition='SVD_python'
RANSAC=20 
sigma=3
opt='osqp'
methods=['None','SRS','CLC','MMS','DES','DOM','PHL','NRMD','PSA','PRD']

#Dataframes with metrics and statistics
metrics = pd.DataFrame([[0] * 6 for _ in range(n_iter)],
                       columns=['RT', 'TT', 'Acc',
                                'Spe', 'Sen', 'F1'])
stats = pd.DataFrame([[0] * 12 for _ in range(len(methods))],
                     index=methods,
                     columns=["RT Mean","RT Var","TT Mean","TT Var","Acc Mean","Acc Var",
                              "Spe Mean","Spe Var","Sen Mean","Sen Var","F1 Mean","F1 Var"])

def experiment(X_train, y_train, X_test, y_test):
    file_path = 'metrics.xlsx'
    for m in methods:
        print(m)
        all_iterations_for_method(X_train, y_train, X_test, y_test, method=m)
        stats.to_excel(file_path, index=True)

def plot_dataset(X0, X1, X0_res=None, X1_res=None, title=None, pic_name=None):
    plt.clf()
    # Plot original data
    plt.scatter(X0[:, 0], X0[:, 1], marker='o', edgecolors='blue', facecolors='none', alpha=0.1, label='Class 0 (Original)')
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', edgecolors='red', facecolors='none', alpha=0.1, label='Class 1 (Original)')
    # Plot additional data if provided
    if X0_res is not None:
        plt.scatter(X0_res[:, 0], X0_res[:, 1], marker='o', color='blue', label='Class 0 (Additional)')
    if X1_res is not None:
        plt.scatter(X1_res[:, 0], X1_res[:, 1], marker='o', color='red', label='Class 1 (Additional)')
    # Add labels, title, and show
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    if pic_name == None:
        plt.show()
    else:
        plt.savefig(pic_name)

def make_model(X_train, y_train):

    input_shape=X_train.shape[1:] 
    
    input_layer = keras.layers.Input(input_shape, name="entrada 1")
    dense1 = keras.layers.Dense(8,activation="relu")(input_layer)
    dropout1 = keras.layers.Dropout(0.25)(dense1)
    dense2 = keras.layers.Dense(4,activation="relu")(dropout1)
    dropout2 = keras.layers.Dropout(0.25)(dense2)
    output_layer_clasification = keras.layers.Dense(1,activation="sigmoid",name="classification")(dropout2)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer_clasification)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics="accuracy")
    model_noNaN = TerminateOnNaN()
    callbacks = [model_noNaN]
    return model, callbacks

def all_iterations_for_method(X_train,y_train,X_test,y_test,method):
    for i in range(n_iter):
        print("Iteration ",i+1,"/",n_iter)
        iteration(X_train,y_train,X_test,y_test,method,index=i)
    stats.loc[method]=(metrics['RT'].mean(),metrics['RT'].var(),
                  metrics['TT'].mean(), metrics['TT'].var(),
                  metrics['Acc'].mean(), metrics['Acc'].var(),
                  metrics['Spe'].mean(), metrics['Spe'].var(),
                  metrics['Sen'].mean(), metrics['Sen'].var(),
                  metrics['F1'].mean(), metrics['F1'].var())

def iteration(X_train,y_train,X_test,y_test,method,index):
    shuffle = np.random.permutation(len(y_train))
    X_train = X_train[shuffle]
    y_train = y_train[shuffle]
    time_start=time.time()
    X_red, y_red = data_reduction(X_train, y_train, method, 
                                  perc,epsilon,perc_base,initial_epsilon,max_iter,topological_radius,scoring_version,
                                  dimension,landmark_type,decomposition,RANSAC,sigma,opt)
    time_end=time.time()
    reduction_time=time_end-time_start
    plot_dataset(X_train[y_train==0],X_train[y_train==1],
             X_red[y_red==0],X_red[y_red==1],
             title=method+' Selection',pic_name=method+'_reduction.png')

    model, callbacks = make_model(X_train=X_red, y_train=y_red)
    
    time_start=time.time()
    history = model.fit(X_red, y_red, callbacks=callbacks,epochs=epochs, validation_data=(X_test,y_test),verbose=0)
    time_end=time.time()
    training_time=time_end-time_start
    
    y_pred=model.predict(X_test)
    acc=0
    specificity=0
    sensitivity=0
    f1=0
    y_pred_opt = [None]*len(y_pred)
    t_opt = 0
    for t in threshold:
        y_pred_t = [1 if i > t else 0 for i in y_pred]
        f1_t = f1_score(y_test, y_pred_t)
        if f1_t > f1:
            y_pred_opt = y_pred_t
            t_opt = t
            f1 = f1_t
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    f1 = f1_score(y_test, y_pred_opt)
    acc = accuracy_score(y_test,y_pred_opt)
    
    metrics.iloc[index,0] = reduction_time
    metrics.iloc[index,1] = training_time
    metrics.iloc[index, 2] = acc  
    metrics.iloc[index, 3] = specificity  
    metrics.iloc[index, 4] = sensitivity  
    metrics.iloc[index, 5] = f1