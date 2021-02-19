#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from matplotlib import pyplot
import pandas as pd 
import numpy as np


#%% data cleaning

data = pd.read_csv('C:/Users/simon/Desktop/progetto EEG/ML-EEG-eyes-model/eyestateeegds.csv', header= None, names=["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4","output"])

output = data["output"].values
data = data[(data<5000)&(data>3500)]
data["output"] = output
data = data.dropna()
values = data.values


# %%. data plot

pyplot.figure(figsize=(20,10))
for i in range(values.shape[1]):
	pyplot.subplot(values.shape[1], 1, i+1)
	pyplot.plot(values[:, i])
pyplot.show()


# %%3. function that goes through the WHOLE dataframe making new x/ytrn e x/yval
#      ALWAYS as long as "array_lenght"value


def sliding_window(dtset,array_lenght,num_dati_nel_dtframe):
  model = RandomForestClassifier(n_estimators=45)
  array_start = 0
  f1_tot = []
  for x in range(array_lenght,num_dati_nel_dtframe,array_lenght):
    xtrn_no_output = dtset.drop(['output'],axis=1)
    xtrn = xtrn_no_output[(array_start):(x)]
    ytrn = dtset.iloc[(array_start):(x),-1]
    xval = xtrn_no_output[(x):((x)+array_lenght)]
    yval = dtset.iloc[(x):((x)+array_lenght),-1]
    model.fit(xtrn,ytrn)
    prev = model.predict(xval)
    f1_value = f1_score(yval,prev,zero_division=1)
    f1_tot.append(f1_value)
    array_start = x
  accuracy = np.mean(f1_tot)
  return accuracy


# %% if you put "return f1_tot" you can have an overlapping plot confirming that the 0% 
#    accuracy points, match the exact moment of opening and 

                                            
f1_list = sliding_window(data,12,14976)

# %%
print(len(f1_list))

# %% 6. Overlapping plots
f1_lst = np.array(f1_list)
f1_masked = np.ma.masked_where(f1_lst == 1 ,f1_list)
x_sec = np.arange(12,14976,12)
pyplot.figure(figsize=(20,10))
pyplot.plot(x_sec,f1_masked,'.-',markersize=20, label="Predictions < 99%")
data["output"].plot(label="Output")
pyplot.legend(bbox_to_anchor=(1.02,1))
pyplot.ylabel('Accuracy %')
pyplot.xlabel('Number of Data')
pyplot.grid(True)
pyplot.show()


#%% 8. walkingforward (5 splits)


def walking_forward_5(dtset,datas_in_onefifth):
    Y_datasVAL = []
    Y_datasTEST = []
    for times in range(1,6,1): 
        TrnVal_set = dtset[:(times*datas_in_onefifth)]
        Test_set = dtset[(times*datas_in_onefifth):((times+1)*datas_in_onefifth)]
        for time in range(12,132,12):          # loop that applies sliding window with values from 0.1 to 1s for TRN e VAL
            accuracy_per_tenth_s = sliding_window(TrnVal_set,time,(times*datas_in_onefifth))
            Y_datasVAL.append(accuracy_per_tenth_s)
        for time in range(12,132,12):         # loop that applies sliding window with values from 0.1 to 1s for TEST
            accuracy_per_tenth_s = sliding_window(Test_set,time,datas_in_onefifth)
            Y_datasTEST.append(accuracy_per_tenth_s)        
    accuracy_T = np.mean(Y_datasTEST)
    accuracy_V = np.mean(Y_datasVAL)
    return accuracy_V, accuracy_T
        
        
#%% 9. testWF

final_accuracy = walking_forward_5(data,2995)
print(len(final_accuracy))

#%%
Y_datasTEST = []
for times in range(1,6,1):
    Test_set = data[(times*2995):((times+1)*2995)]
    for time in range(12,132,12):
        accuracy_per_tenth_s = sliding_window(Test_set,time,2995)
        Y_datasTEST.append(accuracy_per_tenth_s)  
        
# accuracy_T = np.mean(Y_datasTEST)
# print(len(Y_datasTEST))
# print(*Y_datasTEST,sep='\n')
# print(np.mean(Y_datasTEST)) 
# %% Final plot with all different splits
x_sec = np.arange(0.1,1.1,0.1)
pyplot.figure(figsize=(20,10))
pyplot.plot(x_sec,Y_datasTEST[:10],'.-',label="First split")  
pyplot.plot(x_sec,Y_datasTEST[10:20],'.-',label="Second split") 
pyplot.plot(x_sec,Y_datasTEST[20:30],'.-',label="Third split") 
pyplot.plot(x_sec,Y_datasTEST[30:40],'.-',label="Fourth split") 
pyplot.axis([0.08, 1.1, 0.4, 1])
pyplot.legend(loc="lower right")
pyplot.ylabel('Accuracy %')
pyplot.xlabel('Time window analyzed')
pyplot.grid(True)
pyplot.show()
dd





































































