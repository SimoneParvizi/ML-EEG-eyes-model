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
#%%2. trn e test

TrnValSet = data[:13441]
TestSet = data[13441:]
# %%3. Sliding window

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

#test:                                               
#sliding_window(TrainValSet,12,13441)


# %% 4. model function for predicting from 0.1 to 1s

def range_secs(dtframe,dt_in_onetenth_s,dt_in_max_s,dt_in_dtframe):
    y = []
    for time in range(dt_in_onetenth_s,dt_in_max_s,dt_in_onetenth_s):
      accuracy_per_tenth_s = sliding_window(dtframe,time,dt_in_dtframe)
      y.append(accuracy_per_tenth_s)
    return y

# %% 5.test and val model accuracy from 0.1s to 2s 

y_datasTEST = range_secs(TestSet,12,252,1536)
y_datasVAL = range_secs(TrnValSet,12,252,13441) 

#%%

print(y_datasTEST,sep='\n')
print(y_datasVAL, sep='\n')

# %% 6. Test and Val plots
x_sec = np.arange(0.1,2.1,0.1)
pyplot.figure(figsize=(20,10))
pyplot.plot(x_sec,y_datasTEST,'.-')
pyplot.axis([0.08, 2.1, 0.4, 1])
pyplot.plot(x_sec,y_datasVAL,'.-')
pyplot.ylabel('accuracy%')
pyplot.xlabel('seconds')
pyplot.grid(True)
pyplot.show()

y_datas = []
x = sliding_window(TrnValSet,12,13441)
y_datas.append(x)

yyy = np.array(y_datas)
x_sec = np.arange(0.1,112.1,0.1)
pyplot.figure(figsize=(20,10))
pyplot.plot(x_sec,yyy,'.-')
pyplot.axis([-0.1, 113, -0.1, 1.1])
pyplot.ylabel('accuracy%')
pyplot.xlabel('seconds')
pyplot.grid(True)
pyplot.show()

#%% 8. walkingforward
def walking_forward_val_5(TrnVal_set,Test_set,datas_in_onefifth):
    #Y_datasVAL = []
    Y_datasTEST = []
    for times in range(1,5,1):
        #TrnVal_set = data[:(times*datas_in_onefifth)]
        Test_set = data[(times*datas_in_onefifth):((times+1)*datas_in_onefifth)]
        #for time in range(12,132,12):
        #    accuracy_per_tenth_s = sliding_window(TrnVal_set,time,(times*datas_in_onefifth))
        #    Y_datasVAL.append(accuracy_per_tenth_s)
        for time in range(12,132,12):
            accuracy_per_tenth_s = sliding_window(Test_set,time,datas_in_onefifth)
            Y_datasTEST.append(accuracy_per_tenth_s)        
    accuracy_T = np.mean(Y_datasTEST)
    #accuracy_V = np.mean(Y_datasVAL)
    print(accuracy_T)
        
        
#%% 9. testWF
walking_forward_val_5(TrnValSet,TestSet,2995)

#%%
for times in range(1,5,1):
    x = 2995
    first_data = 0
    TrnVal_set = data[:(times*x)]
    Test_set = data[(times*x):((times+1)*x)]
    first_data = times*x
    print (Test_set, sep='\n')
    #%%
Y_datasTEST = []
for times in range(1,5,1):
    Test_set = data[(times*2995):((times+1)*2995)]
    for time in range(12,132,12):
        accuracy_per_tenth_s = sliding_window(Test_set,time,2995)
        Y_datasTEST.append(accuracy_per_tenth_s)        
accuracy_T = np.mean(Y_datasTEST)
print(accuracy_T,sep='\n')
#%%
print(len(Y_datasTEST))
print(*Y_datasTEST,sep='\n')











































































