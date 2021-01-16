from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from matplotlib import pyplot
import pandas as pd 
import numpy as np


#%% pulizia dati
dataaa = pd.read_csv('C:/Users/simon/Desktop/progetto EEG/ML-EEG-eyes-model/eyestateeegds.csv', header= None, names=["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4","output"])

output = dataaa["output"].values
dataaa = dataaa[(dataaa<5000)&(dataaa>3500)]
dataaa["output"] = output
dataaa = dataaa.dropna()
values = dataaa.values
# %%. plot dei dati
pyplot.figure(figsize=(20,10))
for i in range(values.shape[1]):
	pyplot.subplot(values.shape[1], 1, i+1)
	pyplot.plot(values[:, i])
pyplot.show()
#%%.2
TrnValSet = dataaa[:13441]
TestSet = dataaa[13441:]
# %%.3
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


# %%.4
#accuracy modello da 0.1s a 2s del Test Set
y_datasTEST = []
for time in range(12,252,12):
  accuracy_per_tenth_s = sliding_window(TestSet,time,1536)
  y_datasTEST.append(accuracy_per_tenth_s)

print(*y_datasTEST, sep='\n')


# %%
#accuracy modello da 0.1s a 2s del Validation Set
y_datasVAL = []
for time in range(12,252,12):
  accuracy_per_tenth_s = sliding_window(TrnValSet,time,13441)
  y_datasVAL.append(accuracy_per_tenth_s)

print(*y_datasVAL, sep='\n')


# %%
x_sec = np.arange(0.1,2,0.1)
pyplot.figure(figsize=(20,10))
pyplot.plot(x_sec,y_datasTEST,'.-')
pyplot.axis([0.08, 2, 0.4, 1])
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
def walking_forward_val_5(TrnVal_set,test_set,datas_in_onefifth):
    y_datasVAL = []
    y_datasTEST = []
    for times in range(1,5,1):
        first_data = 0
        TrnVal_set = dataaa[:((times*datas_in_onefifth)+1)]
        Test_set = dataaa[((times*datas_in_onefifth)+1):((times+1)*datas_in_onefifth)]
        for time in range(12,132,12):
            accuracy_per_tenth_s = sliding_window(TrnValSet,time,(times*datas_in_onefifth))
            y_datasVAL.append(accuracy_per_tenth_s)
        for time in range(12,132,12):
            accuracy_per_tenth_s = sliding_window(TestSet,time,datas_in_onefifth)
            y_datasTEST.append(accuracy_per_tenth_s)        
        first_data = times*datas_in_onefifth
    accuracy_T = np.mean(y_datasTEST)
    accuracy_V =np.mean(y_datasVAL)
    return accuracy_T,accuracy_V
        
        
#%% 9. testWF
walking_forward_val_5(TrnValSet,TestSet,2995)
