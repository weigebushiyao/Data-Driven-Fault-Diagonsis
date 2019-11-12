import pandas as pd
import numpy as np
from imblearn import SMOTE

X=np.array(pd.read_csv("data.csv"))
Y=np.array(pd.read_csv("labels.csv"))

smo = SMOTE(random_state=42)
X_smo, Y_smo = smo.fit_sample(X, Y)
X_smo=np.array(X_smo)
Y_smo=np.array(Y_smo)
np.savetxt('data_new.csv',X_smo,fmt='%e',delimiter=',')
np.savetxt('labels_new.csv',Y_smo,fmt='%e',delimiter=',')