import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


loaded_model = pickle.load(open('C:/Users/Acer/Downloads/prediction.py/trained_model.sav','rb'))
input = (4,	110,	92,	0	,0,	37.6,	0.191,	30
)
new = np.asarray(input)
reshape = new.reshape(1,-1)
pred = loaded_model.predict(reshape)
print(pred)
if (pred[0] == 0):
    print("The person is not diabetic.")
else:
    print("The person is diabetic")