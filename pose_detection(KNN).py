from asyncore import write
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
 
header = ['nose','l_shouldr','r_shoulder','l_elbow','r_elbow','l_wrist','r_wrist','l_hip','r_hip','l_knee','r_knee','l_ankle','r_ankle','output']

with open('datasets.csv' , 'w' , newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
