import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv()  # File path

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

def Distance(Xi, N):
    d = [(Xi[j] - N[j])**2 for j in range(len(N))]
    return np.sqrt(sum(d))

def K_Distances_Min(X_train, y_train, N, K):
    Distances = []
    for i in range(X_train.shape[0]):
        Xi = list(X_train.iloc[i, :])
        Yi = y_train.iloc[i]
        di = Distance(Xi, N)
        Distances.append((di, Yi))
    DistancesTrie = sorted(Distances, key=lambda x: x[0])
    return DistancesTrie[:K]

Xi = [0, 0, 0]
N = [1, 1, 1]
print("Distance entre Xi et N:", Distance(Xi, N))

Xi = list(X_train.iloc[0, :])  
indN = 13  
N = list(X_test.iloc[indN, :])  
print("Distance entre le premier échantillon de X_train et l'échantillon 13 de X_test:", Distance(Xi, N))

Yn = y_test.iloc[indN]
print("Étiquette réelle de l'échantillon 13 dans y_test:", Yn)

K_Dmin = K_Distances_Min(X_train, y_train, N, 5)  # Par defaut est 5 
print("5 plus proches voisins (distance et étiquette):",K_Dmin)

def frequent(List):
    return max(set(List), key=List.count)

def ClassifierKNN(X_train,y_train,N,K):
    k_distances=K_Distances_Min (X_train,y_train,N,K)
    u=[]
    for e in range(len(k_distances)):
        u.append(k_distances[e][1])
    return frequent(u)
    
N=[6,148,63,35,4.6,0.722,50]
print("Le scalaire :",ClassifierKNN(X_train,y_train,N,5))

def Prediction(X_train, y_train, X_test, K):
    y_pred = []
    for i in range(len(X_test)):
        N = list(X_test.iloc[i, :])
        y_pred.append(ClassifierKNN(X_train, y_train, N, K))
    return y_pred

def Score(y_test, y_pred):
    score = sum([1 for i in range(len(y_test)) if y_test.iloc[i] == y_pred[i]])
    return score / len(y_test)

def Matrice_Confusion(y_test, y_pred):
    return pd.crosstab(pd.Series(y_test, name="Réel"), pd.Series(y_pred, name="Prévu"))

K = int(input("Saisir 'k' :"))   # Saisir K 

y_pred = Prediction(X_train, y_train, X_test, K)
    
accuracy = Score(y_test, y_pred)
    
confusion_matrix = Matrice_Confusion(y_test, y_pred)

print(f"Précision pour k={K}: {accuracy * 100:.2f}%")
print("Matrice de confusion :")
print(confusion_matrix)
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv()  # File path

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

def Distance(Xi, N):
    d = [(Xi[j] - N[j])**2 for j in range(len(N))]
    return np.sqrt(sum(d))

def K_Distances_Min(X_train, y_train, N, K):
    Distances = []
    for i in range(X_train.shape[0]):
        Xi = list(X_train.iloc[i, :])
        Yi = y_train.iloc[i]
        di = Distance(Xi, N)
        Distances.append((di, Yi))
    DistancesTrie = sorted(Distances, key=lambda x: x[0])
    return DistancesTrie[:K]

Xi = [0, 0, 0]
N = [1, 1, 1]
print("Distance entre Xi et N:", Distance(Xi, N))

Xi = list(X_train.iloc[0, :])  
indN = 13  
N = list(X_test.iloc[indN, :])  
print("Distance entre le premier échantillon de X_train et l'échantillon 13 de X_test:", Distance(Xi, N))

Yn = y_test.iloc[indN]
print("Étiquette réelle de l'échantillon 13 dans y_test:", Yn)

K_Dmin = K_Distances_Min(X_train, y_train, N, 5)  # Par defaut est 5 
print("5 plus proches voisins (distance et étiquette):",K_Dmin)

def frequent(List):
    return max(set(List), key=List.count)

def ClassifierKNN(X_train,y_train,N,K):
    k_distances=K_Distances_Min (X_train,y_train,N,K)
    u=[]
    for e in range(len(k_distances)):
        u.append(k_distances[e][1])
    return frequent(u)
    
N=[6,148,63,35,4.6,0.722,50]
print("Le scalaire :",ClassifierKNN(X_train,y_train,N,5))

def Prediction(X_train, y_train, X_test, K):
    y_pred = []
    for i in range(len(X_test)):
        N = list(X_test.iloc[i, :])
        y_pred.append(ClassifierKNN(X_train, y_train, N, K))
    return y_pred

def Score(y_test, y_pred):
    score = sum([1 for i in range(len(y_test)) if y_test.iloc[i] == y_pred[i]])
    return score / len(y_test)

def Matrice_Confusion(y_test, y_pred):
    return pd.crosstab(pd.Series(y_test, name="Réel"), pd.Series(y_pred, name="Prévu"))

K = int(input("Saisir 'k' :"))   # Saisir K 

y_pred = Prediction(X_train, y_train, X_test, K)
    
accuracy = Score(y_test, y_pred)
    
confusion_matrix = Matrice_Confusion(y_test, y_pred)

print(f"Précision pour k={K}: {accuracy * 100:.2f}%")
print("Matrice de confusion :")
print(confusion_matrix)
