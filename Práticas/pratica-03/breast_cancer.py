import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

cancer=load_breast_cancer()

df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
df['target'] = cancer.target
selecao=df.iloc[:,[1,3,5,7,30]]

sns.pairplot(selecao, hue="target")
plt.show()

# correr no jupyter-lab (terminal jupyter-lab)