import numpy as np
from csv import reader
import pandas as pd

class NaivebayesUevora:
    
    # Construtor que recebe o valor de alpha e o nome do ficheiro
    def __init__(self, a, dataset):
        self.aplha = a                         # valor de alpha
        self.data = dataset                    # nome do ficheiro
        
    # Método que faz a leitura dos dados do ficheiro
    def read(self):
        self.table = pd.read_csv(self.data)
        #print(self.table)
        
    # Método que faz a separação 
    def split_features_target(self):
        f = self.table.drop([self.table.columns[-1]], axis = 1)
        t = self.table[self.table.columns[-1]]
        return f, t
    
    # Método que calcula probabilidade com o estimador utilizado
    def calculate_prob(self, cases_number, total, num_values):
        p = (cases_number + self.aplha) / (total + (self.aplha*num_values))
        return p
    
    # Método que gera um classificador a partir de um conjunto de treino com etiquetas
    def fit(self, features, target):
        self.X_train = features                                 # table (possui só as caracteristicas de todos os exemplos/casos)
        self.y_train = target                                   # table (possui apenas as etiquetas "yes" / "no")
        self.examples = len(self.X_train)                       # Número de exemplos/casos (rows)
        self.features = list(features.columns)                  # Lista que contêm o nome das colunas 
        self.num_values_feat = {}                               # Dicionário que guarda o número de diferentes valores possiveis de cada caracteristica/coluna
        self.num_values_targ = len(np.unique(self.y_train))     # Dicionário que guarda o número de diferentes valores possiveis na etiqueta
        self.count_yes = 0                                      # Número de yes's
        self.count_no = 0                                       # Número de no's
        
        for feat in self.features:
            n = len(np.unique(self.X_train[feat]))
            self.num_values_feat.update({feat : n})
            
        for e in range(self.examples):
            if(self.y_train.loc[e] == "yes"):
                self.count_yes+=1
            else:
                self.count_no+=1
            
        self.occurrences_feat_values_yes()         
        self.occurrences_feat_values_no()
        
        p_yes = self.calculate_prob(self.count_yes,self.examples,self.num_values_targ)         # P(yes)
        p_no = self.calculate_prob(self.count_no,self.examples,self.num_values_targ)           # P(no)
        
        self.probability_yes(p_yes)
        self.probability_no(p_no)
        
    # Método que calcula as probabilidades (yes)    P(_|yes)
    def probability_yes(self, py):
        self.all_prob_yes = {}
    
        self.all_prob_yes["Pyes"] = py
        for chave1 in self.occurrences_feat_val_yes.keys():
            for feat in self.features:
                for feat_val in np.unique(self.X_train[feat]):
                    if(chave1 == feat_val):
                        p = self.calculate_prob(self.occurrences_feat_val_yes[chave1],self.count_yes,self.num_values_feat[feat])

                        self.all_prob_yes.update({chave1 : p})
        #print(self.all_prob_yes)
         
    # Método que calcula as probabilidades (no)    P(_|no)
    def probability_no(self, pn):
        self.all_prob_no = {}
        
        self.all_prob_no["Pno"] = pn
        for chave1 in self.occurrences_feat_val_no.keys():
            for feat in self.features:
                for feat_val in np.unique(self.X_train[feat]):
                    if(chave1 == feat_val):
                        p = self.calculate_prob(self.occurrences_feat_val_no[chave1],self.count_no,self.num_values_feat[feat])
                        
                        self.all_prob_no.update({chave1 : p})
        #print(self.all_prob_no)
    
    # Método que determina o número de ocorrências de cada valor possível de uma caracteristica (yes)
    def occurrences_feat_values_yes(self):
        self.occurrences_feat_val_yes = {}      # dicionário que guarda número de ocorrências de cada valor possível de uma caracteristica
        
        for feat in self.features:
            for feat_val in np.unique(self.X_train[feat]):
                count=0
                rows_n = self.X_train.index[self.X_train[feat] == feat_val].tolist()
                
                for i in range(self.examples):
                    for r in rows_n:
                        if(self.y_train.loc[i] == "yes" and i == r):
                            count+=1
                #print(feat_val, ":" , count)
                self.occurrences_feat_val_yes.update({feat_val : count})    # Atualização do dicionário
        #print("yes: " ,self.occurrences_feat_val_yes)
        
    # Método que determina o número de ocorrências de cada valor possível de uma caracteristica (yes)
    def occurrences_feat_values_no(self):
        self.occurrences_feat_val_no = {}      # dicionário que guarda número de ocorrências de cada valor possível de uma caracteristica
        
        for feat in self.features:
            for feat_val in np.unique(self.X_train[feat]):
                count=0
                rows_n = self.X_train.index[self.X_train[feat] == feat_val].tolist()
                
                for i in range(self.examples):
                    for r in rows_n:
                        if(self.y_train.loc[i] == "no" and i == r):
                            count+=1
                #print(feat_val, ":" , count)
                self.occurrences_feat_val_no.update({feat_val : count})     # Atualização do dicionário
        #print("no: " , self.occurrences_feat_val_no)
            
    # Método com base no classificador previamente definido, gerar predições em função dum conjunto de dados de teste
    def predict(self,query):
        arr = []
        examples_test = len(query)
        
        if(isinstance(query,list)):
            mul_yes = self.all_prob_yes["Pyes"]
            mul_no = self.all_prob_no["Pno"]
            
            for q in query:
                for chave1 in self.all_prob_yes.keys():
                    if(chave1 == q):
                        mul_yes*= self.all_prob_yes[chave1]
                        #print(chave1,":",self.all_prob_yes[chave1])
            #print(mul_yes)
             
            for q in query:
                for chave1 in self.all_prob_no.keys():
                    if(chave1 == q):
                        mul_no*= self.all_prob_no[chave1]
                        #print(chave1,":",self.all_prob_no[chave1])
            #print(mul_no)
            
            if(mul_no > mul_yes):
                arr.append("no")
            else:
                arr.append("yes")
        else:    
            for i in range(examples_test):
                    l = query.loc[i].tolist()
                    mul_yes = self.all_prob_yes["Pyes"]
                    mul_no = self.all_prob_no["Pno"]
                    
                    for feat in l:
                        for chave in self.all_prob_yes.keys():
                            if(chave == feat):
                                mul_yes*= self.all_prob_yes[chave]
                    
                    for feat in l:
                        for chave in self.all_prob_no.keys():
                            if(chave == feat):
                                mul_no*= self.all_prob_no[chave]
                                
                    if(mul_no > mul_yes):
                        arr.append("no")
                    else:
                        arr.append("yes")
                    
        return arr
             
    # Método que retorna a exatidão dum classificador e dum conjunto de teste
    def accuracy_score(self,X,y):
        
        p = self.predict(X)
        
        accuracy = float
        count = 0
        total = len(p)
        
        for i in range(len(p)):
            if(p[i] == y.loc[i]):
                count+=1
        #print(count)  
        #print(total)          
        accuracy = count / total
        
        return accuracy
        
    # Método que retorna a precisão dum classificador e dum conjunto de teste
    def precision_score(self,X,y):
                
        r = float
        all_p = []
        p = self.predict(X)
        classes = np.unique(y)
        examples_test = len(X)
         
        for c in classes:
            c_true = c
            
            vp = 0
            fp = 0
        
            for index in range(examples_test):
                if(p[index] == y[index] == c):
                    vp+=1
                elif(p[index] == c != y[index] ):
                    fp+=1
            
            all_p.append(vp/(vp+fp))
        
        r = sum(all_p)/len(all_p)
        
        return r
            
# Main

# Decisão do valor de alpha a utilizar como também do conjunto de treino
naive_bayes = NaivebayesUevora(5.0,"breast-cancer-train.csv")       # breast-cancer-train.csv   weather-nominal.csv

# Leitura do ficheiro
naive_bayes.read()

# Divisão das caracteristicas das etiquetas
X,y = naive_bayes.split_features_target()

# fit
naive_bayes.fit(X,y)

# query1 pode representrar um ficheiro do tipo .csv ou então exclusivamente uma só query            
query1 = "breast-cancer-test.csv" 

# ["overcast","hot","normal",True] -> o valores True e False nao é para por aspas, nao e necessario (SÓ NESTES CASOS É QUE ISTO SE APLICA)
# "breast-cancer-test.csv" 
# ["no-recurrence-events","20-29","ge40","5-9","0-2","yes","3","right","right_up"]

if(isinstance(query1,list)):
    pred = naive_bayes.predict(query1)
    print("Predict: " , pred)
else:
    
    file = pd.read_csv(query1)
    X_test = file.drop([file.columns[-1]], axis = 1)
    y_test = file[file.columns[-1]]

    #print(X_test)
    #print(y_test)
    
    # accuracy
    a = naive_bayes.accuracy_score(X_test,y_test)
    print("Accuracy: " , round(a,2))
    
    X_p = file.drop([file.columns[-1]], axis = 1)
    y_p = y_test = file[file.columns[-1]]
    
    # precision
    r = naive_bayes.precision_score(X_p,y_p)
    print("Precision: " , round(r,2))
    