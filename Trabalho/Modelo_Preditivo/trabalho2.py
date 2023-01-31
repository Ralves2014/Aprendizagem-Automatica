import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


# Classe que define um modelo 
class algorithm:
    def __init__(self, model_name, model_pred):
        self.algorithm = model_name
        self.pred = model_pred
        
    def name(self):
        return self.algorithm
    
    def pred(self):
        return self.pred



# Modelo Preditivo
class Modelo:
    
    # Construtor que define o ficheiro que contem o conjunto de dados
    def __init__(self):
        self.data_file = "dropout-trabalho2.csv"
    
    # Método que faz a leitura do conjunto de dados    
    def read(self):
        self.dataset = pd.read_csv(self.data_file)
    
    # Método que faz a separação do conjunto de dados
    def split_dataset(self): 
        X = self.dataset.drop(['Failure'],axis=1)
        y = self.dataset['Failure']
        self.X_train, X_test, self.y_train, y_test = train_test_split(X,y,test_size=0.2)
        test = pd.read_csv("dropout-test.csv")
        self.y_test = test['Failure']
        
    # Modelos utilizados -> KNN, DecisionTree, GaussianNB, LogisticRegression,
    #                       GradientBoosting, Random_Forest, DummyClassifier, modelo alternativo simplificado
    
    # Método que retorna o predict do Modelo KNN
    def pred_model_KNN(self):
        Knn = KNeighborsClassifier(n_neighbors=2)
        Knn.fit(self.X_train,self.y_train)
        y_pred_KNN = Knn.predict(self.X_test)
        return y_pred_KNN
    
    # Método que retorna o predict do Modelo DecisionTree
    def pred_model_DecisionTree(self):
        Dt = DecisionTreeClassifier()
        Dt.fit(self.X_train,self.y_train)
        y_pred_DT = Dt.predict(self.X_test)
        return y_pred_DT
        
    # Método que retorna o predict do Modelo GaussianNB
    def pred_model_GaussianNB(self):
        Gnb = GaussianNB()
        Gnb.fit(self.X_train,self.y_train)
        y_pred_GNB = Gnb.predict(self.X_test)
        return y_pred_GNB
    
    # Método que retorna o predict do Modelo LogisticRegression
    def pred_model_LogisticRegression(self):
        Lr = LogisticRegression()
        Lr.fit(self.X_train,self.y_train)
        y_pred_LR = Lr.predict(self.X_test)
        return y_pred_LR
    
    # Método que retorna o predict do Modelo GradientBoosting
    def pred_model_GradientBoosting(self):
        Gbc = GradientBoostingClassifier(n_estimators=200)
        Gbc.fit(self.X_train,self.y_train)
        y_pred_GBC = Gbc.predict(self.X_test)
        return y_pred_GBC
    
    # Método que retorna o predict do Modelo Random_Forest
    def pred_model_Random_Forest(self):
        Rf = RandomForestClassifier(n_estimators=200,max_features=10)
        Rf.fit(self.X_train,self.y_train)
        y_pred_RF = Rf.predict(self.X_test)
        return y_pred_RF
    
    # Método que retorna o predict do Modelo DummyClassifier
    def pred_model_DummyClassifier(self):
        Dummy_clf = DummyClassifier(strategy="uniform")
        Dummy_clf.fit(self.X_train,self.y_train)
        y_pred_DC = Dummy_clf.predict(self.X_test)
        return y_pred_DC
    
    # Método que retorna o predict do modelo alternativo simplificado
    def pred_model_RandomForest_2atributes(self):
        average_train ={}
        average_train['Id'] = self.dataset['Id']
        average_train['Program'] = self.dataset['Program']
        average_train['media'] = (self.dataset['Y1s1_grade']+self.dataset['Y1s2_grade']+ \
                        self.dataset['Y2s1_grade']+self.dataset['Y2s2_grade']+ \
                        self.dataset['Y3s1_grade']+self.dataset['Y3s2_grade']+ \
                        self.dataset['Y4s1_grade']+self.dataset['Y4s2_grade'])/8
        
        average_test ={}
        average_test['Id'] = self.X_test['Id']
        average_test['Program'] = self.X_test['Program']
        average_test['media'] = (self.X_test['Y1s1_grade']+self.X_test['Y1s2_grade']+ \
                        self.X_test['Y2s1_grade']+self.X_test['Y2s2_grade']+ \
                        self.X_test['Y3s1_grade']+self.X_test['Y3s2_grade']+ \
                        self.X_test['Y4s1_grade']+self.X_test['Y4s2_grade'])/8
        
        
        new_X_train = pd.DataFrame.from_dict(average_train,orient='columns')
        new_X_test = pd.DataFrame.from_dict(average_test,orient='columns')
        y = self.dataset['Failure']
        X_train, X_test, y_train, y_test = train_test_split(new_X_train,y,test_size=0.2)
        
     
        Rf2 = RandomForestClassifier(n_estimators=500,max_features=10)
        Rf2.fit(X_train,y_train)
        y_pred_RF_2_atributes = Rf2.predict(new_X_test)
        
        return y_pred_RF_2_atributes
        
        
    # Método que retorna a prediction do melhor modelo (precision >= 0.7 & max recall)
    def best_model(self):
        
        First_Verification = []
        Second_Verification = []
        
        pred_KNN = self.pred_model_KNN()
        pred_DT = self.pred_model_DecisionTree()
        pred_GNB = self.pred_model_GaussianNB()
        pred_LR = self.pred_model_LogisticRegression()
        pred_GBC = self.pred_model_GradientBoosting()
        pred_RF = self.pred_model_Random_Forest()
        pred_DC = self.pred_model_DummyClassifier()
        pred_RF_2atr = self.pred_model_RandomForest_2atributes()
          
        
        # São definidos objetos(modelos) do tipo algorithm
        model1 = algorithm("KNN",pred_KNN)
        model2 = algorithm("DecisionTree",pred_DT)
        model3 = algorithm("GaussianN",pred_GNB)
        model4 = algorithm("LogisticRegression",pred_LR)
        model5 = algorithm("GradientBoosting",pred_GBC)
        model6 = algorithm("Random_Forest",pred_RF)
        model7 = algorithm("DummyClassifier",pred_DC)
        model8 = algorithm("RandomForest2atrb",pred_RF_2atr)
        
        # Saõ adicionados todos os modelos(objetos) ao array First_Verification
        First_Verification.append(model1)
        First_Verification.append(model2)
        First_Verification.append(model3)
        First_Verification.append(model4)
        First_Verification.append(model5)
        First_Verification.append(model6)
        First_Verification.append(model7)
        First_Verification.append(model8)
           
        # Neste ciclo for são adicionados todos os modelos
        # com uma presisão >= 0.7 ao array Second_Verification
        for x in range(len(First_Verification)):
            precision = precision_score(self.y_test,First_Verification[x].pred, zero_division=True)
            #print(First_Verification[x].name(), ": ", precision_score(self.y_test,First_Verification[x].pred, zero_division=True))
            if(precision >= 0.7 and precision != 1.0):
                Second_Verification.append(First_Verification[x])
        
        r = 0.0
        
        #print("----------------------------");
        
        # Neste ciclo é definido qual o melhor modelo, ou seja, o que possui uma cobertura 
        # superior em relação aos modelos que passaram na primeira avaliação (precision > 0.7) 
        for x in range(len(Second_Verification)):
            recall = recall_score(self.y_test,Second_Verification[x].pred)
            #print(Second_Verification[x].name(), ":", recall)
            if(r < recall):
                r = recall
                #print(Second_Verification[x].name(), ":", recall)
                best = Second_Verification[x].pred
        
        return best
    
    # Método que retorna o predict do melhor modelo
    def predict(self,X):
        self.X_test = X
        y_pred_best_model = self.best_model()
        
        return y_pred_best_model
        
        
 
modelo = Modelo()
modelo.read()
modelo.split_dataset()
