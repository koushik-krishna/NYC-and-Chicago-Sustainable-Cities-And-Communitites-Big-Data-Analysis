################################################################################
# Team members: Koushik Modugu, Abhishek Deshmukh, Adithya V Ganesan, Manoj Kumar
# Code Description: This file performs hypothesis testing between crime type over months vs various demograhic factors like Perp SEX, Perp Age group and Perp race to understand the demographic attribute behind perp's and the crime activity.
#The second part of the code is a prediction model which forecasts the crime rate in a particular month based on the history of unemployment. [RNN] 
# Data Pipelines used(Line number): Data files in HDFS(26), pyspark(*), pytorch(125).
# Concepts Pipelines used(Line number): RNN(125), Hypothesis Testing(33) 
# System: Single node Hadoop with spark setup in in local 
# Datasets used: NYC Historic Crime, NYC Unemployment 
################################################################################

import pyspark
import torch 
import torch.nn as nn

import numpy as np
import json
from pprint import pprint

from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

################################################################################
# Crime data file path [Large file]
input_file = 'hdfs://Data/NYC_Crime.json'
# Unemployment Data File path [Small File]
unemployment_filepath = 'Data/NYC_UnEmployment.xlsx'
# Epsilon 
e = 7./3 - 4./3 -1
# epochs
epochs = 10

################################################################################
###################### PART 1: CRIME TYPE VS DEMOGRAPHICS ######################

def findCorr(x):
    #Input format: (Crime Desc, (YEAR_MONTH, [('total', totalCasesThatMonth)], [(Age group 1, count), (Age Group 2, count),..], [(M, count), (F, count)], [(race 1, count), (race 2. count),..]))
    '''
        Returns P value and Correlation value for a particular demographics vs particular crime type.
    '''

    def performTest(x, y):
        '''
            Performs the p test.
        '''
        #Reference from https://gist.github.com/brentp/5355925 and slides
        lr = LinearRegression()
        lr.fit(x, y)
        beta = lr.coef_
        y_pred = lr.predict(x)
        dof = max(float(x.shape[0] - x.shape[1] - 1), 1.0)
        sse = np.sum((y_pred - y) ** 2, axis=0) / dof
        se = np.sqrt(sse/(np.sum((x[:, 0] - np.mean(x[:, 0]))**2)+e) ) 
        t = beta[0, 0] / (se+e)
        p = stats.t.cdf(t, x.shape[0] - x.shape[1] - 1)*n_columns if(beta[0, 0]<0) else (1 - stats.t.cdf(t, x.shape[0] - x.shape[1] - 1))*n_columns
        
        return (p[0], np.corrcoef(x.reshape(1,-1 ), y.reshape(1,-1 ))[0, 1])

    crime_desc = x[0]
    dfs = []
    for i in (x[1]):
        month = i[0]
        temp = []
        for j in range(1, len(i)):
            temp.extend(i[j])
        temp = dict(temp)
        temp["month"] = month
        dfs.append(temp)
    
    df = pd.DataFrame(dfs)
    df = df.fillna(0)
    df = df.sort_values("month", ascending=True)
    
    Y = (df['total']).values.reshape(-1,1)
    testColumns = [i for i in df.columns if i not in ['month', 'total']]
    n_columns = len(testColumns)
    results = {}
    for i in testColumns:
        x = df[i].values.reshape(-1,1)
        res = performTest(x, Y)
        if res[0]<0.05:
            results[i] = res  
        else:
            continue
    
    return (crime_desc, results)


################################################################################

sc = pyspark.SparkContext()
sqlContext = pyspark.SQLContext(sc)

txt = sc.textFile(input_file)
# Reading JSON File
txt = txt.flatMap(lambda x: (json.loads(x),))
# Picking select fields: key= (OFFENSE DESC, YEAR_MONTH)
txt = txt.flatMap(lambda x: (((x["OFNS_DESC"], x["ARREST_DATE"].split('/')[2] + '-' + x["ARREST_DATE"].split('/')[0]), 
                        (x["AGE_GROUP"], x["PERP_SEX"], x["PERP_RACE"])),)
                        )

# Grouping all the keys, listing all values
txt = txt.groupByKey().map(lambda x: (x[0], list(x[1])))

# Counting each of the Categorical variable
txt = txt.map(lambda x: (x[0], (Counter(np.array(x[1])[:,0]), Counter(np.array(x[1])[:,1]), Counter(np.array(x[1])[:,2])) )
            )
# Key=OFFENSE DESC, Val= Month, Total Crime in month, AGE Group Wise split, SEX wise split, RACE wise split
txt = txt.map(lambda x: (x[0][0], (x[0][1], [('total', sum(x[1][0].values())),], list(x[1][0].items()), list(x[1][1].items()), list(x[1][2].items())) )
            )
# Sorting it in descending order of total crimes
top_crimes = txt.map(lambda x: (x[0], list(x[1][1])[-1][-1])).reduceByKey(lambda x, y: x+y).sortBy(lambda x: -x[1]).take(10)
top_crimes = list(map(lambda x: x[0], top_crimes))

# Filtering to top 10 crimes and Sorting values based on the date
txt = txt.filter(lambda x: x[0] in top_crimes).groupByKey().map(lambda x: (x[0], sorted(list(x[1]), key=lambda y: y[0])))
#Finding significant variables
txt = txt.map(findCorr)

#Sorting based on p-value
txt = txt.map(lambda x: (x[0], sorted(list(x[1].items()), key=lambda y: y[1]) ) )
pprint (txt.collect())

sc.stop()
################################################################################
###################### PART 2: CRIME(t) = f(UNEMPLOYMENT(t)/CRIME TYPE & UNEMPLOYMENT(t-1. t-2...)) ######################

class LSTM(nn.Module):
    '''
        Defining LSTM model 
    '''
    def __init__(self, input_size=1, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def modelWithFeature(x):
    #Input: [(Year_Month, cases), (Year_Month, cases)....]
    '''
        Model a particular crime over months as a function of Unemployment using LSTM
    '''

    def callModel(train_seq, labels ):
        '''
            Train and test the LSTM model. Return MSE on test set.
        '''
        if (labels.shape[0] != train_seq.shape[0]):
            return ()
        #Scaling X features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_seq)
        
        model = LSTM()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #80-20 split for training and testing
        train_test_split = int(train_seq.shape[0]*0.8)

        #Training
        for j in range(epochs):
            loss = []
            for i in range(train_test_split):
                seq = torch.tensor(train_seq[i], dtype=torch.float32)
                label = torch.tensor(labels[i], dtype=torch.float32)
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, label)
                single_loss.backward()
                optimizer.step()        
        
        #Testing
        model.eval()
        test_pred = []
        for i in range(train_test_split,train_seq.shape[0]):
            seq = torch.tensor(train_seq[i], dtype=torch.float32)
            label = torch.tensor(labels[i], dtype=torch.float32)
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                test_pred.append(loss_function(y_pred, label).item())
        
        test_pred_error = sum(test_pred)/len(test_pred)
        
        return test_pred_error

    df = pd.DataFrame(list(x[1]), columns=['YEAR_MONTH', 'cases'])
    df['unemployment'] = -1
    for i in df.YEAR_MONTH.unique():
        df.loc[df.YEAR_MONTH==i,'unemployment'] = unemployment.value.get(i, -1)
    
    df = df.sort_values(by=['YEAR_MONTH'], ascending=True)
    y = df['cases'].values.reshape(-1,1)
    features = pd.DataFrame(df['unemployment'])
    ### Preparing the time steps.
    features['unemployment-1'] = features['unemployment'].shift(1)
    features['unemployment-2'] = features['unemployment'].shift(2)
    features['unemployment-3'] = features['unemployment'].shift(3)
    features = features.fillna(-1)
    features = features.values

    loss = callModel(features, y)

    return (x[0], loss)
################################################################################

sc = pyspark.SparkContext()
sqlContext = pyspark.SQLContext(sc)

txt = sc.textFile(input_file)
# Reading JSON File
txt = txt.flatMap(lambda x: (json.loads(x),))
# Picking select fields: key= (OFFENSE DESC, YEAR_MONTH), Value = 1
txt = txt.flatMap(lambda x: (((x["OFNS_DESC"], x["ARREST_DATE"].split('/')[2] + '-' + x["ARREST_DATE"].split('/')[0]), 1),)  )

#Filtering top crimes
txt = txt.filter(lambda x: x[0][0] in top_crimes)

#Collecting monthly crime data
txt = txt.reduceByKey(lambda x,y: x+y)

#Key = OFNS_DESC, Value= (YEAR_MONTH, Number of arrests) 
txt = txt.map(lambda x: (x[0][0], (x[0][1], x[1])))

#Read and prepare the unemployment data
unemployment = pd.read_excel(unemployment_filepath)
unemployment = unemployment.reset_index()
unemployment.columns = ['month', 'year', 'unemployment']
unemployment.month = (unemployment.month+1)%12
unemployment.loc[unemployment.month==0, 'month'] = 12
unemployment['YEAR_MONTH'] = unemployment.year.astype(str) + '-' + unemployment.month.map("{:02}".format).astype(str)
unemployment = unemployment[['YEAR_MONTH', 'unemployment']]
#Broadcasting Unemployment data
unemployment = sc.parallelize(unemployment.values.tolist())
unemployment = sc.broadcast(dict(unemployment.collect()))

# Model using LSTM
txt = txt.groupByKey().map(modelWithFeature)

pprint(txt.collect())

sc.stop()

################################################################################