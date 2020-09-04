

################################################################################
# Team members: Koushik Modugu, Abhishek Deshmukh, Adithya V Ganesan, Manoj Kumar
# Code Description: This is an analysis NYC Housing data and Overall Crime Count , We perform Hypothesis Testing to validate if there is correlation between Crime and residential location.
# Pipelines used(Line number): Data files in pyspark(*)
# Concepts used(Line number): HDFS(96,98), Hypothesis Testing(49)
# System: Single node Hadoop with spark setup in in local
# Datasets : NYC Crime dataset, NYC Housing Dataset
################################################################################

from pyspark.sql import SparkSession
import math
from scipy import stats
import numpy as np
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as f
import pandas as pd
from uszipcode import SearchEngine
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName("correaltion").getOrCreate()
sc=spark.sparkContext
search = SearchEngine(simple_zipcode=True)
####### Methods for computations involved with hypothesis testing #######
def compute_std_error(x_j_diff_square, rss, df):
    try:
        return math.pow(rss / (df * (x_j_diff_square + 1e9)), 0.5)
    except:
        print(x_j_diff_square, rss, df)

def eval_p_value(t_value, df):
    result=[]
    for val in t_value:
        if val > 0.5:
            result.append( 2*(1 - stats.t.cdf(val,df=df)))
        else:
            result.append( 2*stats.t.cdf(val,df=df))
    return result

def check_slope_significance(p_values, bonferroni_corrected_alpha):
    return_list = list()
    # if True (p_value > bonferroni_corrected_alpha), accept the null hypothesis that the slope is insignificant
    for p_value in p_values:
        return_list.append(True if p_value > bonferroni_corrected_alpha else False)
    return return_list

def slope_significance_hyp_testing(X, y, y_pred, coefficients):
    df = len(X) - (len(coefficients) + 1)
    rss = np.sum((y_pred-y)**2)
    x_j_diff_squares = np.sum((X-np.mean(X, axis=0).reshape(1,-1))**2, axis=0)
    standard_error = compute_std_error(x_j_diff_squares, rss, df)
    t_vals = np.divide(coefficients, standard_error)
    # Testing significance of all coefficients of X, so correction is division by no of hypotheses
    bonferroni_corrected_alpha = 0.05/len(coefficients)
    p_values = eval_p_value(t_vals, df)
    print("p_values : " ,p_values)
    significant_coeffs = check_slope_significance(p_values, bonferroni_corrected_alpha)
    print("significance : "+ str(significant_coeffs[0]))
    return significant_coeffs

def calculate_zip(row):
    return search.by_coordinates(row['Latitude'] ,row['Longitude'],returns=1)[0].zipcode


####### Calculation of zipcodes based on lat and long for NY Crime data#######
nyArrests=pd.read_csv("Data/NYPD_Arrests_Data__Historic_.csv")
nycolumns=['OFNS_DESC','Latitude','Longitude']
nyArrests=nyArrests[nycolumns]
nyArrests=nyArrests.dropna(subset=['Latitude','Longitude'])
nyArrests["zipcodeNY"]=nyArrests.apply(calculate_zip, axis=1)
nyArrests=nyArrests.drop(columns=["Latitude","Longitude"])
nyArrests.to_csv("Data/nyCrime_By_Zip.csv",index=False)



####### Calculation of zipcodes based on lat and long for NY housing data  #######
dataHousing=pd.read_csv("Data/Housing_New_York_Units_by_Building.csv");
columns=['Latitude','Longitude','Extremely Low Income Units','Very Low Income Units','Low Income Units','Moderate Income Units','Middle Income Units']
dataHousing=dataHousing[columns]
dataHousing=dataHousing.dropna(subset=['Latitude','Longitude'])

#######
# Used a custom metric call housing values based on the no of types of income units available
# Formula : ((Type 1)*(avg of range of Type 1) +....)/(Total Count)
# avg range is calculated from income range of that type , for example Extreme low income range is from 0-30 , so weight for this type is considered as 0.15
# #######
dataHousing["HousingValue"]=((dataHousing['Extremely Low Income Units']*0.15) + (dataHousing['Very Low Income Units']*0.4) +(dataHousing['Low Income Units']*0.7 )+(dataHousing['Moderate Income Units']*1)+(dataHousing['Middle Income Units']*1.5))/dataHousing[columns[2:]].sum(axis =1)
dataHousing["Zipcode"]=dataHousing.apply(calculate_zip, axis=1)
dataHousing=dataHousing.drop(columns=["Latitude","Longitude","Extremely Low Income Units","Very Low Income Units","Low Income Units","Moderate Income Units","Middle Income Units"])
dataHousing.to_csv("Data/HousingNYC_By_Zip.csv",index=False)


####### Group by zipcodes Aggreagting data and merging crime , housing data  #######
nyArrests=spark.read.format("csv").option("header", "true").load("hdfs://Data/nyCrime_By_Zip.csv")
nyArrests=nyArrests.groupBy("zipcodeNY").agg(f.count(f.lit(1)).alias("count"))
dataHousing=spark.read.format("csv").option("header", "true").load("hdfs://Data/HousingNYC_By_Zip.csv")
dataHousing=dataHousing.groupBy("Zipcode").agg(f.sum("HousingValue").alias("HousingValue")).sort("Zipcode")

mergedDf=dataHousing.join(nyArrests,nyArrests.zipcodeNY==dataHousing.Zipcode,'inner').drop("zipcodeNY")

####### Correaltion for housing value and Crime count based on zipcodes  #######
print("Correaltion for housing value and Crime count : ",mergedDf.corr('count','HousingValue'))

####### Linear regression to validate Hypothesis Testing (Concept 1)#######
assembler = VectorAssembler(inputCols=['HousingValue'], outputCol="features")
vgrouped_arrests_unemp = assembler.transform(mergedDf)
lr = LinearRegression(featuresCol = 'features', labelCol='count',
                      maxIter=200, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(vgrouped_arrests_unemp)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
vgrouped_arrests_unemp.describe().show()
lr_predictions = lr_model.transform(vgrouped_arrests_unemp)
output = np.array(lr_predictions.select(["prediction"]+['HousingValue']+['count']).collect())
X, y, y_pred = output[:, 2].reshape(-1, 1), output[:, 1].reshape(-1, 1), output[:,0].reshape(-1, 1)
coefficient_significance = slope_significance_hyp_testing(X, y, y_pred, np.array(lr_model.coefficients).reshape(-1,1))
for i in coefficient_significance:
    if coefficient_significance[i] is True:
        print("Accepting the Null hypothesis. Housing value isn't a good predictor of NumberOfArrests")
    else:
        print("Rejecting the Null hypothesis. HousingValue is actually a good predictor of NumberOfArrests")

# Motivation for this analysis is to validate if the crimes pertaining to an area is related to residential location. Results indicate that a correlation exists .
