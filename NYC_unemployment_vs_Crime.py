################################################################################
# Team members: Koushik Modugu, Abhishek Deshmukh, Adithya V Ganesan, Manoj Kumar
# Code Description: This is an analysis between NYC Crime Rate - By Type  and NYC Unemployment rate. We perform Hypothesis Testing to validate if there is correlation between Crime and Unemployment
#                   Prediction is done for each of the crime type against unemployment rate
# Pipelines used(Line number): Data files in pyspark(*)
# Concepts used(Line number): HDFS(77), Hypothesis Testing(112)
# System: Single node Hadoop with spark setup in in local
# Datasets:  NYC Crime Dataset, NYC Unemployment Dataset
################################################################################


from pyspark.sql import SparkSession
import math
from scipy import stats
import numpy as np
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as f
import pandas as pd
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName("correaltion").getOrCreate()
sc=spark.sparkContext



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

def slope_significance_hyp_testing(X, y, y_pred, coefficients,correlation):
    df = len(X) - (len(coefficients) + 1)
    rss = np.sum((y_pred-y)**2)
    x_j_diff_squares = np.sum((X-np.mean(X, axis=0).reshape(1,-1))**2, axis=0)
    standard_error = compute_std_error(x_j_diff_squares, rss, df)
    t_vals = np.divide(coefficients, standard_error)
    # Testing significance of all coefficients of X, so correction is division by no of hypotheses
    bonferroni_corrected_alpha = 0.05/len(coefficients)
    p_values = eval_p_value(t_vals, df)

    print("(Type,Correlation,Pvalue)",(correlation[0],correlation[1],p_values))
    significant_coeffs = check_slope_significance(p_values, bonferroni_corrected_alpha)
    print("significance : "+ str(significant_coeffs[0]))
    return significant_coeffs

####### Creating index/key for Unemployment data #######

# ------------ input file unemployment#
dataDF=pd.read_excel("Data/NYC_UnEmployment.xlsx");
# ------------ #
dataDF["Month"]=dataDF.index%12+1
dataDF["Date"]=(dataDF["Month"].astype(str)+"-"+dataDF["Year"].astype(str)).apply(lambda x:x.zfill(7))
dataDF=spark.createDataFrame(data=dataDF)

####### grouping Unemployment data by crime type #######

# ------------ input file nyc arrests#
dataNY=spark.read.format("csv").option("header", "true").load("hdfs://Data/NYPD_Arrests_Data__Historic_.csv")
# ------------ #
key=f.split(dataNY.ARREST_DATE,"/")
dataNY=dataNY.withColumn("key",f.concat(key.getItem(0),f.lit("-"),key.getItem(2)))
types=dataNY.toPandas()["OFNS_DESC"].sort_values().unique()
groupeddataNY=dataNY.groupBy(dataNY.key,dataNY.OFNS_DESC).agg(f.count(f.lit(1)).alias("count"))
NYdf=groupeddataNY.groupBy("key").pivot("OFNS_DESC").sum("count").fillna(0).sort("key")

####### merging unemployment data and Crime data by month Year as key #######
mergedDfTotal=dataDF.join(NYdf, dataDF.Date == NYdf.key , 'inner' ).drop("Year","Date","Month","Unemplyment").sort("key").drop("key")

####### Reaplcing spaces and . in columns names #######
for col in mergedDfTotal.columns:
    mergedDfTotal=mergedDfTotal.withColumnRenamed(col, col.replace(".","").replace(" ",""))
columns=mergedDfTotal.schema.names

####### Picking out all columns and Calculating correlation #######
correlation=[]
for col in columns[1:]:
    correlation.append(((col,mergedDfTotal.corr('UnemplymentRate',col))))
    print(correlation[-1])

total_columns=[]

correlation=sorted(correlation,key=lambda x: x[1],reverse=True)
for col,val in correlation:
    total_columns.append(col)


# Top 10 correlated Crime Types= ['OTHERTRAFFICINFRACTION', 'VEHICLEANDTRAFFICLAWS', 'CRIMINALTRESPASS', 'DANGEROUSDRUGS', 'INTOXICATED&IMPAIREDDRIVING', 'OTHEROFFENSESRELATEDTOTHEFT', 'THEFT-FRAUD', 'GRANDLARCENY', 'OTHERSTATELAWS', 'PARKINGOFFENSES']
####### Linear regression to validate Hypothesis Testing between every crime type and unemployment rate #######

for i in range(len(total_columns)):
    columns=[total_columns[i]]
    mergedDf=mergedDfTotal.select(['UnemplymentRate']+ columns)
    ####### Validated results with and without normalisng data per crime, rss seems to do better without normalising so ignoring normalizing using MinMaxScaler#######
    assembler = VectorAssembler(inputCols=['UnemplymentRate'], outputCol="features")
    vgrouped_arrests_unemp = assembler.transform(mergedDf)
    lr = LinearRegression(featuresCol = 'features', labelCol=columns[0],
                          maxIter=200, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(vgrouped_arrests_unemp)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    vgrouped_arrests_unemp.describe().show()
    lr_predictions = lr_model.transform(vgrouped_arrests_unemp)
    output = np.array(lr_predictions.select(["prediction"]+['UnemplymentRate']+columns).collect())
    X, y, y_pred = output[:, 2].reshape(-1, 1), output[:, 1].reshape(-1, 1), output[:,0].reshape(-1, 1)
    coefficient_significance = slope_significance_hyp_testing(X, y, y_pred, np.array(lr_model.coefficients).reshape(-1,1),correlation[i])
    for j in range(len(coefficient_significance)):
        if coefficient_significance[j] is True:
            print("Accepting the Null hypothesis. UnemploymentRate cant be used to predict "+ columns[0] )
        else:
            print("Rejecting the Null hypothesis. UnemploymentRate can be used to predict "+ columns[0] )
    print("------------#-------------")

