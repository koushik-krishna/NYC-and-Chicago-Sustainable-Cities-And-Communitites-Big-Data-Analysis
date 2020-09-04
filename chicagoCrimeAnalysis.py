################################################################################
# Team members: Koushik Modugu, Abhishek Deshmukh, Adithya V Ganesan, Manoj Kumar
# Code Description: This file performs data analysis between crimes in Chicago with various other supplementing 
# information such as unemployment, Income and housing data from Chicago. We furthiur perform hypothesis testing to 
# understand the significance of the correlation.
# Piplines Used: HDFS(42), PySpark(*) 
# Concepts Used : Hypothesis Testing(100)
# System: Single node Hadoop with spark setup in in local 
# Datasets: Chicago Historic Arrests since 2001 & Chicago Unemployment, Housing and Income datasets.
# Note: This file was originally an ipynb. The html format of this file [with the results and output] is present in it.
################################################################################

import os
import json

import findspark
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-preview2-bin-hadoop3.2"
findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


import math
from operator import add
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
from pyspark.sql.window import Window
plt.style.use('seaborn-whitegrid')

spark = SparkSession.builder.appName("Chicago").config("spark.driver.memory", "20g").getOrCreate()


# get chicago arrest data from the json files
arrest_df = spark.read.json("hdfs://Crimes_-_2001_to_present0.json")
for i in range(1, 10):
    arrest_df = arrest_df.union(spark.read.json("hdfs://Crimes_-_2001_to_present"+str(i)+".json"))
#format based on timestamp
arrest_df = arrest_df.withColumn("Date", F.from_unixtime(F.unix_timestamp("Date",'MM/dd/yyyy hh:mm:ss a')))
arrest_df = arrest_df.withColumn("Updated On", F.from_unixtime(F.unix_timestamp("Updated On",'MM/dd/yyyy hh:mm:ss a')))
#filter data based on arrest
actual_arrest_df = arrest_df.filter(arrest_df.Arrest==1)
arrest_df.head(4)

# # Crime and Unemployment Data Analysis

#get unemployment data [smaller data]
unemp_df = spark.read.csv('/content/drive/My Drive/ChicagoUnemployment.csv',header=True)
unemp_df = unemp_df.withColumnRenamed("Value", "UnemploymentRate")
unemp_df = unemp_df.filter(unemp_df.Year > 2010)
#format based on timestamp
unemp_df = unemp_df.withColumn("Label", F.from_unixtime(F.unix_timestamp("Label",'yyyy MMM')))
unemp_df = unemp_df.withColumn('month', F.substring(unemp_df['Period'],2,2).cast(IntegerType()))
unemp_df = unemp_df.withColumn('UnemploymentRate', unemp_df['UnemploymentRate'].cast(DoubleType()))
unemp_df.head(4)

# actual_arrest_df = arrest_df.filter(arrest_df.Arrest==1)
# get aggregated count
grouped_arrests = actual_arrest_df.groupBy([F.month('Date').alias('month'),F.year('Date').alias('year')]).          agg(F.count('Arrest').alias('Case Count'))
# join the arrest and the unemployment data
grouped_arrests_unemp = grouped_arrests.join(unemp_df, (grouped_arrests.month ==                        unemp_df.month) & (grouped_arrests.year == unemp_df.Year))
grouped_arrests_unemp = grouped_arrests_unemp.select(['Case Count', 'UnemploymentRate'])

from pyspark.ml.feature import VectorAssembler
# All UnemploymentRate into a vector "features"
vectorAssembler = VectorAssembler(inputCols = ['UnemploymentRate'], outputCol = 'features')
vgrouped_arrests_unemp = vectorAssembler.transform(grouped_arrests_unemp)

grouped_arrests_unemp.corr('Case Count', 'UnemploymentRate')


# The correlation between UnemploymentRate and NumberOfArrests is 0.88 when the data after 2010 is considered. Interestingly, if we consider data since 2001, the correlation turns out to be 0.24, which implies that from 2001-2010, there were other factors at play. For example, **unemployment rate raised from 2008-2010 due to recession and proportional fraction of the unemployed didn't resort to crime to meet the ends.**



def compute_std_error(x_j_diff_square, rss, df):
    return math.pow(rss/(df*x_j_diff_square), 0.5)

def eval_p_value(t_value, df):
    result = []
    for val in t_value:
        if val > 0.5:
            result.append(2*(1 - stats.t.cdf(val,df=df)))
        else:
            result.append(2*stats.t.cdf(val,df=df))
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
    # print(t_vals, df, p_values)
    significant_coeffs = check_slope_significance(p_values, bonferroni_corrected_alpha)
    return significant_coeffs




from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='Case Count',
                      maxIter=200, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(vgrouped_arrests_unemp)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
vgrouped_arrests_unemp.describe().show()
lr_predictions = lr_model.transform(vgrouped_arrests_unemp)
output = np.array(lr_predictions.select("prediction","Case Count","UnemploymentRate").collect())
X, y, y_pred = output[:, 2].reshape(-1, 1), output[:, 1].reshape(-1, 1), output[:,0].reshape(-1, 1)
coefficient_significance = slope_significance_hyp_testing(X, y, y_pred, np.array(lr_model.coefficients).reshape(-1,1))
if coefficient_significance[0] is True:
    print("Accepting the Null hypothesis that the coefficient of UnemploymentRate in NumberOfArrests prediction is insignificant. UnemploymentRate isn't a good predictor of NumberOfArrests")
else:
    print("Rejecting the Null hypothesis that the coefficient of UnemploymentRate in NumberOfArrests prediction is insignificant. UnemploymentRate is actually a good predictor of NumberOfArrests")

# From 2017, for all the arrests made the following IUCR codes are the highest Crime types committed.
highest_crime_df = actual_arrest_df.filter(actual_arrest_df.Year>=2017).                                groupBy(['IUCR']).count().sort(F.desc("count")).limit(7)
highest_crime_df = highest_crime_df.withColumnRenamed("count", "TotalCrimeTypeCount")

actual_arrest_df_2017 = actual_arrest_df.filter(actual_arrest_df.Year >= 2017)
top_crime_actual_arrest_df_2017 = actual_arrest_df_2017.join(highest_crime_df, 'IUCR')

community_crime_wise_info = top_crime_actual_arrest_df_2017.groupBy(['IUCR', 'Community Area', 'TotalCrimeTypeCount']).count()
community_crime_wise_info = community_crime_wise_info.withColumn("TotalFractionOfCrimeInCommuntiy",                                                                 F.col('count')/F.col('TotalCrimeTypeCount'))

window = Window.partitionBy(community_crime_wise_info['IUCR']).orderBy(community_crime_wise_info['TotalFractionOfCrimeInCommuntiy'].desc())

top_community_crime_wise_info = community_crime_wise_info.select('*', F.rank().over(window).alias('rank')).filter(F.col('rank') <= 3)
top_community_crime_wise_info.show()

top_communities = top_community_crime_wise_info.groupBy(['Community Area']).count()
top_communities = top_communities.withColumnRenamed("count", "Number of Crime Types the Community is ranked in top 3")
top_communities.show()


# # Crime and Income Data Analysis

# Read Chicago Income data using spark [smaller file]
income_df = spark.read.csv('/content/drive/My Drive/HouseholdIncomeChicago.csv',header=True)
income_df =  income_df[income_df["Geography"]=="Chicago, IL"]
# Share is the % of people from the city of chicago who fall in a particular income bucket.
income_df = income_df.withColumn('share', income_df['share'].cast(DoubleType()))
income_df.head(4)

# get Yearwise crimes for Chicago
yearwise_grouped_arrests = actual_arrest_df.groupBy([F.year('Date').alias('year')]).agg(F.count('Arrest').alias('Case Count'))

# join it with the yearwise income data
yearwise_grouped_arrests_income = yearwise_grouped_arrests.join(income_df, (yearwise_grouped_arrests.year == income_df.Year))

# get realtionship between crimes and the income bucket share.
# bucket = 0, meaning income < $10,000
# bucket = 15, meaning income is $200,000+
corrs_list = []
for bucket in range(16):
  bucket = str(bucket)
  bucket_df = yearwise_grouped_arrests_income[yearwise_grouped_arrests_income["ID Household Income Bucket"] == bucket ]
  bucket_new_df = bucket_df.select(['Case Count', 'share'])
  corr = bucket_new_df.corr('Case Count', 'share')
  corrs_list.append(corr)


plt.figure(figsize=(10,6))
plt.plot(corrs_list)
plt.title("Correlation between crimes and Income Bucket")
plt.ylabel('Correlation with Crime')
plt.xlabel('Income Bucket')
plt.show()

yearwise_grouped_arrests_income.head(3)

economically_weaker_df = yearwise_grouped_arrests_income[yearwise_grouped_arrests_income["ID Household Income Bucket"] == "0" ]
economically_weaker_new_df = economically_weaker_df.select(['Case Count', 'share'])

from pyspark.ml.feature import VectorAssembler
# All poor buckets share into a vector "features"
vectorAssembler = VectorAssembler(inputCols = ['share'], outputCol = 'features')
vgrouped_arrests_bucket = vectorAssembler.transform(economically_weaker_new_df)

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='Case Count',
                      maxIter=100, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(vgrouped_arrests_bucket)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

lr_predictions = lr_model.transform(vgrouped_arrests_bucket)
output = np.array(lr_predictions.select("prediction","Case Count","share").collect())

y_pred = output[:,0].reshape(-1, 1)
y = output[:, 1].reshape(-1, 1) 
X = output[:, 2].reshape(-1, 1)

coefficient_significance = slope_significance_hyp_testing(X, y, y_pred, np.array(lr_model.coefficients).reshape(-1,1))
if coefficient_significance[0] is True:
    print("Can't reject the null hupothesis that the coefficient of Income in NumberOfArrests prediction is insignificant. Thus Income isn't a good predictor of NumberOfArrests")
else:
    print("Rejecting the Null hypothesis. Thus Income is actually a good predictor of NumberOfArrests")
    

domestic_df = actual_arrest_df[actual_arrest_df["Domestic"] == True]
domestic_df.head(2)

# group by year
domestic_grouped_arrests = domestic_df.groupBy([F.year('Date').alias('year')]).          agg(F.count('Arrest').alias('Case Count'))
# join with income data
domestic_grouped_arrests_income = domestic_grouped_arrests.join(income_df, (domestic_grouped_arrests.year ==                        income_df.Year))
# checking the domestic crimes amongst economically weaker section
domestic_bucket_df = domestic_grouped_arrests_income[domestic_grouped_arrests_income["ID Household Income Bucket"] == "0" ]
domestic_bucket_new_df = domestic_bucket_df.select(['Case Count', 'share'])
domestic_bucket_new_df.head(2)

corrs_list = []
for bucket in range(0,16,3):
  bucket = str(bucket)
  domestic_bucket_df = domestic_grouped_arrests_income[domestic_grouped_arrests_income["ID Household Income Bucket"] == bucket ]
  domestic_bucket_new_df = domestic_bucket_df.select(['Case Count', 'share'])
  corr = domestic_bucket_new_df.corr('Case Count', 'share')
  corrs_list.append(corr)



plt.figure(figsize=(10,6))
plt.plot(corrs_list)
plt.title("Correlation between Domestic crimes and income bucket")
plt.ylabel('Correlation with Domestic crimes')
plt.xlabel('Income Bucket of 3')
plt.show()


low_income_bucket = []
high_income_bucket = []
actual_arrest_df = actual_arrest_df.withColumn('IUCR', actual_arrest_df['IUCR'].cast(IntegerType()))
#homicide = 110 - 142
#sexual_assault = 261 - 291
#gambling 1610 - 1697
#[110,142]
crimes = [[261,291],[1610,1697]]
for crime in crimes:
  start_iucr,end_iucr = crime
  s_df = actual_arrest_df.filter(actual_arrest_df.IUCR >= start_iucr)
  crime_df = s_df.filter(s_df.IUCR <= end_iucr)

  # group by year
  crime_grouped_df = crime_df.groupBy([F.year('Date').alias('year')]).  agg(F.count('Arrest').alias('Case Count'))
  
  # join with income data
  crime_grouped_arrests_income = crime_grouped_df.join(income_df, (crime_grouped_df.year ==                        income_df.Year))
  
  # checking the crimes for the buckets
  crime_bucket_df = crime_grouped_arrests_income[crime_grouped_arrests_income["ID Household Income Bucket"] == "0" ]
  crime_bucket_new_df = crime_bucket_df.select(['Case Count', 'share'])
  low_income_bucket.append(crime_bucket_new_df.corr('Case Count', 'share'))

  crime_bucket_df = crime_grouped_arrests_income[crime_grouped_arrests_income["ID Household Income Bucket"] == "15" ]
  crime_bucket_new_df = crime_bucket_df.select(['Case Count', 'share'])
  high_income_bucket.append(crime_bucket_new_df.corr('Case Count', 'share'))


# correlation for low income bucket for homicide, sexual_assault and gambling
print (low_income_bucket)


# correlation for high income bucket for homicide, sexual_assault and gambling
print (high_income_bucket)

# Get chicago Housing data
housing_df = spark.read.csv('/content/drive/My Drive/Chicago_Housing.csv',header=True)
housing_df.head(2)

# Get data for new constructions only
new_housing_df = housing_df[housing_df["PERMIT_TYPE"] == "PERMIT - NEW CONSTRUCTION"] 
new_housing_df.head(3)

# Type cast reported cost of the new construction
new_housing_df = new_housing_df.withColumn('REPORTED_COST', new_housing_df['REPORTED_COST'].cast(DoubleType()))
# group by community area and get the mean of the new construction. This acts as a proxy for how good a locality this is.
area_cost_df = new_housing_df.groupBy("COMMUNITY_AREA").agg(F.mean('REPORTED_COST'))
# get crimes per community area
area_grouped_arrests = actual_arrest_df.groupBy("Community Area").agg(F.count('Arrest').alias('Case Count'))

# join the comm
grouped_area_arrests_cost = area_grouped_arrests.join(area_cost_df, (area_grouped_arrests["Community Area"]== area_cost_df["COMMUNITY_AREA"]))
grouped_area_arrests_cost.corr('Case Count', 'avg(REPORTED_COST)')

actual_arrest_df = actual_arrest_df.withColumn('IUCR', actual_arrest_df['IUCR'].cast(IntegerType()))
theft_df = actual_arrest_df.filter(actual_arrest_df.IUCR > 610)
theft_and_burglary_df = theft_df.filter(theft_df.IUCR < 895)
theft_and_burglary_areagrouped_df = theft_and_burglary_df.groupBy("Community Area").agg(F.count('Arrest').alias('theft burglary Count'))
theft_and_burglary_areagrouped_df.head(2)

grouped_area_arrests_cost = theft_and_burglary_areagrouped_df.join(area_cost_df, (theft_and_burglary_areagrouped_df["Community Area"]==                        area_cost_df["COMMUNITY_AREA"]))
grouped_area_arrests_cost.corr('theft burglary Count', 'avg(REPORTED_COST)')




