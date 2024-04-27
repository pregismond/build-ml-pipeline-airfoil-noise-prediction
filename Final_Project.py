#!/usr/bin/env python

"""
SCRIPT: Final_Project.py
AUTHOR: Pravin Regismond
DATE: 2024-04-24
DESCRIPTION: Build a Machine Learning Pipeline for Airflow Noise Prediction

AUDIT TRAIL START                               INIT  DATE
----------------------------------------------  ----- -----------
1. Initial version                              PR    2024-04-24

AUDIT TRAIL END
"""

# Importing the required libraries
import os
import findspark
import warnings


def warn(*args, **kwargs):
    pass


# Suppress generated warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

findspark.init()

# Import functions/Classes for sparkml
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression

# Import functions/Classes for pipeline creation
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel

# Import functions/Classes for metrics
from pyspark.ml.evaluation import RegressionEvaluator

# Create a SparkSession
# Ignore any warnings by SparkSession command
spark = SparkSession \
    .builder \
    .appName("Airfoil Noise Prediction") \
    .getOrCreate()


# Part 1 - PERFORM ETL ACTIVITY
# Load the dataset that you have downloaded in the previous task
df = spark.read.csv("NASA_airfoil_noise_raw.csv", header=True, inferSchema=True)

# Show top 5 rows of dataset
df.show(5)

# Show total number of rows in the dataset
rowcount1 = df.count()
print(rowcount1)

# Remove duplicates, if any
df = df.dropDuplicates()

# Show total number of rows in the dataset
rowcount2 = df.count()
print(rowcount2)

# Drop rows with null values, if any
df = df.dropna()

# Show total number of rows in the dataset
rowcount3 = df.count()
print(rowcount3)

# Rename column "SoundLevel" to "SoundLevelDecibels"
df = df.withColumnRenamed("SoundLevel", "SoundLevelDecibels")

# Store the cleaned data in parquet format, name the file as
# "NASA_airfoil_noise_cleaned.parquet"
df.write.parquet("NASA_airfoil_noise_cleaned.parquet")

# PART 1 - EVALUATION
print("Part 1 - Evaluation")

print("Total rows = ", rowcount1)
print("Total rows after dropping duplicate rows = ", rowcount2)
print("Total rows after dropping duplicate rows and rows with null values = ", rowcount3)
print("New column name = ", df.columns[-1])

print("NASA_airfoil_noise_cleaned.parquet exists :", os.path.isdir("NASA_airfoil_noise_cleaned.parquet"))


# PART 2 - CREATE A MACHINE LEARNING PIPELINE
# Load data from "NASA_airfoil_noise_cleaned.parquet"
df = spark.read.parquet("NASA_airfoil_noise_cleaned.parquet")

# Show total number of rows in the dataset
rowcount4 = df.count()
print(rowcount4)

# Define the VectorAssembler pipeline stage
# Stage 1 - Assemble the input columns into a single column "features".
# Use all the columns except SoundLevelDecibels as input features.
assembler = VectorAssembler(
    inputCols=[
        "Frequency",
        "AngleOfAttack",
        "ChordLength",
        "FreeStreamVelocity",
        "SuctionSideDisplacement"
    ],
    outputCol="features"
)

# Define the StandardScaler pipeline stage
# Stage 2 - Scale the "features" using standard scaler and store in
# "scaledFeatures" column
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Define the StandardScaler pipeline stage
# Stage 3 - Create a LinearRegression stage to predict "SoundLevelDecibels"
lr = LinearRegression(
    featuresCol="scaledFeatures",
    labelCol="SoundLevelDecibels"
)

# Build a pipeline using the above three stages
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Split the data
# Split the data into training and testing sets with 70:30 split.
# Set the value of seed to 42
# DO NOT set the value of seed to any other value other than 42.
(trainingData, testingData) = df.randomSplit([0.7, 0.3], seed=42)

# Fit the pipeline using the training data
pipelineModel = pipeline.fit(trainingData)

# PART 2 - EVALUATION
print("Part 2 - Evaluation")
print("Total rows = ", rowcount4)
ps = [str(x).split("_")[0] for x in pipeline.getStages()]

print("Pipeline Stage 1 = ", ps[0])
print("Pipeline Stage 2 = ", ps[1])
print("Pipeline Stage 3 = ", ps[2])

print("Label column = ", lr.getLabelCol())


# PART 3 - EVALUATE THE MODEL
# Make predictions on testing data
predictions = pipelineModel.transform(testingData)

# Mean Square Error (MSE) - Lower the value the better the model
evaluator = RegressionEvaluator(
    labelCol="SoundLevelDecibels",
    predictionCol="prediction",
    metricName="mse"
)

mse = evaluator.evaluate(predictions)
print(f"MSE = {mse}")

# Mean Absolute Error (MAE) - Lower the value the better the model
evaluator = RegressionEvaluator(
    labelCol="SoundLevelDecibels",
    predictionCol="prediction",
    metricName="mae"
)

mae = evaluator.evaluate(predictions)
print(f"MAE = {mae}")

# R-Squared (R2) - Higher values indicate better performance
evaluator = RegressionEvaluator(
    labelCol="SoundLevelDecibels",
    predictionCol="prediction",
    metricName="r2"
)

r2 = evaluator.evaluate(predictions)
print(f"R Squared = {r2}")

# PART 3 - EVALUATION
print("Part 3 - Evaluation")

print("Mean Squared Error = ", round(mse,2))
print("Mean Absolute Error = ", round(mae,2))
print("R Squared = ", round(r2,2))

lrModel = pipelineModel.stages[-1]

print("Intercept = ", round(lrModel.intercept,2))


# PART 4 - PERSIST THE MODEL
# Save the pipeline model as "Final_Project"
# Persist the model to the path "./Final_Project/"
pipelineModel.write().overwrite().save("./Final_Project/")

# Load the pipeline model you have created in the previous step
loadedPipelineModel = PipelineModel.load("./Final_Project/")

# Use the loaded pipeline model and make predictions using testingData
predictions = loadedPipelineModel.transform(testingData)

# Show top 5 rows from the predections dataframe.
# Display only the label column and predictions
predictions.select("SoundLevelDecibels","prediction").show(5)

# PART 4 - EVALUATION
print("Part 4 - Evaluation")

loadedmodel = loadedPipelineModel.stages[-1]
totalstages = len(loadedPipelineModel.stages)
inputcolumns = loadedPipelineModel.stages[0].getInputCols()

print("Number of stages in the pipeline = ", totalstages)
for i,j in zip(inputcolumns, loadedmodel.coefficients):
    print(f"Coefficient for {i} is {round(j,4)}")

# Stop Spark Session
spark.stop()
