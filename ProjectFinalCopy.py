# SECTION 1: DATA LOADING AND CLEANING
!pip install pandas numpy matplotlib cufflinks plotly openpyxl scikit-learn seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly
import seaborn as sns

d1_student_usage = pd.read_excel("datasetNEWAI.xlsx", engine="openpyxl")
d2_major_employment = pd.read_csv("recent-grads.csv")

# Dataset Checkers:
# # of rows
len(d1_student_usage)
len(d2_major_employment)
#Helper: head
d1_student_usage.head()
d2_major_employment.head()
#helper: column
d1_student_usage.columns
d2_major_employment.columns
#Helper: info
d1_student_usage.info()
d2_major_employment.info()
#Helper: finding all unique naming for merging
d2_major_employment["Major_category"].unique()
d1_student_usage["Fields of Study"].unique()
#MERGING DATA
# Goal is to add per major info: median salary, and unemployment rate

#IN DATASET they use numerical values assoicated with the respective major.
academic_mapping_d1 = {
    0: "Language",
    1: "Economics",
    2: "Natural Science",
    3: "Social Science",
    4: "Education",
    5: "Mathematics",
    6: "Art",
    7: "Engineering"
}
d1_student_usage["Academic Field"] = d1_student_usage["Fields of Study"].astype("Int64").map(academic_mapping_d1)
d1_student_usage.columns.tolist()

# FOR EDA 1 of PHASE TWO: NEED MAPPING FROM AI ENGINES #s TO NAMES
ai_naming_mapping = {
    1: "ChatGPT",
    2: "Gemini",
    3: "Grammarly",
    4: "Quillbot",
    5: "Notion AI",
    6: "Midjourney",
    7: "DALL-E",
    8: "Perplexity AI",
    9: "Eduten",
    10: "Shutterstock AI",
    11: "Sheet Plus",
    12: "Others"
}

# Dataset 2: MAJOR NAME PATHING
connected_disp_cat = {
    "Language": "Humanities & Liberal Arts",
    "Economics": "Business",
    "Natural Science": "Biology & Life Science",
    "Social Science": "Social Science",
    "Education": "Education",
    "Mathematics": "Computers & Mathematics",
    "Art": "Arts",
    "Engineering": "Engineering"
}

#University MAPPING
university_mapping = {
    1: "Public University",
    2: "Private University"
}

#Gender Mapping
gender_mapping = {
    1: "Male",
    2: "Female"
}

# Education Mapping
education_mapping = {
    1: "Associate Degree",
    2: "Undergraduate",
    3: "Masters"

}

d1_student_usage["Major_category"] = d1_student_usage["Academic Field"].map(connected_disp_cat)

# CLEANING: Some Major Categorys were mapped to value of 8, which doesn't exist
d1_student_usage = d1_student_usage[d1_student_usage["Fields of Study"].between(0, 7)]

# Making the two categories added: median income and unemployment rate
grads_by_category = d2_major_employment.groupby("Major_category").agg({"Median": "mean", "Unemployment_rate": "mean"}).reset_index()

# ACTUAL MERGE
Data_merged = pd.merge(d1_student_usage, grads_by_category, on = "Major_category", how = "left")
Data_merged = Data_merged.iloc[:535]

# adding AI NAMES
Data_merged["AI Names"] = (Data_merged["Type of AI"].astype("Int64").map(ai_naming_mapping))

# University Mapping (num -> int)
Data_merged["University"] = ( Data_merged["University"].astype("Int64").map(university_mapping))

# Gender Mapping(num -> int)
Data_merged["Gender"] = ( Data_merged["Gender"].astype("Int64").map(gender_mapping))

# Education Mapping
Data_merged["Level of Education"] = ( Data_merged["Level of Education"].astype("Int64").map(education_mapping))

# Merged Dataset Print
Data_merged

## PART 2: BASELINE MODEL: The baseline model (constant model) predicts the mean of our target variable median income for every observation.
# Constant Model MSE
constant_prediction = Data_merged['Median'].mean()
Data_merged['pred_constant'] = constant_prediction
Data_merged['error'] = Data_merged['Median'] - Data_merged['pred_constant']
Data_merged['squared_error'] = Data_merged['error'] ** 2

# Constant Model RMSE
mse_constant = Data_merged['squared_error'].mean()
rmse_constant = np.sqrt(mse_constant)

print("MSE of the constant model: $", mse_constant)
print("RMSE of the constant model: $", rmse_constant)

# PART 3: SIMPLE LINEAR REGRESSION: Model the relationship between PE1 and median income by fitting a linear equation. We calculated the slope and interception of this regression line and visualized it using a scatterplot.

def standard_units(x):
    return (x - np.mean(x)) / np.std(x)

def correlation(x, y):
    return np.mean(standard_units(x) * standard_units(y))

def slope(x, y):
    return correlation(x, y) * np.std(y) / np.std(x)

def intercept(x, y):
    return np.mean(y) - slope(x, y) * np.mean(x)

# x and y setting again for calculations
x = Data_merged["PE1"]
y = Data_merged["Median"]

# Slope + Intercept
m = slope(x, y)
b = intercept(x, y)

# Predicted Median
Predicted_Median = b + m * x
# Sklearn Verification
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x.values.reshape(-1,1), y)

print(model.intercept_)  # 37987.79390014965
print(model.coef_)   # 1154.1923081

# Scatterplot with Regression Line

# Predicted median calc
Predicted_Median = b + m * x
plot_4c = plt.figure()
axis_4c = plt.axes()
axis_4c.scatter(x, y)
axis_4c.plot(x, Predicted_Median)

# plot labels
axis_4c.set_xlabel("PE1")
axis_4c.set_ylabel("Median Income")
axis_4c.set_title("PE1 vs Median Income with Regression Line")

plt.show()

# Residual Plot

Data_merged['Predicted_Median'] = Predicted_Median
residuals = Data_merged['Median'] - Data_merged['Predicted_Median']

plt.scatter(Data_merged['Predicted_Median'], residuals)
plt.xlabel('Predicted Median Income')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='b', linestyle = '--')
plt.show()

# PART 4: TRAIN / TEST SPLIT: Evaluate how the model worked when using held-out data and fix any potential problems of overfitting.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

x = Data_merged[['PE1']]
y = Data_merged['Median']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Check for 80/20 split
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# Fit model
model = LinearRegression()

model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
print('Test RMSE:', rmse_test)

