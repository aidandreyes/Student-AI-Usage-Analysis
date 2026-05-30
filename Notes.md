# Notes and Observations

## SECTION 1: DATA LOADING & CLEANING
Our data came from an extensive study from Indonesian higher education at the State University of Padang where students reported their expectations of AI in relation to their education. The dataset includes demographic variables like university type, field of study, and educational level, along with students' self-reported experiences with AI in academic settings.

- Source: https://data.mendeley.com/datasets/b89t4x2c2y/1

## SECTION 2: BASELINE MODEL
We introduced the constant model as our baseline model. This predicts the mean of our target variable median income for every observation.

Constant Model RMSE output: 

- MSE of the constant model: $ 107559718.61872235
- RMSE of the constant model: $ 10371.10016433755

The calculated MSE of the constant model is $107,559,718.62. The calculated RMSE of the constant model is $10,371.10. This means that if we predict every major's median income to be the overall average median income, our predictions will be off by approximately $10,371.10 on average.

# SECITON 3: SIMPLE LINEAR REGRESSION
We ran a simple Linear Regression to model the relationship between PE1 and median income by fitting a linear equation. We calculated the slope and interception of this regression line and visualized it using a scatterplot.

- Model Equation: Predicted_Median = 37987.79 + 1154.19 * PE1

Scatterplot with Regression Line
- The regression line represents the predicted median income based on the value of PE1. There is an upwards trend but we believe it is not a good fit for the data due to the fact you can see most of the points are spread out with clear outliers. The line does not have an aggressive slope, but a small incline. There is a lot of variation in median income where the answers from PE1 are not sufficient enough to be the sole indicator.

Residual Plot
- The residuals are not randomly scattered around zero. They show more of a vertical spread, but do not have a clear pattern. Considering the majority of points on our plot are far from 0, our model using PE1 as a feature does not accurately predict median income within the dataset and has large error. This concurs with the fact that we had such a low R-squared value.

- If there were a pattern, it would indicate our model was not a good fit or that the relationship between our feature and target was not linear.

- There are outliers far from our y = 0 line at each range. Each range has outlier points above 15,000 indicating our model significantly underestimated the actual median income.

- This plot would decrease our confidence in the model due to the amount of outliers and large vertical spread. There are not enough points that are scattered around zero, indicating using PE1 as a predictor provides very little information.
