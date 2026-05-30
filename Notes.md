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

## SECTION 3: SIMPLE LINEAR REGRESSION
We ran a simple Linear Regression to model the relationship between PE1 and median income by fitting a linear equation. We calculated the slope and interception of this regression line and visualized it using a scatterplot.

Outputs:
- 37987.79390014965
- [1154.1923081]

**Model Equation: Predicted_Median = 37987.79 + 1154.19 * PE1**

Scatterplot with Regression Line
- The regression line represents the predicted median income based on the value of PE1. There is an upwards trend but we believe it is not a good fit for the data due to the fact you can see most of the points are spread out with clear outliers. The line does not have an aggressive slope, but a small incline. There is a lot of variation in median income where the answers from PE1 are not sufficient enough to be the sole indicator.

Residual Plot
- The residuals are not randomly scattered around zero. They show more of a vertical spread, but do not have a clear pattern. Considering the majority of points on our plot are far from 0, our model using PE1 as a feature does not accurately predict median income within the dataset and has large error. This concurs with the fact that we had such a low R-squared value.

- If there were a pattern, it would indicate our model was not a good fit or that the relationship between our feature and target was not linear.

- There are outliers far from our y = 0 line at each range. Each range has outlier points above 15,000 indicating our model significantly underestimated the actual median income.

- This plot would decrease our confidence in the model due to the amount of outliers and large vertical spread. There are not enough points that are scattered around zero, indicating using PE1 as a predictor provides very little information.

## SECTION 4: TRAIN / TEST SPLIT
We introduced a train/test split on our dataset to evaluate how the model worked when using held-out data and fix any potential problems of overfitting.

Outputs:
- ((419, 1), (105, 1), (419,), (105,))
- Test RMSE: 10191.327005442155

The model performed better on the held-out data, as our test RMSE came out to be 10,191.33, in comparison to our original RMSE of 10,371.10. This was a small decrease, of approximately only 1.73%. This tells us that our model was not signficantly overfit, as we originally expected our Phase 3 RMSE to go up. However, the minimal decrease and large RMSE still indicates the lack of predictive power with PE1.

## SECTION 5: FEATURE ENGINEERING
For feature engineering, a log transformation was applied on the feature variable PE1. We also created an interaction term between PE1 and PE3 because their perception of AI for information is likely affected by how they view AI in terms of productivity. Lastly, we used binning to capture any potential non-linear relationships.

Log Transformation
- After the transformation, the log transformed version of PE1 showed the points are less evenly distributed along the x-axis and the fit did not change. This indicates a lack of linearization between PE1 and median income. Therefore, using a log transformation on PE1 did not help.

Feature: PE1 x PE3
- The interaction term of PE1 and PE3 may have a combined effect because their perception of AI for information is likely affected by how they view AI in terms of productivity. Since the interaction terms have a multiplicative relationship with the target variable of median income, their effects are not independent of each other.

Feature: Binning
- We used binning since the relationship between PE1 and median income is not linear, as the scatterplot for PE1 and median income did not show a clear slope. The binning on PE1 divided the values of the respones into 4 equally spaced intervals and categorized them into groups.
- Bin 0: PE1 value = 1.0
- Bin 1: PE1 value = 2.0
- Bin 2: PE1 value = 3.0
- Bin 3: PE1 value = 4.0-5.0

## SECTION 6: FINAL MODEL / EVALUATION
In this final section, we integrated the engineered features of the log-transformed PE1 and the interaction term of PE1 x PE3 into a linear regression model.

Residual Plot
- Compared to our Phase 3 residual plot, we are able to see tighter concentration of grouping around zero. This shows a slight improvement in the prediction but we are still seeing a lot of large gaps in between data points. This shows that the model is still struggling to find a relationship and could benefit from more data.

Unified Comparison Table
- The evidence from the comparison table suggests that the previous Phase 3 SLR performed slightly better than the other models. While the engineered feature model improved the training RMSE, the Phase 3 SLR had a better test RMSE. We did see a small gap between the train and test RMSE, which suggests no overfitting. But overall, we are still seeing high RMSE values, which suggests that the low predictability between PE1 and median income.

Actual vs. Predicted Plot on Test Set
- This plot shows that the model's predictions were very similar to the actual values of median income.

- The lack of points being on the diagonal line indicate there is not many close predictions where the model's predicted income matches the actual income. Points above the line mean the predicted median income is higher than the actual median income, while points below the line mean the predicted median income is lower than the actual median income.

- The plot shows the model lacks significant predictive power (as also indicated with the low coefficient of determination) considering there are many points both above and below the line, with only one actually on it.

## 
The pattern we consistently saw was that our models were not able to find a strong connection between PE1 and median income. We did not see anything to suggest overfitting. The residual plots for the graphs and data showed high RMSE values, which show inability to properly predict a pattern between the two.

Instead of relying on PE1, a future improvement would be if we combined related survey responses and prompts to create a better measurement of students' perception of AI to better improve the connection from AI usage in high median income majors. This would help us reduce noise from individual survey questions, improving the ability to predict and correlate findings. Another improvement would be the incorporation of grouping by major. Since majors have the strongest connection to our median income data, including majors could improve the model's ability to predict relationships.




