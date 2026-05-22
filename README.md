# Research on AI Usage Among College Students & Their Major's Median Income

Research Question: How do college students' perceptions of AI relate to their major's median income?

Target: Median Income

Features: PE1, PE3

PE1 is a survey response from a scale of 1-5 to the statement, "The use of AI helps me access various sources of information relevant to my course, especially for academic assignments and understanding course materials."

PE3 is a survey response from a scale of 1-5 to the statement, "AI helps me improve my independent learning productivity in the digital era."

Final Dataset Shape: 524 rows x 53 columns

Our first two phases were irrelevant to our eventual findings, as we ended up changing our dataset due to our original being synthetically generated. In Phase 3, we looked at median income for each major and used responses to PE1 as a feature to see if students' views on AI for academic purposes related to their major's median income. We found that there was very little correlation (R-squared value being 0.00746) between the two variables due to the large amount of error in our residual plot. In Phase 4, we introduced a train/test split as well as new features such as the interaction term of PE1 x PE3 and PE1 binned. There was slight improvement in the training RMSE but it still underperformed; we consistently saw our models were not able to find a strong connection between PE1 and median income as the residual plots showed high RMSE values. We did not see anything to suggest overfitting; we saw strong precision, but low accuracy.
