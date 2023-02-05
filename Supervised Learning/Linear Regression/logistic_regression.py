# Logistic regression is a regression model. The model builds a regression model to 
# predict the probability that a given data entry belongs to the category numbered as “1”.

# The setting of the threshold value is a very important aspect of 
# Logistic regression and is dependent on the classification problem itself.
 
# Accuracy: the total number of correct predictions divided by the total number of predictions made for a dataset.
# accuracy is not a good performance measure when the data is imbalanced.
# accuarcy = true positives + true negatives / total examples

# Precision quantifies the number of positive class predictions that actually belong to the positive class.
# precision = true positives / true positives + false positives

# Recall quantifies the number of positive class predictions made out of all positive examples in the dataset.
# recall = true positives / true positives + false negatives

# F-Measure provides a single score that balances both the concerns of precision and recall in one number.
# F1 = 2 * precision * recall / precision + recall

# Logistic Regression Types:
# binomial: target variable can have only 2 possible types: “0” or “1” which may represent “win” vs “loss”, “pass” vs “fail”, “dead” vs “alive”, etc.
# multinomial: target variable can have 3 or more possible types which are not ordered(i.e. types have no quantitative significance) like “disease A” vs “disease B” vs “disease C”.
# ordinal: it deals with target variables with ordered categories. For example, a test score can be categorized as:“very poor”, “poor”, “good”, “very good”. Here, each category can be given a score like 0, 1, 2, 3.