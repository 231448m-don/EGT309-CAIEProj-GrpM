# How to run the pipeline

Kedro:
You have to clone the git repository to obtain all the necessary folders.

Then, create a virtual environment in your command prompt by running 'python -m venv (virtual environment name)'.

Run the virtual environment with '(virtual environment name)\Scripts\activate.bat'

Install the necessary python libraies and dependencies by running 'pip install -r requirements.txt'.

Finally, ensure you are in the virtual environment and the necessaru dependencies are installed before running 'kedro run'.

Kedro on Docker:
You have to first change your command prompt directory to the kedro-pipeline directory.

Then, create a docker container by running 'Docker build -t (docker container name) .'

After that, you have to run the container with 'Docker run (docker container name), and the container shall start running the kedro pipeline.

# Contributions

Kay Kit: Exploratory Data Analysis + Slides

Brandon: Kedro Pipeline + Docker

Ilhan: Machine Learning Algorithms + Docker + Slides

# Overview of EDA
1. Cleaning and Standardising the Raw Data

The dataset originally contained messy text fields, strange value formats, and several placeholder codes.
I performed the following cleaning steps:

1.1. Standardised text columns

All string values were converted to lowercase and formatted in snake_case.
This prevents the model from treating “Cell”, “cell”, and “cellular” as different categories.

1.2. Cleaned the Age field

The age column was stored as text (e.g., “57 years”).
I removed the words, converted it to numeric, and replaced unrealistic ages (like 150) with NaN.

1.3. Removed placeholder or incorrect values

Values such as:

150 for age, 999 for previous contact days, negative values in campaign calls and None / “unknown” were standardised to NaN or corrected.

This makes the data consistent and allows for proper imputation later.

2. Understanding Relationships (Why It Matters)

Before filling in missing values, I analysed how key demographic features relate to one another:

Age vs Occupation

Age vs Marital Status

Age vs Education Level

Education vs Occupation

Marital Status vs Occupation

These relationships showed clear and realistic life-stage patterns, which guided all the imputation steps.

3. Multi-Stage Imputation (More Accurate Than Simple Median/Mode)

Instead of using a single median or mode for missing values, I used multi-stage imputation to maintain real-world patterns.

3.1. Age Imputation

Filled missing ages using:

Education + Occupation + Marital Status

Occupation + Marital Status

Occupation only

Global median

This ensures the imputed ages are realistic and match the client’s profile.

3.2. Occupation, Marital Status, and Education Imputation

These were imputed using staged mode-based rules that consider age group and other demographic features.
This keeps the dataset logically consistent.

4. Creating Derived Features (Feature Engineering)

To improve model performance, I added several engineered features that provide more meaningful signals:

4.1. previous_contact_flag

Indicates whether the client was previously contacted.
This is more useful than the raw number of days.

4.2. contact_intensity

A ratio between campaign calls and recency.
Helps capture how aggressively a client was contacted.

4.3. loan_burden

A simple score combining housing loan, personal loan, and credit default status.
Gives a quick sense of financial risk.

4.4. age_group

Age buckets such as 17–25, 26–35, 36–50, etc.
These make patterns easier to learn than raw numeric ages.

4.5. is_higher_edu

Binary indicator to check if the client has higher education.
Education level strongly affects subscription behaviour.

4.6 One-Hot Encoding

All categorical columns were one-hot encoded so that the model can properly understand them.

5. Final Prepared Dataset

After preprocessing and feature engineering, the final modelling dataset contains:

Clean, consistent values

No unrealistic placeholders

Fully imputed demographic fields

Useful engineered features

One-hot encoded categorical variables

Total of 33 model-ready features

This final dataset is reliable, structured, and optimised for machine learning training.

# Reasons for choice of models

1. Logistic Regression
   One of the simpler and more interpretable classification models.
   It works well when there are relationships between features, such as the age group, marital status, education level in the given data.
   Quick to train + fast to predict.
   However. it does not capture nonlinear patterns and is sensitive to outliers and multicollinearity.
   
2. Random Forest
   This model makes up for Logistic Regressions' incapability of capturing nonlinear patterns.
   Works very well with tabular data like the one that was provided.
   Is able to handle categorical, numerical relationships, and provides feature importances easily.
   The downside is that it is slower than Logistic Regression, predictions are less interpretable, and the accuracy is not as good as XGBoost.
   
3. Xtreme Gradient Boost (XGBoost)
   This model is one of the most powerful algorithms for tabular data.
   Handles complex interactions between features really well, and missing data automatically.
   Also, captures nonlinear patters and has the highest predictive performance compared to Random Forest and Logistic Regression.
   But, it is more complex to tune than the other 2 models, and training time is longer as well.

Therefore, we chose these 3 models to mitigate each model's disadvantages and all in all have more advantages than disadvantages.
