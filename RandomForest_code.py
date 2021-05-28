# To display sklearn interactive diagrams:
from sklearn import set_config
set_config(display='diagram')

#Load required packages
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Creating a list of numerical and categorcal columns based on their dtype
from sklearn.compose import make_column_selector as selector

num_cols = selector(dtype_exclude=object)
cat_cols = selector(dtype_include=object)

num_cols = num_cols(X)
cat_cols = cat_cols(X)

#Categorical and numerical columns transformation pipelines
#Encoding categorical columns with scikitlearn OneHotEnccoder
#Scaling the numerical columns with StandardScaler

cat_transformer_onehot = Pipeline(steps=[('onehot_transf',OneHotEncoder(handle_unknown='ignore'))])
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

#Applying the ColumnTransformer preprocessing for numerical and categorical data
preprocessor = ColumnTransformer([('categoricals', cat_transformer_onehot, cat_cols),
                                  ('numericals', num_transformer, num_cols)],
                                 remainder = 'passthrough')

# Model Training                                 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

rfc_model = RandomForestClassifier()

#Bundle preprocessing and modeling code in a pipeline
my_pipeline_RFC = Pipeline(steps=[('preprocessor', preprocessor), 
                                  ('rfc_model', rfc_model)])

#Hyperparameter tuning implementation
param_grid_RFC = {
    'rfc_model__criterion': ['gini','entropy'],
    'rfc_model__n_estimators': [100, 235, 300], 
    'rfc_model__max_depth': [10, 30,50, 100], 
    'rfc_model__min_samples_split': [5, 15, 25],
    'rfc_model__random_state': [10],
    
}

searchCV_RFC = RandomizedSearchCV(my_pipeline_RFC,
                                  param_distributions=param_grid_RFC,
                                  cv=5, scoring='accuracy',n_jobs=-1)


final = searchCV_RFC.fit(X_train, y_train)

print('Best parameters for the Random Forest Classifier: \n',
      searchCV_RFC.best_params_) 
print('Best accuracy score for the Random Forest Classifier: ',
      searchCV_RFC.best_score_)


# Model Evaluation
best_model_RFC = RandomForestClassifier(criterion = 'entropy',
                                        max_depth=30,
                                        min_samples_split= 25,
                                        n_estimators= 300,
                                        random_state= 10,
                                        )


my_pipeline_best_RFC = Pipeline(steps=[('preprocessor', preprocessor), 
                                       ('best_model_RFC', best_model_RFC)])

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

cv_results = cross_validate(my_pipeline_best_RFC,
                            X_test,
                            y_test,
                            cv=cv,
                            scoring="accuracy",
                            return_train_score=True)


cv_results = pd.DataFrame(cv_results)#[['test_score','train_score']])
cv_results.head()

import matplotlib.pyplot as plt

cv_results[['test_score','train_score']].plot.hist(bins=15, edgecolor="white", density=True,alpha=0.5)
plt.xlabel("Accuracy")
_ = plt.title("Test score distribution")

print(f"Classifier accuracy is on the test dataset was: {cv_results['test_score'].min():.2f} +/- {cv_results['test_score'].std():.2f}")

# Kaggle submission

my_pipeline_final = searchCV_RFC.best_estimator_
predictions = my_pipeline_final.predict(test_data)

final_data = predictions.astype(int)

#Generate output
output = pd.DataFrame({'PassengerId': test_data.index, 
                       'Survived': final_data})
output.to_csv('submission_v12-rfc.csv', index=False)
print("File saved!")





