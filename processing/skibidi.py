import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
df = pd.read_csv('C:/2024/ML/machineLearning/Github/machinelearning/data/dataset_traffic_accident_prediction1.csv')

columns_with_missing = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol',
                       'Driver_Age', 'Driver_Experience',
                       'Accident']


for i in columns_with_missing:
    print(f"{i} - trung binh : {df[i].mean()}, trung vi: {df[i].median()}")
    df[i].plot(kind='hist', bins=30, title=i)
    plt.axvline(df[i].mean(), color='red', linestyle='dashed', linewidth=2)
    plt.axvline(df[i].median(), color='blue', linestyle='dashed', linewidth=2)
    plt.show()

df['Traffic_Density'].fillna(df['Traffic_Density'].median(), inplace=True)
df['Speed_Limit'].fillna(df['Speed_Limit'].median(), inplace=True)
df['Number_of_Vehicles'].fillna(df['Number_of_Vehicles'].median(), inplace=True)
df['Driver_Alcohol'].fillna(df['Driver_Alcohol'].median(), inplace=True)
df['Driver_Age'].fillna(df['Driver_Age'].median(), inplace=True)
df['Driver_Experience'].fillna(df['Driver_Experience'].median(), inplace=True)
df['Accident'].fillna(df['Accident'].median(), inplace=True)


columns_miss = df.columns[df.isnull().sum() > 0]

df[columns_miss].info()

df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
df['Road_Type'] = df['Road_Type'].fillna(df['Road_Type'].mode()[0])
df['Time_of_Day'] = df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
df['Accident_Severity'] = df['Accident_Severity'].fillna(df['Accident_Severity'].mode()[0])
df['Road_Condition'] = df['Road_Condition'].fillna(df['Road_Condition'].mode()[0])
df['Vehicle_Type'] = df['Vehicle_Type'].fillna(df['Vehicle_Type'].mode()[0])
df['Road_Light_Condition'] = df['Road_Light_Condition'].fillna(df['Road_Light_Condition'].mode()[0])

df = df.drop_duplicates()

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

sns.boxplot(x=df['Speed_Limit'], ax=axes[0, 0])
axes[0, 0].set_title('Speed_Limit')

sns.boxplot(x=df['Number_of_Vehicles'], ax=axes[0, 1])
axes[0, 1].set_title('Number_of_Vehicles')

sns.boxplot(x=df['Driver_Age'], ax=axes[1, 0])
axes[1, 0].set_title('Driver_Age')

sns.boxplot(x=df['Driver_Experience'], ax=axes[1, 1])
axes[1, 1].set_title('Driver_Experience')

sns.boxplot(x=df['Traffic_Density'], ax=axes[2, 0])
axes[2, 0].set_title('Traffic_Density')

fig.delaxes(axes[2, 1])

plt.tight_layout()
plt.show()

cols = ['Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience']
sns_plot = sns.pairplot(df[cols])
sns_plot.savefig('pairplot.png')

numeric_df = df.select_dtypes(include=['number'])
spearman_corr = numeric_df.corr(method='spearman')

# Ma trận tương quan
print(spearman_corr)

# Heatmap tuong quan
plt.figure(figsize=(8, 7))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Spearman Correlation Matrix')
plt.show()


df= df.drop(['Accident_Severity'], axis =1)
df = pd.get_dummies(df, columns=['Weather', 'Road_Type', 'Time_of_Day', 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition'], drop_first=True)

df['Experience'] = df['Driver_Age'] - df['Driver_Experience']
df = df.drop(['Driver_Age', 'Driver_Experience'], axis = 1)



numeric_columns = ['Speed_Limit', 'Number_of_Vehicles', 'Experience']
outlier = ['Speed_Limit', 'Number_of_Vehicles']
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
for col in outlier:
    df = remove_outliers(df, col)

X = df.drop(['Accident'], axis = 1)
y = df['Accident']
feature_names = X.columns

scaler = StandardScaler()

X_scaled = X.copy()
X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state=100)
smote = SMOTE(random_state=100)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
knn_classifier = KNeighborsClassifier()

#Định nghĩa các tham số cho GridSearchCV
parameters_knn = {
    'n_neighbors':range(1,30),      # Số lượng láng giềng
    'weights': ['uniform', 'distance'],   # Kiểu trọng số (đều hay theo khoảng cách)
    'metric': ['euclidean', 'manhattan'], # Khoảng cách giữa các điểm
    'algorithm': ['auto','ball_tree','kd_tree'],
    'leaf_size': [5,10,50]
}

grid_knn = GridSearchCV(KNeighborsClassifier(), parameters_knn, cv =5)
grid_knn.fit(X_train_resampled, y_train_resampled)
# best_params_knn = grid_knn.best_estimator_
#
# print('Score on train data = ', round(best_params_knn.score(X_train, y_train), 4))
# print('Score on test data = ', round(best_params_knn.score(X_test, y_test), 4))


#random forest
# rf_classifier = RandomForestClassifier()
#
# parametrs_rf = {'max_depth':[3, 5, 10],
#                 'n_estimators':[150, 200, 300],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4]}
#
# grid_search_reg_forest3 = GridSearchCV(rf_classifier, parametrs_rf, cv = 5)
#
# grid_search_reg_forest3.fit(X_train_resampled, y_train_resampled)
#
# best_gs_rf = grid_search_reg_forest3.best_estimator_
#
# print('Score on train data = ', round(best_gs_rf.score(X_train, y_train), 4))
# print('Score on test data = ', round(best_gs_rf.score(X_test, y_test), 4))


# logis_classifier = LogisticRegression()
#
# param_logis = {
#
#     'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty':['l1', 'l2', 'elasticnet', 'none'],
#     'solver':['lbfgs', 'liblinear', 'saga', 'newton-cg'],
#     'max_iter':[50, 100, 200]
# }
#
# grid_search_logis = GridSearchCV(logis_classifier, param_logis, cv = 5)
#
# grid_search_logis.fit(X_train_resampled, y_train_resampled)
#
# best_gs_rf_logis = grid_search_logis.best_estimator_

# print('Score on train data = ', round(best_gs_rf_logis.score(X_train, y_train), 4))
# print('Score on train data = ', round(best_gs_rf_logis.score(X_test, y_test), 4))

# stacking_model = StackingClassifier(
#     estimators=[('logreg', grid_search_logis), ('rf', grid_search_reg_forest3)],  # Base learners
#     final_estimator=LogisticRegression(),                   # Meta learner
#     passthrough=False                                       # Sử dụng đầu ra của base learners
# )
# stacking_model.fit(X_train, y_train)
#
# y_pred = stacking_model.predict(X_test)
# print("Accuracy of Stacking Classifier:", accuracy_score(y_test, y_pred))

# xg boost
# xgb_classifier = xgb.XGBClassifier(eval_metric='mlogloss')
#
# param_xgb = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.1, 0.2],
#     'max_depth': [3, 5, 10],
#     'subsample': [0.8, 1.0],
#     'min_child_weight': [1, 3, 5]
# }
#
#
# grid_search__xgb = GridSearchCV(xgb_classifier, param_xgb, cv=5)
#
# grid_search__xgb.fit(X_train_resampled, y_train_resampled)
#
test_set = X_test
y_test_set= y_test
y_predict = grid_knn.predict(test_set)


results = pd.DataFrame({
    'Actual': y_test_set,
    'Predicted': y_predict
})

# In kết quả
print(results)
print(classification_report(y_test_set, y_predict))



