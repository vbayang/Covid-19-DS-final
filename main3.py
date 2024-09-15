'''
Team Member 1 Name: Uchechi Ndubueze
Team Member 1 Github Username: precious3017
Team Member 1 PSID: 1934230

Team Member 2 Name: Victoria Bayang
Team Member 2 Github Username: vbayang
Team Member 2 PSID: 2247177

Team Member 3 Name: Clifford Thompson
Team Member 3 Github Username: cjt497
Team Member 3 PSID: 1873121

Team Member 4 Name: Victor Carrillo
Team Member 4 Github Username: victoreczz
Team Member 4 PSID: 2208438

Team Member 5 Name: Vu Ho
Team Member 5 Github Username: hovu92tx
Team Member 5 PSID: 2137899

'''

# git push and commit test
import pandas as pd # Package to read data files and store columns as a dataframe
import matplotlib.pyplot as plt # Package to support plots
import numpy as np # Package to support data types
import seaborn as sns # Package to support heatmap plots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans # clustering library
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('covid/us_records_subset.csv')
daily_df = df.loc[:821]
weekly_df = df.loc[823:].loc[(df[['new_cases', 'new_deaths','daily_case_change_rate','daily_death_change_rate']] != 0).all(axis=1)]

daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df = daily_df.dropna()

# print('DAILY DATA')
# print(daily_df)
# print()
# print('WEEKLY DATA')
# print(weekly_df)
# print()

''' -------------------------------Scatter Plot---------------------------------'''
"""scatter plot with less xticks"""
x = daily_df['date']
y = daily_df['new_cases']
x = np.asarray(x, dtype='datetime64[s]')

plt.scatter(x, y)
plt.xlabel('Date')
plt.ylabel('New Covid Cases')
plt.title('Scatter Plot: Date vs. New Covid Cases')
plt.tick_params(axis='x', labelrotation=90)
plt.show()

''' -------------------------------CORRELATION MATRIX---------------------------------'''
"""process data to make icu_requirements numerical"""
df_numerical_daily = daily_df.copy()
df_numerical_weekly = weekly_df.copy()
df_numerical_daily['icu_requirement'] = df_numerical_daily['icu_requirement'].replace({'High': 3, 'Medium': 2, 'Low': 1})
df_numerical_weekly['icu_requirement'] = df_numerical_weekly['icu_requirement'].replace({'High': 3, 'Medium': 2, 'Low': 1})

"""process data to separate feature types"""
original_features = ['total_cases','new_cases','total_deaths','new_deaths','total_cases_per_million','total_deaths_per_million','icu_patients','hosp_patients','weekly_hosp_admissions', 'icu_requirement']
derived_features = ['daily_case_change_rate','daily_death_change_rate','hospitalization_rate','icu_rate','case_fatality_rate','7day_avg_new_cases','7day_avg_new_deaths', 'icu_requirement']
daily_df_original = df_numerical_daily[original_features]
daily_df_derived = df_numerical_daily[derived_features]
weekly_df_original = df_numerical_weekly[original_features]
weekly_df_derived = df_numerical_weekly[derived_features]

"""correlation matrix for original features measured daily"""
pause_value = True
sns.heatmap(daily_df_original.corr(numeric_only=True), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix - Original Features Measured Daily')
plt.xticks(rotation=35, ha='right')
#plt.savefig('Daily_Corr_Orig.png')
plt.show()

"""correlation matrix for derived features measured daily"""
sns.heatmap(daily_df_derived.corr(numeric_only=True), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix - Derived Features Measured Daily')
plt.xticks(rotation=35, ha='right')
#plt.savefig('Daily_Corr_Derived.png')
plt.show()
if (pause_value):
    pause = input('Press enter to continue...')

"""correlation matrix for original features measured weekly"""
sns.heatmap(weekly_df_original.corr(numeric_only=True), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix - Original Features Measured Weekly')
plt.xticks(rotation=35, ha='right')
#plt.savefig('Weekly_Corr_Orig.png')
plt.show()

"""correlation matrix for derived features measured weekly"""
sns.heatmap(weekly_df_derived.corr(numeric_only=True), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix - Derived Features Measured Weekly')
plt.xticks(rotation=35, ha='right')
#plt.savefig('Weekly_Corr_Derived.png')
plt.show()
if (pause_value):
    pause = input('Press enter to continue...')

''' --------------------------Boxplots of Attributes per Class--------------------------'''
cols = ['total_cases','new_cases','total_deaths','new_deaths','total_cases_per_million','total_deaths_per_million',
                        'icu_patients','hosp_patients','weekly_hosp_admissions','daily_case_change_rate','daily_death_change_rate',
                        'hospitalization_rate','icu_rate','case_fatality_rate','7day_avg_new_cases','7day_avg_new_deaths']

for label in cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=daily_df['icu_requirement'], y=daily_df[label])
    plt.xlabel('icu_requirement')
    plt.ylabel(label)
    plt.title('Box Plot')
    plt.show()


'''Task 1: Regression '''
print("Task 1: Regression\n")
# Select the features and the target
X = daily_df[['total_cases', 'total_deaths']]
y = daily_df['icu_patients']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")


'''Task 2: Classification'''
# Separate features and target variable
X = daily_df.drop(['date', 'hospitalization_need'], axis=1)  # Adjust this line if there are other non-numeric columns
y = daily_df['hospitalization_need']

# Encode categorical variables (if any)
# Assuming 'icu_requirement' is categorical and needs encoding
X = pd.get_dummies(X, columns=['icu_requirement'], drop_first=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Model development
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# Model evaluation
y_pred = model.predict(X_test_selected)
print(classification_report(y_test, y_pred))

# Plot
numeric_cols = daily_df.select_dtypes(include=['int64', 'float64']).columns

'''Task 4: Exploratory Data Analysis (EDA)'''
# Visualize trends over time
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(daily_df['date'], daily_df['total_cases'], label='Total Cases')
plt.title('Trend of Total Cases')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(daily_df['date'], daily_df['total_deaths'], label='Total Deaths', color='red')
plt.title('Trend of Total Deaths')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(daily_df['date'], daily_df['hospitalization_rate'], label='Hospitalization Rate', color='green')
plt.title('Trend of Hospitalization Rate')
plt.legend()

plt.tight_layout()
plt.show()

# Clustering
# Scaling data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(daily_df[['total_cases', 'total_deaths', 'hospitalization_rate']])

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
daily_df['cluster'] = kmeans.labels_

# Visualize clusters
sns.pairplot(daily_df, vars=['total_cases', 'total_deaths', 'hospitalization_rate'], hue='cluster', palette='viridis')
plt.suptitle('Pairplot with Clustering', y=1.02)
plt.show()

# Analyze clusters
for i in range(3):
    cluster_data = daily_df[daily_df['cluster'] == i]
    print(f"Cluster {i}:")
    print(f"Average Total Cases: {cluster_data['total_cases'].mean()}")
    print(f"Average Total Deaths: {cluster_data['total_deaths'].mean()}")
    print(f"Average Hospitalization Rate: {cluster_data['hospitalization_rate'].mean()}")
    print()