import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, __all__, metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate
from yellowbrick.cluster import SilhouetteVisualizer

# -------------------------------preprocessing of dataset bike -------------------------------#

# Dataset shape
data = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')
data.head(10)
data.head()

data.rename({"Dew point temperature(°C)": "Dewpointtemperature",
             "Temperature(°C)": "Temperature",
             "Rented Bike Count": "RentBikeCount",
             "Humidity(%)": "Humidity",
             "Wind speed (m/s)": "Windspeed",
             "Solar Radiation (MJ/m2)": "SolarRadiation",
             "Rainfall(mm)": "Rainfall",
             "Snowfall (cm)": "Snowfall",
             "Visibility (10m)": "Visibility",
             "Functioning Day": "FunctioningDay"}, axis=1, inplace=True)

print('data shape: ', data.shape)
# Drop extra column
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
print('data shape: ', data.shape)  # verify
# Dataset shape
nrows, ncols = data.shape
# Dataset info
data.info()
# Verify
data.count()
print("Null values")
print(data[data.isnull().any(axis=1)])  # no null values
print("NaN values")
print(data[data.isna().any(axis=1)])  # no nan values
print("Duplicates")
print(data.duplicated().any())  # No duplicated rows
print("Single value columns")
print(data.nunique())  # No columns with single same value

data.head(10)

# Export to csv and save cleaned dataset


# ------------------------------- exploration of dataset bike -------------------------------#
data.isnull().sum()
data.duplicated().sum()

file = 'SeoulBikeData_Cleaned_data.csv'
data.to_csv(file, index=False)

file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

data["Date"] = pd.to_datetime(data["Date"])
data["Month"] = data.Date.dt.month_name().str[:3]
data["Year"] = data.Date.dt.year
data["Weekday"] = data.Date.dt.weekday
data.Weekday = data.Weekday.map(
    {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"})
data["Weekday"] = pd.Categorical(data["Weekday"],
                                 categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                             'Sunday'],
                                 ordered=True)
data.head()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={'hspace': 0.3})
sns.barplot(data=data, x="Seasons", y="RentBikeCount", palette="colorblind", ax=ax1, ci=None)
sns.lineplot(data=data, x="Month", y="RentBikeCount", palette="colorblind", ax=ax3, ci=None)
sns.barplot(data=data, x="Weekday", y="RentBikeCount", palette="colorblind", ax=ax2, ci=None)
sns.lineplot(data=data, x="Hour", y="RentBikeCount", color="tab:blue", ax=ax4, ci=None)
ax1.set_title("Seasonal Bike Rents")
ax3.set_title("Monthly Bike Rents")
ax1.set_xlabel("")
ax3.set_xlabel("")
ax2.set_title("Weekday Bike Rents")
ax4.set_title("Hourly Bike Rents")
ax2.set_xlabel("")
plt.show()

plt.figure(figsize=(8, 14))
corr = data[["Temperature", "Windspeed", "Humidity", "SolarRadiation", "Rainfall", "Visibility", "Snowfall",
             "RentBikeCount"]].corr()
sns.heatmap(data=corr[["RentBikeCount"]].sort_values(by='RentBikeCount', ascending=False), vmin=-1, vmax=1,
            center=0, cmap="coolwarm", annot=True, square=True, linewidths=.5)
plt.title("Correlation with weather data")
plt.show()

weather_cols = ['Temperature', 'Humidity',
                'Windspeed', 'Visibility', 'Dewpointtemperature',
                'SolarRadiation', 'Rainfall', 'Snowfall']

fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(nrows=2, ncols=4)

ax0.hist(data['Temperature'], bins=94)
ax0.set_title('Temperature')

ax1.hist(data['Humidity'], bins=94)
ax1.set_title('Humidity')

ax2.hist(data['Windspeed'], bins=94)
ax2.set_title('Windspeed')

ax3.hist(data['Visibility'], bins=94)
ax3.set_title('Visibility')

ax4.hist(data['Dewpointtemperature'], bins=94)
ax4.set_title('Dew Point Temp')

ax5.hist(data['SolarRadiation'], bins=94)
ax5.set_title('Solar Radiation')

ax6.hist(data['Rainfall'], bins=94)
ax6.set_title('Rainfall')

ax7.hist(data['Snowfall'], bins=94)
ax7.set_title('Snowfall')

fig.tight_layout()
plt.show()

file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

# Separate predictor and target features
X = data.iloc[:, 1:]  # subset features
y = data.iloc[:, 0]  # subset class
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X),
                 columns=X.columns, index=X.index)

# Plot Histograms to visualize distributions
histogram = X.hist(figsize=(20, 20), bins=30)
plt.suptitle('Histograms of Features', fontsize=16)
plt.show()

# -------------------------------Feature Selection Bike -------------------------------#
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

# ANOVA FEATURE SELECTION
X = data.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
fs = SelectKBest(score_func=f_classif, k=10)
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)
fs_score_df = pd.DataFrame()
feature_no = 0
feature_score = 0

for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    fs_score_df = fs_score_df.append({feature_no: i,
                                      feature_score: fs.scores_[i]}, ignore_index=True)
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
fs_score_df['Feature Name'] = data.columns[1:]
fs_score_df.rename(columns={0: 'Score'}, inplace=True)
fs_score_df = fs_score_df.sort_values(by=['Score'], ascending=False)
plt.xlabel('Bike Features')
plt.ylabel('Feature Selection Scores')
plt.show()

# %% Reduced Feature Set - by removing highly correlated features
# Load file
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
# ANOVA Refined No High correlation
data = data[[
    'Date', 'RentBikeCount', 'Hour', 'Temperature', 'Humidity', 'Windspeed', 'Visibility', 'Dewpointtemperature',
    'SolarRadiation', 'Rainfall', 'Seasons', 'Holiday', 'FunctioningDay']]

X = data.iloc[:, 1:]
y = data.iloc[:, 1]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
fs = SelectKBest(score_func=f_classif, k=6)
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)
fs_score_df = pd.DataFrame()
feature_no = 0
feature_score = 0
print("Completed Bike Feature Selection")
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    fs_score_df = fs_score_df.append({feature_no: i,
                                      feature_score: fs.scores_[i]}, ignore_index=True)

##################################################################################
# ------------------------------- K Means Bike  ------------------------------- #
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

data = data[[
    'Date', 'RentBikeCount', 'Hour', 'Temperature', 'Humidity', 'Windspeed', 'Visibility', 'Dewpointtemperature',
    'SolarRadiation', 'Rainfall', 'Seasons', 'Holiday', 'FunctioningDay']]

X = data.loc[:, ['RentBikeCount', 'Temperature']]

# image size
plt.figure(figsize=(10, 5))

# ploting scatered graph
plt.scatter(x=X['RentBikeCount'], y=X['Temperature'])
plt.xlabel('RentBikeCount')
plt.ylabel('Temperature(°C)')
plt.title("Kmeans Bike")
plt.show()

# ------------------------------- K VALUE Bike  ------------------------------- #
wcss = []

# for loop
for i in range(1, 11):
    # k-mean cluster model for different k values
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)

    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

# figure size
plt.figure(figsize=(10, 5))
sns.lineplot(range(1, 11), wcss, marker='o', color='green')
print(wcss)
# labeling
plt.title('The Elbow Method Bikes')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
Kmean_n_clusters_Bike = 3
# ------------------------------- Kmeans Bike with clusters ------------------------------- #
# Elbow Method Shows 3 is the optimal number of clusters
start = time.time()
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

data = data[
    ['Date', 'RentBikeCount', 'Hour', 'Temperature', 'Humidity', 'Windspeed', 'Visibility', 'Dewpointtemperature',
     'SolarRadiation', 'Rainfall', 'Seasons', 'Holiday', 'FunctioningDay']]

X = data.loc[:, ['RentBikeCount', 'Temperature']]

km = KMeans(Kmean_n_clusters_Bike)
km.fit(X)

# ploting the graph of the clusters
plt.figure(figsize=(10, 5))
scatter = plt.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=km.labels_, cmap="Set2")
plt.xlabel('RentBikeCount')
plt.ylabel('Temperature(°C)')
plt.title("Kmeans Bike with clusters")
plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1, 2, 3])
plt.show()
finish = time.time()
# ------------------------------- Kmeans Bike with Time Taken ------------------------------- #

Kmean_time_taken = finish - start

# ------------------------------- Kmeans Bike with Davis-Bouldin score  ------------------------------- #
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['Temperature']
x = data[['RentBikeCount']]

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=30).fit(x)
labels = kmeans.fit_predict(x)

kmeans_Davis_Bouldin_score = davies_bouldin_score(x, labels)

# ------------------------------- Kmeans with CSM ------------------------------- #
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['Temperature']

x = data[['RentBikeCount']]
km = KMeans(n_clusters=3, random_state=42)

km.fit_predict(X)

kmeans_silhouette_avg = silhouette_score(X, km.labels_, metric='euclidean')

fig, ax = plt.subplots(2, 2, figsize=(15, 8))

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
    if k in (2, 4):
        plt.title("$k={}$".format(k))

    if k in (3, 5):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.title("$k={}$".format(k))

    q, mod = divmod(k, 2)
    plt.title("$k={}$".format(k))

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
    visualizer.fit(x)
plt.show()

# ------------------------------- Agglomerative Bike with clusters ------------------------------- #
start = time.time()

file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

data = data[
    ['Date', 'RentBikeCount', 'Hour', 'Temperature', 'Humidity', 'Windspeed', 'Visibility', 'Dewpointtemperature',
     'SolarRadiation', 'Rainfall', 'Seasons', 'Holiday', 'FunctioningDay']]

X = data.loc[:, ['RentBikeCount', 'Temperature']]
aggloclust = AgglomerativeClustering(n_clusters=3).fit(X)
plt.figure(figsize=(10, 5))
labels = aggloclust.labels_
plt.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=labels, cmap="Set2")
plt.xlabel('RentBikeCount')
plt.ylabel('Temperature(°C)')
plt.title("3 Cluster Agglomerative ")
plt.show()
finish = time.time()
# ------------------------------- Agglomerative Bike with Time Taken ------------------------------- #

Agglomerative_time_taken = finish - start

# ------------------------------- Agglomerative Bike with Davis-Bouldin score  ------------------------------- #
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['Temperature']
x = data[['RentBikeCount']]

n_clusters = 3
model = AgglomerativeClustering(n_clusters=3)
# fit model and predict clusters
yhat_2 = model.fit_predict(x)

Agglomerative_Davis_Bouldin_score = davies_bouldin_score(x, yhat_2)

# ------------------------------- Agglomerative with CSM ------------------------------- #
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['Temperature']

x = data[['RentBikeCount']]

agg_avg = AgglomerativeClustering(linkage='average', n_clusters=3)
as_avg = agg_avg.fit(x)
Agglomerative_silhouette_avg = silhouette_score(x, as_avg.labels_, metric='euclidean')

# ------------------------------- DBSCAN Bike with clusters ------------------------------- #
start = time.time()

file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

data = data[
    ['Date', 'RentBikeCount', 'Hour', 'Temperature', 'Humidity', 'Windspeed', 'Visibility', 'Dewpointtemperature',
     'SolarRadiation', 'Rainfall', 'Seasons', 'Holiday', 'FunctioningDay']]

data = data[['RentBikeCount', 'Temperature']]
X = np.nan_to_num(data)
X = np.array(X, dtype=np.float64)
X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.4, min_samples=5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
data['Clus_Db'] = db.labels_

realClusterNum = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
clusterNum = len(set(db.labels_))

plt.figure(figsize=(30, 20))
unique_labels = set(db.labels_)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
class_member_mask = (db.labels_ == k)
xy = X[class_member_mask & core_samples_mask]
plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
         markeredgecolor='k', markersize=14)
xy = X[class_member_mask & ~core_samples_mask]
plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
         markeredgecolor='k', markersize=6)
plt.title('Estimated Number of Clusters: %d' % realClusterNum, fontweight='bold', fontsize=20)
plt.legend(fontsize=20)
n_noise_ = list(db.labels_).count(-1)
print('number of noise(s): ', n_noise_)
plt.xlabel('RentBikeCount')
plt.ylabel('Temperature(°C)')
plt.show()
finish = time.time()
# ------------------------------- DBSCAN Bike with Time Taken ------------------------------- #

DBSCAN_time_taken = finish - start

# ------------------------------- DBSCAN Bike with Davis-Bouldin score  ------------------------------- #
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['Temperature']
x = data[['RentBikeCount']]

db = DBSCAN(eps=0.3, min_samples=10).fit(x)

DBSCAN_Davis_Bouldin_score = davies_bouldin_score(x, db.labels_)

# ------------------------------- DBSCAN with CSM ------------------------------- #
file = 'SeoulBikeData_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['Temperature']
x = data[['RentBikeCount']]
db = DBSCAN(eps=0.3, min_samples=10).fit(x)

DBSCAN_silhouette_avg = metrics.silhouette_score(x, db.labels_)

# ------------------------------- Tabulate Form ------------------------------- #

all_data = [["CSM", "Davis Bouldin Score", "TimeTaken"],
            [("No.Cluster", Kmean_n_clusters_Bike, "Avg", kmeans_silhouette_avg), kmeans_Davis_Bouldin_score,
             (Kmean_time_taken, "Secs in Kmean")],
            [("No.Cluster", Kmean_n_clusters_Bike, "Avg", Agglomerative_silhouette_avg),
             Agglomerative_Davis_Bouldin_score,
             (Agglomerative_time_taken, "Secs in Agglomerative")],
            [("No.Cluster", Kmean_n_clusters_Bike, "Avg", DBSCAN_silhouette_avg),
             DBSCAN_Davis_Bouldin_score,
             (DBSCAN_time_taken, "Secs in DBscan")]
            ]

print(tabulate(all_data, headers='firstrow', tablefmt='grid'))
