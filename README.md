# Madhavan
DATA SOURCE:
import numpy as np import pandas as pd
import matplotlib.pyplot as plt

import os print(os.listdir("../input"))


FEATURE EXPLORATION:


Index(['Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Depth Error', 'Depth Seismic Stations', 'Magnitude', 'Magnitude Type',
'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 'Horizontal Error', 'Root Mean Square', 'ID', 'Source', 'Location Source', 'Magnitude Source', 'Status'], dtype='object')
Figure out the main features from earthquake data and create a object of that features, namely, Date, Time, Latitude, Longitude, Depth, Magnitude.


data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']] data.head()
VISUALIZATION:
from mpl_toolkits.basemap import Basemap

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=- 180,urcrnrlon=180,lat_ts=20,resolution='c')

longitudes = data["Longitude"].tolist() latitudes = data["Latitude"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
#resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.) x,y = m(longitudes,latitudes)
In [9]:
fig = plt.figure(figsize=(12,10)) plt.title("All affected areas")
m.plot(x, y, "o", markersize = 2, color = 'blue') m.drawcoastlines() m.fillcontinents(color='coral',lake_color='aqua') m.drawmapboundary()
m.drawcountries() plt.show()

TRAINING AND EVALUATION


# demonstrate that the train-test split procedure is repeatable from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split # create dataset
X, y = make_blobs(n_samples=100) # split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) # summarize first 5 rows
print(X_train[:5, :])
# split again, and we should see the same split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) # summarize first 5 rows
print(X_train[:5, :])


HYPERPARAMETER TUNING

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model, verbose=0)

# param_grid = {
#     "neurons": [16, 64], 
#     "batch_size": [10, 20], 
#     "epochs": [10],
#     "activation": ['sigmoid', 'relu'],
#     "optimizer": ['SGD', 'Adadelta'],
#     "loss": ['squared_hinge']
# }
param_grid = {
    "neurons": [16], 
    "batch_size": [10, 20], 
    "epochs": [10],
    "activation": ['sigmoid', 'relu'],
    "optimizer": ['SGD', 'Adadelta'],
    "loss": ['squared_hinge']
}

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

best_params = grid_result.best_params_
best_params

FEATURE ENGINEERING
data['Year'] = data['Date'].apply(lambda x: x[-4:])

data = data.drop('Date', axis=1)


data['Month'] = data['Month'].astype(np.int)
invalid_year_indices = data[data['Year'].str.contains('Z')].index

data = data.drop(invalid_year_indices, axis=0).reset_index(drop=True)


data['Year'] = data['Year'].astype(np.int)


data['Hour'] = data['Time'].apply(lambda x: np.int(x[0:2]))

data = data.drop('Time', axis=1)


 categorical_vars = ["land_surface_condition", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status"] for var in categorical_vars:
df = fuck_naman(df, var)

df.to_csv('data/labeled_train.csv') df["damage_grade"]

2	148259
3	87218
1	25124
Name: damage_grade, dtype: int64
# Pie chart
labels = ['Damage 1', 'Damage 2', 'Damage 3']
sizes = [25124, 148259, 87218,]
# only "explode" the 2nd slice (i.e. 'Hogs') explode = (0, 0, 0)
fig1, ax1 = plt.subplots()
patches, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
for text in texts: text.set_color('white') text.set_size(13)
for autotext in autotexts: autotext.set_color('white') autotext.set_size(13)
#draw circle
centre_circle = plt.Circle((0,0),0.80,fc='black') fig = plt.gcf() fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle ax1.axis('equal')
plt.tight_layout() plt.show()
value_counts()

normalized_df=(df-df.min())/(df.max()-df.min()) df = normalized_df

X = df.drop("damage_grade", axis=1) y = df["damage_grade"]
targets = df["damage_grade"].unique() print(targets)
pca = PCA(n_components=2) X_r = pca.fit(X).transform(X) print(X_r.shape)
PCA_Df = pd.DataFrame(data = X_r
, columns = ['principal component 1', 'principal component 2']) print(PCA_Df.head())
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors): indicesToKeep = df['damage_grade'] == target
plt.scatter(PCA_Df.loc[indicesToKeep, 'principal component 1']
, PCA_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50) plt.legend((targets + .5) * 2)
plt.title('PCA of Damage dataset')
plt.xlabel("First Principal Component") plt.ylabel('Second Principal Component')
[0.5 1. 0. ]
(260601, 2)
principal component 1 principal component 2
0	0.029283	-0.653984
1	-0.605595	-0.171097
2	-0.417904	1.041256
3	-0.719406	-0.306778
4	-0.102610	-0.187304

Text(0, 0.5, 'Second Principal Component')

X = df.drop("damage_grade", axis=1) y = df["damage_grade"]
targets = df["damage_grade"].unique() print(targets)
pca = PCA(n_components=3) X_r = pca.fit(X).transform(X) print(X_r.shape)

 PCA_Df = pd.DataFrame(data = X_r[0:2000]
, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
colors = ['r', 'g', 'b'] fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') for target, color in zip(targets,colors):
indicesToKeep = df['damage_grade'] == target ax.scatter(PCA_Df.loc[indicesToKeep, 'principal component 1']
, PCA_Df.loc[indicesToKeep, 'principal component 2'], PCA_Df.loc[indicesToKeep, 'principal component 3'], c = color, s = 50) plt.legend((targets + .5) * 2)
plt.title('3-Component PCA of Damage ') plt.xlabel("First Principal Component") plt.ylabel('Second Principal Component')
[0.5 1. 0. ]
(260601, 3)
Text(0.5, 0, 'Second Principal Component')
X = df.drop("damage_grade", axis=1).astype('int') y = ((df["damage_grade"] + 0.5) * 2).astype('int') lda = LDA(n_components=2)
dmg_lda = lda.fit_transform(X, y) print(dmg_lda)
l_x = dmg_lda[:,0] l_y = dmg_lda[:,1]
cdict={1:'red',2:'green',3:'blue'}
labl={1:'Class1',2:'Class2',3:'Class3'} for l in np.unique(y):
ix=np.where(y==l)
ax = plt.scatter(l_x[ix],l_y[ix],c=cdict[l],s=40, label=labl[l])
plt.title("LDA Analysis") plt.legend()

/home/jordanrodrigues/anaconda3/lib/python3.7/site- packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
warnings.warn("Variables are collinear.")
[[-0.39785383 -1.70783617]
[ 0.42992898 -0.02751219]
[ 0.87874376 0.49585697]
...
[ 0.61089644 0.2825365 ]
[ 0.75307395 0.58296292]
[ 1.70574956 7.67655413]]


<matplotlib.legend.Legend at 0x7f6fe9e9d908>
labels = ["MaxAbsScaler RandomForest", "MaxAbsScaler SGD", "StandardScaler ExtremeTrees", "MaxAbsScaler BernoulliNaiveBayes", "MaxAbsScaler ExtremeTrees"]

values = [.3720, .4521, .2893, .4397, .3116]

plt.bar(labels, values, color='tab:cyan') plt.xticks(rotation=70) plt.rcParams.update({'font.size': 15}) plt.title("AutoML Performance") plt.ylabel("F1 Score")


Text(0, 0.5, 'F1 Score')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
#data.info()
data.head()

VISUAL EXPLOTARY DATA ANALYSIS:
data.boxplot(column='Attack',by = 'Legendary')
# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
<matplotlib.axes._subplots.AxesSubplot at 0x7eb7c84999e8>

BAR PLOT:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline


kill.head()
# Most common 15 Name or Surname of killed people
separate = kill.name[kill.name != 'TK TK'].str.split() 
a,b = zip(*separate)                    
name_list = a+b                         
name_count = Counter(name_list)         
most_common_names = name_count.most_common(15)  
x,y = zip(*most_common_names)
x,y = list(x),list(y)
# 
plt.figure(figsize=(10,5))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')
sorted_data2 = data.reindex(new_index)
# high school graduation rate vs Poverty rate of each state
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])
data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)
data.sort_values('area_poverty_ratio',inplace=True)

# visualize
f,ax1 = plt.subplots(figsize =(10,5))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)
plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('States',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')
plt.grid()
VISUALIZATION TOOLS:
Parallel Plots (Pandas):
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_venn as venn
from math import pi
from pandas.tools.plotting import parallel_coordinates
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")

linkcode
data = pd.read_csv('../input/iris/Iris.csv')
data = data.drop(['Id'],axis=1)
# Make the plot
plt.figure(figsize=(15,10))
parallel_coordinates(data, 'Species', colormap=plt.get_cmap("Set1"))
plt.title("Iris data class visualization according to features (setosa, versicolor, virginica)")
plt.xlabel("Features of data set")
plt.ylabel("cm")
plt.savefig('graph.png')
plt.show()
TESTING WITH ASSERTS:
# Lets chech Type 2
data["Type 2"].value_counts(dropna =False) 
# As you can see, there are 386 NAN value

NaN         386
Flying       97
Ground       35
Poison       34
Psychic      33
Fighting     26
Grass        25
Fairy        23
Steel        22
Dark         20
Dragon       18
Rock         14
Water        14
Ice          14
Ghost        14
Fire         12
Electric      6
Normal        4
Bug           3
Name: Type 2, dtype: int64
 # Lets drop nan values
data1=data.copy()   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
assert  data1['Type 2'].notnull().all() # returns nothing because we drop nan values
data1["Type 2"].fillna('empty',inplace = True) # ıstersen empty ıle de doldurabılırız
# # With assert statement we can check a lot of thing. For example
# assert data.columns[1] == 'Name'
# assert data.Speed.dtypes == np.int



data[data['Year'].str.contains('Z')]
