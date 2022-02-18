#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[70]:


df = pd.read_csv(r"C:\Users\MRIDHU\Downloads\archive (7)\Bengaluru_House_Data.csv")
df.head()


# In[71]:


df.shape


# In[72]:


df.groupby('area_type')['area_type'].agg('count')


# In[73]:


df1 = df.drop(['area_type','availability','balcony','society'],axis='columns')
df1.shape


# In[74]:


df1.head()


# In[75]:


df1.isnull().sum()


# In[76]:


import math 
bath_median = math.floor(df.bath.median())
bath_median


# In[77]:


df1.bath = df1.bath.fillna(bath_median)


# In[78]:


df2 = df1.dropna()
df2.isnull().sum()


# In[79]:


df2['size'].unique()


# In[80]:


df2['BHK'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))


# In[81]:


df2.head()


# In[82]:


df3 = df2.drop(['size'],axis='columns')


# In[83]:


df3.head()


# In[84]:


df3[df3.BHK>15]


# In[85]:


df3.total_sqft.unique()


# In[86]:


def is_float(x):
    try:
        float(x)
    except:
     return False
    return True


# In[87]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[88]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
       return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
       return None


# In[89]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# In[90]:


df4.loc[30]


# In[91]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']


# In[92]:


df5.head()


# In[93]:


len(df5['location'].unique())


# In[94]:


df5.location = df5.location.apply(lambda x:x.strip())


# In[95]:


location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[96]:


len(location_stats[location_stats<=10])


# In[97]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[98]:


df5.location = df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)


# In[99]:


len(df5.location.unique())


# In[100]:


df5[df5.total_sqft/df5.BHK<300].head()


# In[101]:


df5.shape


# In[102]:


df6 = df5[~(df5.total_sqft/df5.BHK<300)]
df6.shape


# In[103]:


df6.price_per_sqft.describe()


# In[104]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
 
df7 = remove_pps_outliers(df6)
df7.shape


# In[105]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 BHK',s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"KR Puram")


# In[109]:


def remove_BHK_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        BHK_stats = {}
        for BHK, BHK_df in location_df.groupby('BHK'):
            BHK_stats[BHK] = {
                'mean' : np.mean(BHK_df.price_per_sqft),
                'std' : np.std(BHK_df.price_per_sqft),
                'count' : BHK_df.shape[0]
            }
            for BHK, BHK_df in location_df.groupby('BHK'):
                stats = BHK_stats.get(BHK-1)
                if stats and stats['count']>5:
                    exclude_indices = np.append(exclude_indices, BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
        return df.drop(exclude_indices,axis='index')
df8 = remove_BHK_outliers(df7)
df8.shape


# In[110]:


df8.bath.unique()


# In[111]:


df8[df8.bath>10]


# In[112]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel('Number of bathrooms')
plt.ylabel('Count')


# In[113]:


df8[df8.bath>df8.BHK+2]


# In[114]:


df9 = df8[df8.bath<df8.BHK+2]
df9.shape


# In[115]:


df10 = df9.drop(['price_per_sqft'],axis='columns')
df10.shape


# In[116]:


df10.head(3)


# In[117]:


dummies = pd.get_dummies(df10.location) 
dummies.head()


# In[118]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head(3)


# In[119]:


df12 = df11.drop('location',axis='columns')
df12.head(3)


# In[120]:


df12.shape


# In[121]:


X = df12.drop('price',axis='columns')
X.head()


# In[122]:


y = df12.price
y.head(3)


# In[123]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[127]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
reg.score(X_test,y_test)


# In[128]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(),X,y,cv=cv)


# In[129]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model' : LinearRegression(),
            'params' : {
                'normalize' : [True,False]
            }
        },
        'lasso' : {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1,2],
                'selection' : ['random','cyclic']
            }
        },
        'decision_tree' : {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse','friedman_mse'],
                'splitter' : ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model' : algo_name,
            'best_score' : gs.best_score_,
            'best_params' : gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[130]:


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return reg.predict([x])[0]


# In[132]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[134]:


predict_price('1st Phase JP Nagar',1000,3,3)


# In[137]:


predict_price('Indira Nagar',1000,2,2)


# In[139]:


predict_price('Indira Nagar',1000,3,3)


# In[141]:


import pickle
with open('bangalore_home_prices_model.pickle','wb') as f:
    pickle.dump(reg,f)


# In[142]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




