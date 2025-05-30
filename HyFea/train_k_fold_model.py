#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import gc
import json
import os
import numpy as np
import pandas as pd
# import lightgbm as lgbm
from datetime import datetime
from time import gmtime, strftime
from scipy import stats
# from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.stats import norm, skew #for some statistics
# from scipy.special import boxcox1p
# from scipy.stats import boxcox_normmax
# from sklearn.preprocessing import StandardScaler
# In[ ]:


# all_data = pd.read_csv('../both_dataset_fea_traintype.csv')
all_data = pd.read_csv('../feature_data_530.csv', low_memory=False)
# columns = ['Title_len', 'Title_number', 'Alltags_len', 'Alltags_number', 'photo_count', 'totalTags', 'totalGeotagged', 'totalFaves',
#           'totalInGroup','photoCount','meanView', 'meanTags', 'meanFaves', 'followerCount','followingCount']
# all_data[columns] = all_data[columns].apply(lambda x: (x-x.mean())/x.std())
# KNN_FEA_COLUMNS = ['Title_number', 'Alltags_number', 'totalTags', 'totalFaves', 'totalInGroup', 'meanView', 'followerCount','followingCount']
# ['knn1_Title_number', 'knn1_Alltags_number', 'knn1_totalTags','knn1_totalFaves','knn1_totalInGroup','knn1_meanView', 'knn1_followerCount','knn1_followingCount']
# ['knn2_Title_number', 'knn2_Alltags_number', 'knn2_totalTags','knn2_totalFaves','knn2_totalInGroup','knn2_meanView', 'knn2_followerCount','knn2_followingCount']
# ['mean5_Title_number', 'mean5_Alltags_number', 'mean5_totalTags','mean5_totalFaves','mean5_totalInGroup','mean5_meanView', 'mean5_followerCount','mean5_followingCount']
# ['mean9_Title_number', 'mean9_Alltags_number', 'mean9_totalTags','mean9_totalFaves','mean9_totalInGroup','mean9_meanView', 'mean9_followerCount','mean9_followingCount']
# glove
glove_tags = pd.read_csv('../alltags_feature.csv')
# glove_text = glove_tags.values()

glove_title = pd.read_csv('../title_feature.csv')
# KNN_fea = pd.read_csv('../KNN_fea.csv')
# useless_columns = ['Title_number', 'Alltags_number', 'totalTags', 'totalFaves', 'totalInGroup', 'meanView', 'followerCount','followingCount']
# useless_columns += ['mean5_Title_number', 'mean5_Alltags_number', 'mean5_totalTags','mean5_totalFaves','mean5_totalInGroup','mean5_meanView', 'mean5_followerCount','mean5_followingCount']
# useless_columns += ['mean9_Title_number', 'mean9_Alltags_number', 'mean9_totalTags','mean9_totalFaves','mean9_totalInGroup','mean9_meanView', 'mean9_followerCount','mean9_followingCount']
# KNN_fea = KNN_fea.drop(useless_columns, axis=1)

all_data = pd.concat([all_data, glove_tags, glove_title], axis=1)
# del KNN_fea
# mm_fea = pd.read_csv('../vilt_fea_flickr.csv')
# mm_fea = pd.read_csv('../preRoberta_fea.csv')
# all_data = pd.concat([all_data, mm_fea], axis=1)

# all_data = pd.concat([all_data, glove_tags, glove_title], axis=1)

# pseudo_label = json.load(open('../results/BEST_725.json', encoding="utf-8"))
###############pseudo########################
pseudo_label = json.load(open('../75428.json', encoding="utf-8"))
test_resuts = [sample['popularity_score'] for sample in pseudo_label['result']]
test_results_df = pd.DataFrame(test_resuts)
test_results_df.columns = ['label']
########################################
# In[ ]:
## 1.对数据进行标准化噻

## 2.将文本特征作为embedding_features
## 3.试试看更深的树以及
columns = ['Title_len', 'Title_number', 'Alltags_len', 'Alltags_number', 'photo_count', 'totalTags', 'totalGeotagged', 'totalFaves',
          'totalInGroup','photoCount','meanView', 'meanTags', 'meanFaves', 'followerCount','followingCount']



skew_features = all_data[columns].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[abs(skew_features) > 0.75]
skew_index = high_skew.index
for i in skew_index:
    all_data[i] = np.log1p(all_data[i])


train_all_data = all_data[all_data['train_type'] != -1]
submit_all_data = all_data[all_data['train_type'] == -1]

del all_data

# indexxx = pd.read_csv('../index.csv')
# train_all_data = train_all_data.iloc[indexxx['index']]

## 待会儿试下label_encoder对各个列别，以及试试把user_features改为text_features，以及采用KNN

## norm the training set distribution
# train_all_data['label'] = train_all_data['label'] * 0.80
# train_all_data.loc[train_all_data['label'] == 0.80] = 1.0

train_all_data = train_all_data.reset_index(drop=True)
submit_all_data = submit_all_data.reset_index(drop=True)
###########pseudo##########
submit_all_data['label'] = test_results_df['label']
###############pseudo#########
# ss1 = [ss*0.95 for ss in train_all_data['Alltags_len']]
# ss2 = [ss*0.95 for ss in train_all_data['Alltags_number']]
# train_all_data['Alltags_len'] = pd.DataFrame(ss1, columns=['Alltags_len'])
# train_all_data['Alltags_number'] = pd.DataFrame(ss2, columns=['Alltags_number'])

#####################pseudo##############
# samples = submit_all_data.sample(n=50000, replace=True, random_state=2020)
# samples = samples.reset_index(drop=True)
train_all_data = train_all_data.append(submit_all_data,  ignore_index=True)
train_all_data = train_all_data.reset_index(drop=True)
##################pseudo+++++++++++++++
# submit_all_data = submit_all_data.reset_index(drop=True)

feature_columns = ['Pid', 'train_type', 'label', 'mean_label']
feature_columns += ['user_fe_{}'.format(i) for i in range(399)]
feature_columns += ['loc_fe_{}'.format(i) for i in range(400)]

train_label_df = train_all_data[['Pid', 'label']]
train_feature_df = train_all_data.drop(feature_columns, axis=1)

submit_label_df = submit_all_data[['Pid', 'label']]
submit_feature_df = submit_all_data.drop(feature_columns, axis=1)
del train_all_data
del submit_all_data
gc.collect()

print(len(train_feature_df), len(submit_feature_df), len(train_feature_df.columns))
print(len(train_label_df), len(submit_label_df), len(train_feature_df.columns))


# In[ ]:


from catboost import CatBoostRegressor, Pool

cb_params = {
    # 'objective': 'Tweedie[:variance_power = 1.5] ',
    # Tweedie:variance_power = 1.5,
    # 'loss_function':'LogCosh',
# FairLoss
#     'objective':'RMSE',
    'loss_function':'Huber:delta=0.8',
    'eval_metric': 'MAE',
    'learning_rate': 0.03,
    'l2_leaf_reg': 3,
    'max_ctr_complexity': 1,
    'depth': 8,
    'leaf_estimation_method': 'Gradient',
    'use_best_model': True,
    'iterations': 200000,
    'early_stopping_rounds': 5000,
    'verbose': 1000,
    'task_type': 'GPU',
    'devices':'0'
}

 # ['Pid', 'Uid', 'titlelen', 'tagcount', 'avgview', 'groupcount', 'avgmembercount', 'haspeople', 'year', 'month', 'day', 'label']
cate_cols = ['Uid', 'Category', 'Subcategory', 'Concept', 'Mediatype', 'hour', 'day', 'weekday', 'week_hour',
             'Geoaccuracy', 'ispro' , 'Ispublic', 'img_model']
# cate_cols = ['Uid', 'year', 'month', 'day']
submit_data = Pool(data=submit_feature_df, label=submit_label_df['label'], cat_features=cate_cols)


valid_ans = []
submit_proba = []
kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
k = 0

for train_idx, valid_idx in kfold.split(train_feature_df, train_label_df):
    fold_train_x, fold_train_y = train_feature_df.loc[train_idx], train_label_df['label'].loc[train_idx]
    fold_valid_x, fold_valid_y = train_feature_df.loc[valid_idx], train_label_df['label'].loc[valid_idx]
    
    train_data = Pool(data=fold_train_x, label=fold_train_y, cat_features=cate_cols)
    valid_data = Pool(data=fold_valid_x, label=fold_valid_y, cat_features=cate_cols)
    
    cb_model=CatBoostRegressor(**cb_params)
    cb_model.fit(train_data, eval_set=valid_data)
    
    valid_pred = cb_model.predict(valid_data)
    valid_mse = mean_squared_error(fold_valid_y, valid_pred)
    valid_mae = mean_absolute_error(fold_valid_y, valid_pred)
    valid_src = stats.spearmanr(fold_valid_y, valid_pred)[0]
    
    print("MSE: %.4f, MAE: %.4f, SRC: %.4f"%(valid_mse, valid_mae, valid_src))
    valid_ans.append([valid_mse, valid_mae, valid_src])
        
    submit_pred = cb_model.predict(submit_data)
    submit_proba.append(submit_pred)
    
    cb_model.save_model('./save_model/KFold_catboost_' + str(k) + '.pkl')
    k += 1

valid_ans = np.mean(valid_ans, axis=0)

print("valid: MSE: %.4f, MAE: %.4f, SRC: %.4f"%(valid_ans[0], valid_ans[1], valid_ans[2]))

# save result json
submit_ans = np.mean(submit_proba, axis=0)
result = pd.DataFrame()
result['post_id'] = submit_label_df['Pid'].apply(lambda x: 'post' + str(x))
result['popularity_score'] = submit_ans.round(decimals=4)

out_json = dict()
out_json["version"] = "VERSION 5.28"
out_json["result"] = result.to_dict(orient='records')
out_json["external_data"] = {"used": "true", "details": "catboost"}
f = open('Best_KFold_catboost.json', "w")
json.dump(out_json, f)
f.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




