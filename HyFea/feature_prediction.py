import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgbm
from datetime import datetime
from time import gmtime, strftime
from scipy import stats
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from catboost import Pool, cv, CatBoostClassifier
# from bayes_opt import BayesianOptimization
from sklearn.model_selection import * 
from sklearn.metrics import *
import gc
# import matplotlib.pyplot as plt
# # import stats
# from catboost import CatBoostRegressor, Pool


# In[ ]:


# all_data = pd.read_csv('../data/rolling_combine91.csv',low_memory=False)
all_data = pd.read_csv('../feature_pathalias_data.csv',low_memory=False)
# glove
glove_tags = pd.read_csv('../alltags_feature.csv')
glove_title = pd.read_csv('../title_feature.csv')
all_data = pd.concat([all_data, glove_tags, glove_title], axis=1)
# all_data = all_data.sort_values(by="datetime", ascending=True)
# columns = ['Title_len', 'Title_number','Alltags_len','Alltags_number', 'photo_count', 'totalTags', 'totalFaves',
#           'totalInGroup','photoCount','meanView', 'meanTags', 'meanFaves', 'followerCount','followingCount']
columns = ['photo_count', 'totalTags', 'totalFaves','totalInGroup','photoCount','meanView', 'meanTags', 'meanFaves', 'followerCount','followingCount','totalGeotagged']
# 'Title_len', 'Title_number','Alltags_len','Alltags_number', 
pseudo_label = json.load(open('../best7225.json', encoding="utf-8"))
test_resuts = [sample['popularity_score'] for sample in pseudo_label['result']]
test_results_df = pd.DataFrame(test_resuts)
test_results_df.columns = ['label']
#
#NEXT STEP USING PSEUDO_LABEL
#mean_View = json.load(open('./meanView1.json', encoding="utf-8"))
#meanView_resuts = [sample['popularity_score'] for sample in mean_View['result']]
#meanView_resuts_df = pd.DataFrame(meanView_resuts)
#meanView_resuts_df.columns = ['meanView']
#
#mean_Fave = json.load(open('./meanFaves1.json', encoding="utf-8"))
#mean_Fave_resuts = [sample['popularity_score'] for sample in mean_Fave['result']]
#mean_Fave_resuts_df = pd.DataFrame(mean_Fave_resuts)
#mean_Fave_resuts_df.columns = ['meanFaves']
#
#mean_Tags = json.load(open('./meanTags1.json', encoding="utf-8"))
#mean_Tags_resuts = [sample['popularity_score'] for sample in mean_Tags['result']]
#mean_Tags_resuts_df = pd.DataFrame(mean_Tags_resuts)
#mean_Tags_resuts_df.columns = ['meanTags']
#
#
totalInGroup = json.load(open('../totalInGroup1.json', encoding="utf-8"))
totalInGroup_resuts = [sample['popularity_score'] for sample in totalInGroup['result']]
totalInGroup_resuts_df = pd.DataFrame(totalInGroup_resuts)
totalInGroup_resuts_df.columns = ['totalInGroup']

followerCount = json.load(open('../followerCount1.json', encoding="utf-8"))
followerCount_resuts = [sample['popularity_score'] for sample in followerCount['result']]
followerCount_resuts_df = pd.DataFrame(followerCount_resuts)
followerCount_resuts_df.columns = ['followerCount']

followingCount = json.load(open('../followingCount1.json', encoding="utf-8"))
followingCount_resuts = [sample['popularity_score'] for sample in followingCount['result']]
followingCount_resuts_df = pd.DataFrame(followingCount_resuts)
followingCount_resuts_df.columns = ['followingCount']

photoCount = json.load(open('../photoCount1.json', encoding="utf-8"))
photoCount_resuts = [sample['popularity_score'] for sample in photoCount['result']]
photoCount_resuts_df = pd.DataFrame(photoCount_resuts)
photoCount_resuts_df.columns = ['photoCount']

# skew_features = all_data[columns].apply(lambda x: skew(x)).sort_values(ascending=False)
# high_skew = skew_features[abs(skew_features) > 0.75]
# skew_index = high_skew.index
for i in columns:
    all_data[i] = np.log1p(all_data[i])

del glove_tags
del glove_title

train_all_data = all_data[all_data['train_type'] != -1]
submit_all_data = all_data[all_data['train_type'] == -1]
del all_data

train_all_data = train_all_data.reset_index(drop=True)
submit_all_data = submit_all_data.reset_index(drop=True)

submit_all_data['label'] = test_results_df['label']
train_all_data = train_all_data.append(submit_all_data, ignore_index=True)
train_all_data = train_all_data.reset_index(drop=True)
del submit_all_data

train_data = train_all_data[train_all_data['Pathalias'] != 'None']
test_data = train_all_data[train_all_data['Pathalias'] == 'None']
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
#test_data['meanView'] = meanView_resuts_df['meanView']
#test_data['meanFaves'] = mean_Fave_resuts_df['meanFaves']
#test_data['meanTags'] = mean_Tags_resuts_df['meanTags']
test_data['totalInGroup'] = totalInGroup_resuts_df['totalInGroup']
test_data['followerCount'] = followerCount_resuts_df['followerCount']
test_data['followingCount'] = followingCount_resuts_df['followingCount']
#test_data['totalGeotagged'] = totalGeotagged_resuts_df['totalGeotagged']
test_data['photoCount'] = photoCount_resuts_df['photoCount']
#test_data['totalViews'] = test_data['meanView'] + test_data['photoCount']
#test_data['totalTags'] = test_data['meanTags'] + test_data['photoCount']
#test_data['totalFaves'] = test_data['meanFaves'] + test_data['photoCount']

new_all_data = train_data.append(test_data, ignore_index=True)
new_all_data = new_all_data.reset_index(drop=True)
del train_data
del test_data

train_all_data = new_all_data[new_all_data['train_type'] != -1]
submit_all_data = new_all_data[new_all_data['train_type'] == -1]
del new_all_data

train_all_data = train_all_data.reset_index(drop=True)
submit_all_data = submit_all_data.reset_index(drop=True)

#
#submit_all_data['label'] = test_results_df['label']
#train_all_data = train_all_data.append(submit_all_data, ignore_index=True)
#train_all_data = train_all_data.reset_index(drop=True)
#del submit_all_data
#
#train_data = train_all_data[train_all_data['Pathalias'] != 'None']
#test_data = train_all_data[train_all_data['Pathalias'] == 'None']
#train_data = train_data.reset_index(drop=True)
#test_data = test_data.reset_index(drop=True)
#test_data['meanView'] = meanView_resuts_df['meanView']
#test_data['meanFaves'] = mean_Fave_resuts_df['meanFaves']
#test_data['meanTags'] = mean_Tags_resuts_df['meanTags']
#test_data['totalInGroup'] = totalInGroup_resuts_df['totalInGroup']
#test_data['followerCount'] = followerCount_resuts_df['followerCount']
#test_data['followingCount'] = followingCount_resuts_df['followingCount']





# submit_all_data = pd.concat([submit_all_data, ustc_test], axis=1)
feature_columns = ['Pid', 'train_type', 'mean_label', 'Pathalias','Mediastatus','Alltags', 'Title', 'Postdate','photo_firstdatetaken','location_description', 'user_description','canbuypro', 'timezone_timezone_id', 'photo_firstdate', 'timezone_offset', 'img', 'img_file','datetime','label'] 
#feature_columns += ['user_fe_{}'.format(i) for i in range(399)]
#feature_columns += ['loc_fe_{}'.format(i) for i in range(400)]
#feature_columns += target_columns

train_label_df = train_all_data[['Pid', 'label']]
train_feature_df = train_all_data.drop(feature_columns, axis=1)

submit_label_df = submit_all_data[['Pid', 'label']]
submit_feature_df = submit_all_data.drop(feature_columns, axis=1)


# submit_all_data = submit_all_data.reset_index(drop=True)



gc.collect()




print(len(train_feature_df), len(submit_feature_df), len(train_feature_df.columns))
print(len(train_label_df), len(submit_label_df), len(train_feature_df.columns))


cate_cols = ['Uid', 'Category', 'Subcategory', 'Concept', 'Mediatype', 'hour', 'day', 'weekday', 'week_hour', 'year_weekday','Geoaccuracy', 'ispro' , 'Ispublic']
#####################################################################


cb_params = {
    'objective': 'RMSE',
    'eval_metric': 'MAE',
    'learning_rate': 0.03,
    'l2_leaf_reg': 3,
    'max_ctr_complexity': 1,
    'depth': 8,
    'leaf_estimation_method': 'Gradient',
    'use_best_model': True,
    'iterations': 200000,
    'early_stopping_rounds': 5000,
    'verbose': 500,
    'task_type': 'GPU',
    'devices':'0'
}


submit_data = Pool(data=submit_feature_df, label=submit_label_df['label'], cat_features=cate_cols)


valid_ans = []
submit_proba = []
kfold = KFold(n_splits=20, shuffle=True, random_state=2020)
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
    
    cb_model.save_model('../data/save_model/KFold_catboost_' + str(k) + '.pkl')

    k += 1

valid_ans = np.mean(valid_ans, axis=0)
print("valid: MSE: %.4f, MAE: %.4f, SRC: %.4f"%(valid_ans[0], valid_ans[1], valid_ans[2]))

# save result json
submit_ans = np.mean(submit_proba, axis=0)
result = pd.DataFrame()
result['post_id'] = submit_label_df['Pid'].apply(lambda x: 'post' + str(x))
result['popularity_score'] = submit_ans.round(decimals=4)

out_json = dict()
out_json["version"] = "photoCount"
out_json["result"] = result.to_dict(orient='records')
out_json["external_data"] = {"used": "true", "details": "catboost"}
f = open('label1.json', "w")
json.dump(out_json, f)
f.close()




