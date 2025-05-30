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
from scipy import stats
from scipy.stats import norm, skew #for some statistics
# import matplotlib.pyplot as plt
# # import stats
# from catboost import CatBoostRegressor, Pool



# In[ ]:
# def multiplier(results):
#     import numpy as np
#     list_all = np.linspace(0, 1.80581, 180581)
#     norm2 = [results[i] - list_all[i] for i in range(len(list_all))]
#     for ss in range(len(norm2)):
#         if norm2[ss] < 1:
#             norm2[ss] = 1
#     return norm2
    # list_all



# all_data = pd.read_csv('../data/rolling_combine91.csv',low_memory=False)
all_data = pd.read_csv('../feature_data_530.csv',low_memory=False)
# glove
glove_tags = pd.read_csv('../alltags_feature.csv')
glove_title = pd.read_csv('../title_feature.csv')
title_simi = pd.read_csv('../alltitle_simi.csv')
tags_simi = pd.read_csv('../alltags_simi.csv')
all_data = pd.concat([all_data, glove_tags, glove_title,title_simi,tags_simi], axis=1)

# aug_data = pd.read_csv('../train_data_aug.csv',low_memory=False)
#
columns = ['Title_len', 'Title_number', 'Alltags_len', 'Alltags_number', 'photo_count', 'totalTags', 'totalGeotagged', 'totalFaves',
          'totalInGroup','photoCount','meanView', 'meanTags', 'meanFaves', 'followerCount','followingCount']



skew_features = all_data[columns].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[abs(skew_features) > 0.75]
skew_index = high_skew.index
for i in skew_index:
    all_data[i] = np.log1p(all_data[i])


#

del glove_tags
del glove_title

train_all_data = all_data[all_data['train_type'] != -1]
submit_all_data = all_data[all_data['train_type'] == -1]
del all_data

train_all_data = train_all_data.reset_index(drop=True)
submit_all_data = submit_all_data.reset_index(drop=True)


# train_all_data = train_all_data.append(submit_all_data,  ignore_index=True)
# train_all_data = train_all_data.reset_index(drop=True)
# submit_all_data = submit_all_data.reset_index(drop=True)

feature_columns = ['Pid', 'train_type', 'mean_label','label']
feature_columns += ['user_fe_{}'.format(i) for i in range(399)]
feature_columns += ['loc_fe_{}'.format(i) for i in range(400)]

train_label_df = train_all_data[['Pid', 'label']]
train_feature_df = train_all_data.drop(feature_columns, axis=1)

submit_label_df = submit_all_data[['Pid', 'label']]
submit_feature_df = submit_all_data.drop(feature_columns, axis=1)
del train_all_data
del submit_all_data
gc.collect()




# submit_all_data = pd.concat([submit_all_data, ustc_test], axis=1)
# feature_columns = ['Pid', 'train_type', 'mean_label', 'Pathalias','Mediastatus','Alltags', 'Title', 'Postdate','photo_firstdatetaken','location_description', 'user_description','canbuypro', 'timezone_timezone_id', 'photo_firstdate', 'timezone_offset', 'img', 'img_file','datetime','label']
#feature_columns += ['user_fe_{}'.format(i) for i in range(399)]
#feature_columns += ['loc_fe_{}'.format(i) for i in range(400)]
#feature_columns += target_columns

# train_label_df = train_all_data[['Pid', 'label']]
# train_label_df = train_label_df.append(aug_data[['Pid', 'label']], ignore_index=True)
# columns = ['Title_len', 'Title_number','Alltags_len','Alltags_number', 'photo_count', 'totalTags', 'totalFaves',
#           'totalInGroup','photoCount','meanView', 'meanTags', 'meanFaves', 'followerCount','followingCount']
# skew_features = all_data[columns].apply(lambda x: skew(x)).sort_values(ascending=False)
# high_skew = skew_features[abs(skew_features) > 0.75]
# skew_index = high_skew.index
# for i in skew_index:
#     all_data[i] = np.log1p(all_data[i])

# train_label_df = train_all_data[['label']]
# train_label_df = train_label_df.append(aug_data[['label']], ignore_index=True)
#
# train_feature_df = train_all_data.drop(feature_columns, axis=1)
# train_feature_df = train_feature_df.append(aug_data, ignore_index=True)
# train_feature_df = train_feature_df.drop(['label'], axis=1)

# skew_features = train_feature_df[columns].apply(lambda x: skew(x)).sort_values(ascending=False)
# high_skew = skew_features[abs(skew_features) > 0.75]
# skew_index = high_skew.index
# for i in skew_index:
#     train_feature_df[i] = np.log1p(train_feature_df[i])
# all_data = pd.concat([all_data, glove_tags, glove_title], axis=1)
# train_feature_df['week_hour'] = pd.DataFrame(train_feature_df['week_hour'], dtype='int')
# train_feature_df['year_weekday'] = pd.DataFrame(train_feature_df['year_weekday'], dtype='int')
# data_aug['weekday'] = pd.DataFrame(data_aug['weekday'], dtype='int')

# submit_label_df = submit_all_data[['Pid', 'label']]
# submit_feature_df = submit_all_data.drop(feature_columns, axis=1)
# submit_feature_df = submit_feature_df.drop(['label'], axis=1)
# for i in skew_index:
#     submit_feature_df[i] = np.log1p(submit_feature_df[i])


# submit_all_data = submit_all_data.reset_index(drop=True)




gc.collect()




print(len(train_feature_df), len(submit_feature_df), len(train_feature_df.columns))
print(len(train_label_df), len(submit_label_df), len(train_feature_df.columns))


# cate_cols = ['Uid', 'Category', 'Subcategory', 'Concept', 'Mediatype', 'hour', 'day', 'weekday', 'week_hour', 'year_weekday','Geoaccuracy', 'ispro' , 'Ispublic']
#####################################################################
cate_cols = ['Uid', 'Category', 'Subcategory', 'Concept', 'Mediatype', 'hour', 'day', 'weekday', 'week_hour',
             'Geoaccuracy', 'ispro' , 'Ispublic', 'img_model']


cb_params = {
    # 'objective': 'RMSE',
    # 'loss_function' : 'RMSEWithUncertainty',
    # 'posterior_sampling' : True,
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
    'verbose': 500,
    'task_type': 'GPU',
    'devices':'0'
}


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

    cb_model.save_model('../data/save_model/KFold_catboost_' + str(k) + '.pkl')

    k += 1

valid_ans = np.mean(valid_ans, axis=0)
print("valid: MSE: %.4f, MAE: %.4f, SRC: %.4f"%(valid_ans[0], valid_ans[1], valid_ans[2]))

# save result json
submit_ans = np.mean(submit_proba, axis=0)
# multiplier_re = multiplier(submit_ans)
result = pd.DataFrame()
result['post_id'] = submit_label_df['Pid'].apply(lambda x: 'post' + str(x))
result['popularity_score'] = submit_ans.round(decimals=4)

out_json = dict()
out_json["version"] = "train_data_augmentation"
out_json["result"] = result.to_dict(orient='records')
out_json["external_data"] = {"used": "true", "details": "catboost"}
f = open('train_augment.json', "w")
json.dump(out_json, f)
f.close()




