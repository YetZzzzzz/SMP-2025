import gc
import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool
from scipy import stats




all_data = pd.read_csv('../feature_data_530.csv')
# glove
glove_tags = pd.read_csv('../alltags_feature.csv')
glove_title = pd.read_csv('../title_feature.csv')
# clip_text = pd.read_csv('../clip_text_fea_512.csv')
# clip_image = pd.read_csv('../clip_image_fea_512.csv')
# vilt_flickr = pd.read_csv('../vilt_fea_flickr.csv')
# vilt_vqa = pd.read_csv('../vilt_fea_vqa.csv')
bert_fea = pd.read_csv('../bert_fea.csv')
bertweet_fea = pd.read_csv('../bertweet_fea.csv')
gptneo_fea = pd.read_csv('../gptneo_fea_2048.csv')
xlnet_fea = pd.read_csv('../xlm_roberta_fea_1024.csv')
all_data = pd.concat(
    [all_data, glove_tags, glove_title, clip_text, vilt_flickr, bert_fea, bertweet_fea, gptneo_fea, xlnet_fea], axis=1)
del glove_tags
del glove_title
del clip_text
del vilt_flickr
del bert_fea
del bertweet_fea
del gptneo_fea
del xlnet_fea



train_all_data = all_data[all_data['train_type'] != -1]
submit_all_data = all_data[all_data['train_type'] == -1]
del all_data

## load ustc train data
bert_title_avg_train = pd.read_csv('../Bert_tags+title_305613_average_3.csv')
fasttxt_tag_avg_train = pd.read_csv('../FastText_tags+title_305613_average_3.csv')
glove_tag_avg_train = pd.read_csv('../Glove_tags_305613_average_3.csv')
lsa_tags_title_train = pd.read_csv('../LSA_tags+title_305613_average_3.csv')
pca_tfidf_train = pd.read_csv('../pca_tfidf_train_512_windows3.csv')

ustc_train = pd.merge(bert_title_avg_train, fasttxt_tag_avg_train)
ustc_train = pd.merge(ustc_train, glove_tag_avg_train)
ustc_train = pd.merge(ustc_train, lsa_tags_title_train)
ustc_train = pd.merge(ustc_train, pca_tfidf_train)

train_all_data = train_all_data.reset_index(drop=True)
train_all_data = pd.concat([train_all_data, ustc_train], axis=1)
## free the dataframe to save RAM memory
del bert_title_avg_train
del fasttxt_tag_avg_train
del glove_tag_avg_train
del lsa_tags_title_train
del pca_tfidf_train
del ustc_train




## load ustc test data
bert_title_avg_test = pd.read_csv('../Bert_tags+title_180581_average_3.csv')
fasttxt_tag_avg_test = pd.read_csv('../FastText_tags+title_180581_average_3.csv')
glove_tag_avg_test = pd.read_csv('../Glove_tags_180581_average_3.csv')
lsa_tags_title_test = pd.read_csv('../LSA_tags+title_180581_average_3.csv')
pca_tfidf_test = pd.read_csv('../pca_tfidf_test_512_windows3.csv')



ustc_test = pd.merge(bert_title_avg_test, fasttxt_tag_avg_test)
ustc_test = pd.merge(ustc_test, glove_tag_avg_test)
ustc_test = pd.merge(ustc_test, lsa_tags_title_test)
ustc_test = pd.merge(ustc_test, pca_tfidf_test)

submit_all_data = submit_all_data.reset_index(drop=True)
submit_all_data = pd.concat([submit_all_data, ustc_test], axis=1)

##  free the dataframe to save RAM memory

del bert_title_avg_test
del fasttxt_tag_avg_test
del glove_tag_avg_test
del lsa_tags_title_test
del pca_tfidf_test
del ustc_test
gc.collect()


train_all_data = train_all_data.reset_index(drop=True)
submit_all_data = submit_all_data.reset_index(drop=True)

feature_columns = ['Pid', 'train_type', 'label', 'mean_label', 'pid', 'uid']
feature_columns += ['user_fe_{}'.format(i) for i in range(399)]
feature_columns += ['loc_fe_{}'.format(i) for i in range(400)]

train_label_df = train_all_data[['Pid', 'label']]
train_feature_df = train_all_data.drop(feature_columns, axis=1)

submit_label_df = submit_all_data[['Pid', 'label']]
submit_feature_df = submit_all_data.drop(feature_columns, axis=1)

print(len(train_feature_df), len(submit_feature_df), len(train_feature_df.columns))
print(len(train_label_df), len(submit_label_df), len(train_feature_df.columns))


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
    'border_count': 254,
    "gpu_ram_part": 0.9,
    "logging_level": "Debug",
    'verbose': 1000,
    'task_type': "GPU",
    'devices': '2,4,5,6'
}

cate_cols = ['Uid', 'Category', 'Subcategory', 'Concept', 'Mediatype', 'hour', 'day', 'weekday', 'week_hour',
             'year_weekday', 'Geoaccuracy', 'ispro', 'Ispublic', 'img_model']
submit_data = Pool(data=submit_feature_df, label=submit_label_df['label'], cat_features=cate_cols)

valid_ans = []
submit_proba = []
# fold_valid_x, fold_valid_y = train_feature_df, train_label_df['label']
valid_data = Pool(data=train_feature_df, label=train_label_df['label'], cat_features=cate_cols)

cb_model = CatBoostRegressor()
cb_model.load_model('./save_model/KFold_catboost_0_best.pkl')

valid_pred = cb_model.predict(valid_data)
valid_mse = mean_squared_error(train_label_df['label'], valid_pred)
valid_mae = mean_absolute_error(train_label_df['label'], valid_pred)
valid_src = stats.spearmanr(train_label_df['label'], valid_pred)[0]
print("MSE: %.4f, MAE: %.4f, SRC: %.4f" % (valid_mse, valid_mae, valid_src))
valid_ans.append([valid_mse, valid_mae, valid_src])

submit_pred = cb_model.predict(submit_data)
submit_proba.append(submit_pred)
valid_ans = np.mean(valid_ans, axis=0)
print("valid: MSE: %.4f, MAE: %.4f, SRC: %.4f"%(valid_ans[0], valid_ans[1], valid_ans[2]))

# save result json
submit_ans = np.mean(submit_proba, axis=0)
result = pd.DataFrame()
result['post_id'] = submit_label_df['Pid'].apply(lambda x: 'post' + str(x))
result['popularity_score'] = submit_ans.round(decimals=4)

out_json = dict()
out_json["version"] = "VERSION 1.0"
out_json["result"] = result.to_dict(orient='records')
out_json["external_data"] = {"used": "true", "details": "use_catboost"}
f = open('KFold_catboost.json', "w")
json.dump(out_json, f)
f.close()
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# k = 0
#
# for train_idx, valid_idx in kfold.split(train_feature_df, train_label_df):
#     fold_train_x, fold_train_y = train_feature_df.loc[train_idx], train_label_df['label'].loc[train_idx]
#     fold_valid_x, fold_valid_y = train_feature_df.loc[valid_idx], train_label_df['label'].loc[valid_idx]
#
#     train_data = Pool(data=fold_train_x, label=fold_train_y, cat_features=cate_cols)
#     valid_data = Pool(data=fold_valid_x, label=fold_valid_y, cat_features=cate_cols)
#
#     cb_model = CatBoostRegressor(**cb_params)
#     cb_model.fit(train_data, eval_set=valid_data)
#
#     valid_pred = cb_model.predict(valid_data)
#     valid_mse = mean_squared_error(fold_valid_y, valid_pred)
#     valid_mae = mean_absolute_error(fold_valid_y, valid_pred)
#     valid_src = stats.spearmanr(fold_valid_y, valid_pred)[0]
#
#     print("MSE: %.4f, MAE: %.4f, SRC: %.4f" % (valid_mse, valid_mae, valid_src))
#     valid_ans.append([valid_mse, valid_mae, valid_src])
#
#     submit_pred = cb_model.predict(submit_data)
#     submit_proba.append(submit_pred)
#
#     cb_model.save_model('../data/save_model/KFold_catboost_' + str(k) + '.pkl')
#     k += 1

valid_ans = np.mean(valid_ans, axis=0)
print("valid: MSE: %.4f, MAE: %.4f, SRC: %.4f" % (valid_ans[0], valid_ans[1], valid_ans[2]))

# save result json
submit_ans = np.mean(submit_proba, axis=0)
result = pd.DataFrame()
result['post_id'] = submit_label_df['Pid'].apply(lambda x: 'post' + str(x))
result['popularity_score'] = submit_ans.round(decimals=4)

out_json = dict()
out_json["version"] = "VERSION 5.1"
out_json["result"] = result.to_dict(orient='records')
out_json["external_data"] = {"used": "true", "details": "use catboost to do the regression"}
f = open('prediction0501.json', "w")
json.dump(out_json, f)
f.close()
