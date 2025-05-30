# SMP-2025
The solution to Social Media Prediction Challenge 
### Models 
In general, we have adopted several methods. 

The first one is HyFea, of which the codes could be found in https://github.com/runnerxin/HyFea. The codes were applied to download the images, generate the glove embeddings of the titles and tags, crawl the user additional information such as 'Pathalias', 'totalViews', 'totalTags', 'totalGeotagged', 'totalFaves','totalInGroup','photoCount', 'followerCount' and 'followingCount'. The crawled information were proved to be very useful in pratice. HyFea used the catboost as the backbone and we trained several catboost models, including using different feature combinations, different normed_labels.  

The second one is DAE+1DCNN, which was often used in time series models. We used this model to catch some times-series related information. The codes could be found in https://www.kaggle.com/code/isaienkov/keras-autoencoder-dae-neural-network-starter/notebook and https://www.kaggle.com/code/yhx003/baseline-denoiseautoencoder-1dcnn/notebook.

The third one is AutoGluon, which could be found in https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/beginner_multimodal.html. We applied several models using different combinations of modalities. Such as we first ignored the categorical columns, then assigned them as categorical features.

The fourth one is Tabular, which could be found in https://www.kaggle.com/code/optimo/tabnetbaseline. We pretrained the models with 50% of the training data, the rest were left for fintuning.

The fifth is the roberta model only using the text data. The pretraining codes was in https://www.kaggle.com/code/rhtsingh/commonlit-readability-prize-roberta-torch-itpt/notebook?scriptVersionId=63560998, while the finetuning codes in https://www.kaggle.com/code/rhtsingh/commonlit-readability-prize-roberta-torch-fit/notebook, and the inferencing codes in https://www.kaggle.com/code/rhtsingh/commonlit-readability-prize-roberta-torch-infer/notebook. Besides, we also used the https://www.kaggle.com/code/andretugan/lightweight-roberta-solution-in-pytorch/notebook for regression.

The sixth one is multimodal toolkit, which could be found in https://github.com/georgian-io/Multimodal-Toolkit.

The seventh one is lightGBM.

The eighth is DIR model, which could be found in https://github.com/YyzHarry/imbalanced-regression.

### Features Extraction
Besides, we have adapted many pretrained models to compore the performance. The models includes but not limited to CLIP_TXT(https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K), 
 CLIP_IMG(https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K),  BERT(https://huggingface.co/bert-base-uncased),  BERTWEET(https://huggingface.co/vinai/bertweet-base),  USTC features(https://github.com/Corleone-Huang/Social-Media-Popularity-Prediction-Challenge-2020,  https://pan.baidu.com/share/init?surl=wRMKmb3OIol_Yd_ltYyAwg (pwd:539j)),  vlit_vqa(https://huggingface.co/dandelin/vilt-b32-finetuned-vqa), 
 vilt_flickr(https://huggingface.co/dandelin/vilt-b32-finetuned-flickr30k),  BLIP2(https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_feature_extraction.ipynb),  GPT2-NEO-1.3B(https://huggingface.co/EleutherAI/gpt-neo-1.3B),  GPT2-NEO-2.7B(https://huggingface.co/EleutherAI/gpt-neo-2.7B),  XLM-ROBERTA(https://huggingface.co/xlm-roberta-large), 
 KNN(https://www.kaggle.com/code/remekkinas/keras-tuner-knn-features-simplex-optimization/notebook),  HyFea(https://github.com/runnerxin/HyFea).

**The detailed performance and data analysis, and some checkpoints will be updated soon.**
**Note that the post-processing of the label (or dark magic) is also very useful, which will also be updated soon.**

### Code Implementation
The quickest way to reproduce our model is to run the **./HeyFea-main/test_k_fold_model.py** using the checkpoints in **./HeyFea-main/save_model/**. We used the pesudo-labels technique to get these checkpoints.
If you want to reproduce the our model from scratch, please follow the steps blow.

Firstly, re-pretrain the roberta models using the script: Waiting to be updated.

Next, extract the features using pretrained models: Waiting to be updated

Finally, use all the features to do the prediction: Waiting to be updates.
