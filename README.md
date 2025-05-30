# SMP-2025
The solution to Social Media Prediction Challenge 

## This section contains a total of 8 folders, described as follows:

**data analysis**: Used for analyzing the official dataset.

**Files**: Contains additional reference datasets.

**feature extraction**: Used to extract textual and visual representations using pre-trained models and store them.

**DAE+1DCNN**, **HyFea**, **RoBERTa**, and **Tablur**: These are the four different models we used.

**post_blending**: Contains code for combining and processing the prediction results from the four models.

**Note**: The **HyFea** folder includes a script named **download_img_and_user.py**, which can automatically crawl additional meta-information related to the data.

During execution, the code generates over 100GB of model files. Due to this large size, we are currently unable to make all of them publicly available. We will selectively release some of the models. Also, due to time constraints and code dependencies, the current implementation is not organized as one-click runnable scripts. We plan to refactor and improve the code structure in future updates to enhance usability.


## Models 
In general, we have adopted several methods. 

The first one is [HyFea](https://github.com/runnerxin/HyFea). The codes were applied to download the images, generate the glove embeddings of the titles and tags, crawl the user additional information such as 'Pathalias', 'totalViews', 'totalTags', 'totalGeotagged', 'totalFaves','totalInGroup','photoCount', 'followerCount' and 'followingCount'. The crawled information were proved to be very useful in pratice. HyFea used the catboost as the backbone and we trained several catboost models, including using different feature combinations, different normed_labels.  

The second one is [AutoGluon](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/beginner_multimodal.html) and [Tabular](https://www.kaggle.com/code/optimo/tabnetbaseline), which were often used in time series models. We used this model to catch some times-series related information. For Tabular, we pretrained the models with 50% of the training data, the rest were left for fintuning.

The third one is [DAE](https://www.kaggle.com/code/isaienkov/keras-autoencoder-dae-neural-network-starter/notebook)+[1DCNN](https://www.kaggle.com/code/yhx003/baseline-denoiseautoencoder-1dcnn/notebook), which was often used in time series models. We used this model to catch some times-series related information.

The fourth is the roberta model only using the text data. The codes are modified from [pre-training codes](https://www.kaggle.com/code/rhtsingh/commonlit-readability-prize-roberta-torch-itpt/notebook?scriptVersionId=63560998), [finetuning codes](https://www.kaggle.com/code/rhtsingh/commonlit-readability-prize-roberta-torch-fit/notebook), [inferencing codes](https://www.kaggle.com/code/rhtsingh/commonlit-readability-prize-roberta-torch-infer/notebook)Besides, we also used the [codes](https://www.kaggle.com/code/andretugan/lightweight-roberta-solution-in-pytorch/notebook) for regression.


## Features Extraction
Besides, we have adapted many pretrained models to compore the performance. The models includes but not limited to [CLIP_TXT](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K), 
 [CLIP_IMG](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K),  [BERT](https://huggingface.co/bert-base-uncased), [BERTWEET](https://huggingface.co/vinai/bertweet-base), [USTC features1](https://github.com/Corleone-Huang/Social-Media-Popularity-Prediction-Challenge-2020), and [USTC features2](https://pan.baidu.com/share/init?surl=wRMKmb3OIol_Yd_ltYyAwg (pwd:539j).  [vlit_vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa), [vilt_flickr](https://huggingface.co/dandelin/vilt-b32-finetuned-flickr30k),  [BLIP2](https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_feature_extraction.ipynb), [GPT2-NEO-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B), [GPT2-NEO-2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B), [XLM-ROBERTA](https://huggingface.co/xlm-roberta-large), [KNN](https://www.kaggle.com/code/remekkinas/keras-tuner-knn-features-simplex-optimization/notebook),  and [HyFea](https://github.com/runnerxin/HyFea).




