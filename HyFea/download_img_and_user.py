import pandas as pd
from bs4 import BeautifulSoup
import requests
import urllib
import urllib.request
import os
import time
import re
import json
from concurrent.futures import ProcessPoolExecutor

path = './data'
requests.adapters.DEFAULT_RETRIES = 5
s = requests.session()
s.keep_alive = False


# get train data img url
def crawl(url):
    page = ''
    while page == '':
        try:
            page = s.get(url)
        except:
            time.sleep(5)
            continue
    soup = BeautifulSoup(page.content, 'html.parser')
    imgs_list = soup.find_all('img')
    if len(imgs_list) != 0:
        img_url = 'https:' + imgs_list[-1]['src']
    else:
        img_url = 'error_img'
    # return url + '\t' + img_url
    return img_url

# download image
# def get_image(img, url):
#     usr = img.split('/')[-2]
#     # file_suffix = img.split('/')[-1] + '.jpg'
#     file_suffix = img.split('/')[-1]
#     file_path = os.path.join(path, 'train', usr)
#     try:
#         if not os.path.exists(file_path):
#             os.makedirs(file_path)
#         if url != 'error_img':
#             filename = '{}{}{}'.format(file_path, os.sep, file_suffix)
#             urllib.request.urlretrieve(url, filename=filename)
#         return 'Success'
#     except:
#         return img + '\t' + url
#
#
# def job(z):
#     return get_image(z[0], z[1])



# get_user_add_info
def get_d(pa):
    page = ''
    while page == '':
        try:
            page = s.get('https://www.flickr.com/people/' + pa + '/')
        except:
            time.sleep(5)
            continue
    soup = BeautifulSoup(page.content, 'html.parser')
    tag = soup.find(attrs={"class": "modelExport"})
    totalViews = re.findall('"totalViews":[0-9]+', tag.string)
    totalTags = re.findall('"totalTags":[0-9]+', tag.string)
    totalGeotagged = re.findall('"totalGeotagged":[0-9]+', tag.string)
    totalFaves = re.findall('"totalFaves":[0-9]+', tag.string)
    totalInGroup = re.findall('"totalInGroup":[0-9]+', tag.string)
    photoCount = re.findall('"photoCount":[0-9]+', tag.string)
    followerCount = re.findall('"followerCount":[0-9]+', tag.string)
    followingCount = re.findall('"followingCount":[0-9]+', tag.string)
    return pa + "\t" + get_count(totalViews) + "\t" + get_count(totalTags) + "\t" + get_count(
        totalGeotagged) + "\t" + get_count(totalFaves) + "\t" + get_count(totalInGroup) + "\t" + get_count(
        photoCount) + "\t" + get_count(followerCount) + "\t" + get_count(followingCount)


def get_count(x):
    if len(x) == 0:
        return '0'
    else:
        return x[0].split(":")[1]



def main():
    ## imgs是爬取的url
    # imgs = pd.read_csv(os.path.join(path, 'train_img_filepath.txt'), header=None, names=['img'])
    # # img count 305613
    # # for j in range(0, 3050):
    # for j in range(0, 3050):
    #     print('save:', j * 100, (j + 1) * 100)
    #     f = open(os.path.join(path, 'url_stats.txt'), 'a+')
    #     with ProcessPoolExecutor(8) as pool:
    #         if j == 3049:
    #             p = pool.map(crawl, imgs.img[j * 100:])  # crawl对应url
    #         else:
    #             p = pool.map(crawl, imgs.img[j * 100:(j + 1) * 100])
    #         for i in p:
    #             f.write(i + '\n')
    #             # f 之后是成功与否的记录?
    #         f.close()
    #
    # # img_url = pd.read_csv(os.path.join(path, 'train_img_filepath.txt'), delimiter='\t', header=None,
    # #                       names=['img', 'url'])
    # #############################
    # # img_url = pd.read_csv(os.path.join(path, 'train_img_filepath.txt'), header=None,
    # #                       names=['url'])
    # imgs_new = pd.read_csv(os.path.join(path, 'train_img.txt'), header=None,
    #                       names=['img'])
    # img_url = pd.read_csv(os.path.join(path, 'url_stats.txt'), header=None,
    #                       names=['url'])
    # z = list(zip(imgs_new['img'], img_url['url']))
    # for j in range(0, 306):
    #     # 'wrong_img_url.txt' save img status
    #     f = open(os.path.join(path, 'status_img_url.txt'), 'a+')
    #     print('save:', j * 1000, (j + 1) * 1000)
    #     with ProcessPoolExecutor() as pool:
    #         if j == 305:
    #             p = pool.map(job, z[j * 1000:])  ## 将图像存储进来
    #         else:
    #             p = pool.map(job, z[j * 1000:(j + 1) * 1000])
    #         for i in p:
    #             f.write(i + '\n')
    #         f.close()


    train_add = pd.read_json(os.path.join(path, 'train_additional_information.json'))
    test_add = pd.read_json(os.path.join(path, 'test_additional_information.json'))
    all_data = pd.concat([train_add, test_add], axis=0, sort=False)
    # pa = all_data.Pathalias.unique()
    pa = all_data.Uid.unique()
    print('the length is %d', len(pa))
    for j in range(0, 365):
        print('save:', j * 1000, (j + 1) * 1000)
        f = open(os.path.join(path, 'user_additional.txt'), 'a+')
        with ProcessPoolExecutor(8) as pool:
            if j == 364:
                p = pool.map(get_d, pa[j * 1000:])
            else:
                p = pool.map(get_d, pa[j * 1000:(j + 1) * 1000])
            for i in p:
                f.write(i + '\n')
            f.close()

    user_info = pd.read_csv(os.path.join(path, 'user_additional_aa.txt'), delimiter='\t', header=None,
                            names=['Pathalias', 'totalViews', 'totalTags', 'totalGeotagged', 'totalFaves',
                                   'totalInGroup',
                                   'photoCount', 'followerCount', 'followingCount'])
    user_info.to_csv(os.path.join(path, 'user_additional_aa.csv'))



if __name__ == "__main__":
    main()