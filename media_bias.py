#!/usr/bin/env python
# coding: utf-8

# # 動態網頁資料爬不全
# 
# Filters setting:

# ![image.png](attachment:image.png)

# In[51]:


import requests
from bs4 import BeautifulSoup

url="https://www.allsides.com/media-bias/media-bias-ratings?field_featured_bias_rating_value=All&field_news_source_type_tid[1]=1&field_news_source_type_tid[2]=2&field_news_source_type_tid[3]=3&field_news_source_type_tid[4]=4"
r = requests.get(url)

soup = BeautifulSoup(r.text, "html.parser")


# In[34]:


# 全部的投票分數 不知道為何出現 agree disagree
for d in soup.find_all(class_="views-field views-field-nothing community-feedback"):
     print(d.text)


# 我嘗試的結果：只能分開爬奇數和偶數的資料ＱＱ

# In[53]:


#even media 偶數資料
for even in soup.find_all(class_="even"):
    a =even.find_all('a')
    print(a)


# In[55]:


media=[]
for even in soup.find_all(class_="even"):
    a =even.find_all('a')
    media.append(a[0].text)


# In[57]:


media  #動態網頁資料不全 用selenium


# In[62]:


#偶數資料中
for even in soup.find_all(class_="even"):
    a =even.find_all('a')
    print(a[1])

'''
<a href="/media-bias/left-center">
<a href="/media-bias/allsides">
'''


# In[65]:


type(a[1]) # 如何 bs4.element.tag to string? 再去做文字處理


# In[52]:


# odd media 奇數資料
for odd in soup.find_all(class_="odd"):
    a =odd.find_all('a')
    print(a)


# # 練習用範例

# In[41]:


import numpy as np
import requests as rq
from bs4 import BeautifulSoup

url = 'https://www.ptt.cc/bbs/NBA/index.html'
response = rq.get(url)
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "html.parser") # 指定 lxml 作為解析器

author_ids = [] # 建立一個空的 list 來放作者 id
recommends = [] # 建立一個空的 list 來放推文數
post_titles = [] # 建立一個空的 list 來放文章標題
post_dates = [] # 建立一個空的 list 來放發文日期

posts = soup.find_all("div", class_ = "r-ent")
for post in posts:
    try:
        author_ids.append(post.find("div", class_ = "author").string)    
    except:
        author_ids.append(np.nan)
    try:
        post_titles.append(post.find("a").string)
    except:
        post_titles.append(np.nan)
    try:
        post_dates.append(post.find("div", class_ = "date").string)
    except:
        post_dates.append(np.nan)

# 推文數藏在 div 裡面的 span 所以分開處理
recommendations = soup.find_all("div", class_ = "nrec")
for recommendation in recommendations:
    try:
        recommends.append(int(recommendation.find("span").string))
    except:
        recommends.append(np.nan)

print(type(author_ids))
print(recommends)
print(post_titles)
print(post_dates)


# In[ ]:




