import re
from snownlp import sentiment
import numpy as np

from snownlp import SnowNLP
import matplotlib.pyplot as plt
from snownlp import sentiment
from snownlp.sentiment import Sentiment


import codecs
import re
import numpy as np

from snownlp import SnowNLP
import matplotlib.pyplot as plt
from snownlp import sentiment
from snownlp.sentiment import Sentiment

comment = []
with open('微博评论/李小璐', mode='r', encoding='utf-8') as f:
    rows = f.readlines()
    for row in rows:
        if row not in comment:
            comment.append(row.strip('\n'))

def snowanalysis(self):
    sentimentslist = []
    for li in self:
        li = re.sub(r'(?:回复)?(?://)?@[\w\u2E80-\u9FFF]+:?|\[\w+\]', ',',li)
        #print(li)

        try:
            s = SnowNLP(li)
            print(s.sentiments)# 可能会出现异常的一段代码
        except:     # try中任意一行语句出现异常，直接跳转至except，程序继续运行
            continue


        sentimentslist.append(s.sentiments)
    plt.hist(sentimentslist, bins=np.arange(0, 1, 0.01))
    plt.show()
snowanalysis(comment)

print('分析完成')


import pickle
from os import path
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def make_worldcloud(file_path):
    text_from_file_with_apath = open(file_path,'r',encoding='UTF-8').read()
    wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all=False)
    wl_space_split = " ".join(wordlist_after_jieba)
    print(wl_space_split)
    backgroud_Image = plt.imread('心1.jpg')
    print('加载图片成功！')
    ''设置词云样式''
    stopwords = STOPWORDS.copy()
    stopwords.add("哈哈")
    stopwords.add('回复')
    stopwords.add('李小璐')#加多个屏蔽词

    wc = WordCloud(
        width=1024,
        height=768,
        background_color='white',# 设置背景颜色
        mask=backgroud_Image,# 设置背景图片
        font_path='simsun.ttf',  # 设置中文字体，若是有中文的话，这句代码必须添加，不然会出现方框，不出现汉字
        max_words=600, # 设置最大现实的字数
        stopwords=stopwords,# 设置停用词
        max_font_size=400,# 设置字体最大值
        random_state=50,# 设置有多少种随机生成状态，即有多少种配色方案
    )
    wc.generate_from_text(wl_space_split)#开始加载文本
    image_colors = ImageColorGenerator(backgroud_Image)

    wc.recolor(color_func= image_colors)#字体颜色为背景图片的颜色
    plt.imshow(wc)# 显示词云图
    plt.axis('off')# 是否显示x轴、y轴下标
    plt.show()#显示
    # 获得模块所在的路径的
    d = path.dirname(__file__)
    # os.path.join()：  将多个路径组合后返回
    wc.to_file(path.join(d, "心1.jpg"))
    print('生成词云成功')

make_worldcloud('微博评论/李小璐')