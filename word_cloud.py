

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

    stopwords = STOPWORDS.copy()
    stopwords.add("哈哈")
    stopwords.add('回复')
    stopwords.add('李小璐')

    wc = WordCloud(
        width=1024,
        height=768,
        background_color='white',
        mask=backgroud_Image,
        font_path='simsun.ttf',
        max_words=600,
        stopwords=stopwords,
        max_font_size=400,
        random_state=50,
    )
    wc.generate_from_text(wl_space_split)
    image_colors = ImageColorGenerator(backgroud_Image)

    wc.recolor(color_func= image_colors)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    d = path.dirname(__file__)
    wc.to_file(path.join(d, "心1.jpg"))
    print('生成词云成功')

make_worldcloud('微博评论/李小璐')