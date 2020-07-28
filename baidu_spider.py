import requests
import urllib
import re
from requests_html import HTMLSession
from bs4 import BeautifulSoup
def get_response(url):
    headers =dict()
    # url='https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word=女人为什么比男人活得久'
    res = requests.get(url)
    cookie_dic = requests.utils.dict_from_cookiejar(res.cookies)
    headers['Cookie'] =';'.join([key+'='+value for key,value in cookie_dic.items()]) 
    response = requests.get(url, headers= headers)
    response.encoding = 'gbk'
    return response

def zhidao_search(key_word):
    key_word = urllib.parse.quote(key_word,encoding = 'GBK', errors = 'replace')
    url = 'https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word=' + key_word
    back_list= get_response(url)
    bs_bl_page = BeautifulSoup(back_list.text)
    try:
        most_simques_ele =  bs_bl_page.select('#wgt-list > dl:nth-child(1) > dt > a')[0]
    except IndexError:
        # print('对不起，我不太理解')
        return '对不起，我不太理解'
    most_simques_link = most_simques_ele.attrs['href']
    answer_page = get_response(most_simques_link)
    bs_an_page = BeautifulSoup(answer_page.text)
    pattern = re.compile('bd answer')
    answer_list = bs_an_page.find_all('div',class_=pattern)
    answer=''
    for index,answer_ele in enumerate(answer_list):
        if answer_ele.find('vvideo'):
            continue
        else:
            answer = answer_ele.find(id=re.compile('best-content-'))
            
            if index==0 and answer:
                answer = answer.text
            else:
                answer = answer_ele.find(id=re.compile('answer-content-')).text
            return re.split(r'参考资料|已赞过|展开全部|本回答被网友采纳',answer)[1].strip()
            

if __name__ == "__main__":
    zhidao_search('毛泽东生日')