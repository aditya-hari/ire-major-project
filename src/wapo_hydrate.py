from newspaper import Article
from bs4 import BeautifulSoup
import pandas as pd 
import glob
from selenium import webdriver

authors = glob.glob("../datasets/wapo/*")
authors.sort()
df_dict = {"author":[], "link":[], "article":[]}

# browser = webdriver.Firefox(executable_path="/home/green/Downloads/geckodriver-v0.30.0-linux64/geckodriver")
# browser.implicitly_wait(15)

# browser.get('https://www.washingtonpost.com/subscribe/signin/')
# browser.find_element_by_id("username").send_keys('arthurbrox69@gmail.com')
# browser.find_element_by_xpath('/html/body/div[2]/div/div/div[1]/div[2]/form/button').click()
# browser.find_element_by_id("password").send_keys('temp_pass_123')
# browser.find_element_by_xpath('/html/body/div[2]/div/div/div[1]/div[2]/form/button').click()





for author_file in authors:
    author_name = " ".join(author_file.split('/')[-1][:-4].split('-'))
    print(author_name)
    links = open(author_file, 'r').readlines()
    for link in links:
        full = ""
        article = Article(link)
        article.download()
        soup = BeautifulSoup(article.html, features="lxml")
        paras = soup.find_all("p", class_="font--article-body font-copy gray-darkest ma-0 pb-md")
        for para in paras:
            if(para.text!= "Read more:"):
                full+=para.text+'\n'
            else:
                break

        df_dict["author"].append(author_name)
        df_dict["link"].append(link)
        df_dict["article"].append(full)
        
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv('wapo.csv')

df = pd.DataFrame.from_dict(df_dict)
df.to_csv('wapo.csv')


    

    
