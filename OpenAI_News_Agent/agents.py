from openai import OpenAI
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import configparser

class WebAgent:
    def __init__(self, keyword):
        self.keyword = keyword
    
    def fetch_keyword_news(self):
        url = f"https://news.google.com/rss/search?q={self.keyword}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.find_all('item')
        else:
            return response.status_code
   
    @staticmethod
    def fetch_news_content(link):
        response = requests.get(link)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.text
        return None

    def analyse_web_content(self, agent, content):
        completion = agent.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                    {"role": "system", "content":f'''You are a news asistent, your job is to analyse the webpage for news related to {self.keyword}
                                                                    and genrate the news content in the given webpage. Remove tags and scripts if present in
                                                                    your final result.'''},
                                    {"role": "user", "content": content}
                                    ]
                            )
        news_content = completion.choices[0].message.content
        return news_content

class NewsAgent:
    def __init__(self, keyword):
        self.config_path = 'config.ini'
        self.keyword = keyword
        self.dataList = []
        self.gatheredData = ''
        self.llm = None
        self.webagent = None
    
    @staticmethod
    def getapikey(config):
        return config.get('Settings', 'api_key')
    
    def fetchNewsAlerts(self):
        src_list = []
        self.webagent = WebAgent(self.keyword)
        googleXML = self.webagent.fetch_keyword_news()
        if googleXML:
            for xml in googleXML:
                date = xml.pubdate.text
                pub_date = datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %Z')
                filt_date = pub_date - timedelta(days=1)
                title = xml.title.text.split('-')[0].strip()
                src = xml.title.text.split('-')[1].strip()
                link = r'https://news.google.com/rss/articles/' + xml.guid.text.strip()
                fetched_data = {
                                'source': src,
                                'title': title,
                                'link': link,
                                'publist_date': pub_date
                            }
                src_list.append(fetched_data['source'])
                self.dataList.append(fetched_data)
        return src_list

    def gatherNewsData(self, filtered_src):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        self.llm = OpenAI(api_key=NewsAgent.getapikey(config))
        for item in self.dataList:
            filter_date = item['publist_date'] - timedelta(days=1)
            if item['source'] in filtered_src and item['publist_date'] > filter_date:
                content = self.webagent.fetch_news_content(item['link'])
                if content:
                    web_content = self.webagent.analyse_web_content(self.llm, content)
                    self.gatheredData += web_content

    def final_article_genration(self):
        prompt = f'''Consider yourself as an expert in writing news blog for Indian audiences. As a master with SEO-friendly
                    copywriting skills you should do proper keyword research and use the keywords, in your article, that have
                    high search volume and less competition. Now, I will give you collected news articles on {self.keyword} and I
                    want you to write a comprehensive news article on this topic which includes price, performance, specification,
                    interior exterior looks, launch offers, etc. Remember to use simple english and power keywords to deliver Accurate,
                    and Legitimate news structured in a format to give best reader experience and the article length should be in a range
                    of 500 to 600 words.'''
        completion = self.llm.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": self.gatheredData}
                                    ]
                            )
        news_content = completion.choices[0].message.content
        return news_content