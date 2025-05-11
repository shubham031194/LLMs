# OpenAI_News_Agent

### Introduction
This project is a news aggregation tool designed for developers, leveraging Python, OpenAI API services, Google Search News XML fetch, and Streamlit. It simplifies the process of gathering, filtering, and presenting news articles, making it easier for developers to stay informed.

![NewAgent.](https://github.com/shubham031194/OpenAI_News_Agent/blob/main/resources/newsagent.png)

### Services Used:
1. **OpenAI API Service:** We leverage OpenAI's powerful API to generate professional-looking articles based on the collected news data. This service enhances the readability and coherence of the news content, providing users with a seamless reading experience.
2. **Google Colab:** We use Google Colab, a cloud-based platform, for collaborative coding and data analysis. It allows us to develop and test our code in a cloud environment, providing flexibility and scalability.
3. **Ngrok:** Ngrok is utilized for local tunneling, allowing us to expose our Streamlit app running on localhost to the internet. This enables us to share our app with others and test it in a real-world scenario.
4. **Streamlit:** Streamlit is the backbone of our news agent, providing a user-friendly interface for users to interact with. It allows us to create dynamic and responsive web applications with minimal effort, making it ideal for our news aggregation tool.

### Install Packages
```
!pip install streamlit
!pip install pyngrok
!pip install streamlit-option-menu
!pip install feedparser
!pip install requests
!pip install beautifulsoup4
!pip install openai
```

### Settingup localtunneling
To make you streamlit app public while running on google colab notebook you need to have authorix=zation token from ngrok. 
To get the token follow below steps:
1. Create account on https://dashboard.ngrok.com/
2. Go to Tunnel --> Auththokes ![](https://github.com/shubham031194/OpenAI_News_Agent/blob/main/resources/Screenshot%202024-05-02%20124406.png)
3. Genrate your token and use it.

### Project Execution flow
The tool begins by prompting users to enter a keyword of interest. Using Google Search News XML fetch, it searches the web for articles related to the keyword. The tool then extracts key information such as the source name, publishing date, and website link for each article.

One of the key functionalities of our tool is its ability to filter news sources based on user preferences. This allows developers to tailor their news feed to include only the sources they trust and find valuable.

After selecting their preferred sources, the tool fetches the full article content from the respective websites. This ensures that developers have access to complete information from sources they trust, without needing to visit multiple websites.

To enhance the readability and professionalism of the articles, our tool utilizes OpenAI API services to generate a coherent and well-structured article based on the collected data. This feature provides developers with a concise summary of the news, saving them time and effort in digesting the information.

In conclusion, our news aggregation tool offers developers a streamlined way to stay up-to-date with the latest news. By combining advanced technologies, it provides a customizable and efficient news reading experience, empowering developers to focus on their work without missing important updates from the tech world.
