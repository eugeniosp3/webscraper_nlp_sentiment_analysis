import requests
from bs4 import BeautifulSoup
import time
from random import randint

# main page
URL = 'https://www.trustradius.com/reviews'
page = requests.get(URL)
time.sleep(randint(1, 5))
soup = BeautifulSoup(page.text, 'html.parser')

# find user review links
final_link =[x.find('a')['href'] for x in soup.findAll('li', {'class': 'last'})]
current_link = 0
user_review_links = [x.find('a')['href']for x in soup.findAll('h3', {'class': 'h3 review-title'})]
for urlink in user_review_links:
    current_link = 'https://www.trustradius.com/reviews{linkn}'.format(linkn=urlink)

reviews_scraped = 0
# while current_link != final_link and reviews_scraped !=25:







# find the next page link
page_links = soup.find('ul', {'class': 'pagination'})
link_next = str([li.find('a')['href'] for li in page_links if
                 soup.find('li', {'class': 'active'}) != True and li.find('a').get_text() == 'Next']).replace('[\'', '').replace('\']', '')
URL2 = 'https://www.trustradius.com/reviews{linkn}'.format(linkn=link_next)





def soupme(url_1, url_2, r_soup):
    # first page
    page = requests.get(url_1)
    time.sleep(randint(1, 5))
    soup = BeautifulSoup(page.text, 'html.parser')
    # next page
    page2 = requests.get(url_2)
    time.sleep(randint(1, 5))
    soup2 = BeautifulSoup(page2.text, 'html.parser')
    # user review links
    for urlink in user_review_links:
        print('https://www.trustradius.com/reviews{linkn}'.format(linkn=urlink))
    page3 = requests.get(url_2)
    time.sleep(randint(1, 5))
    rsoup = BeautifulSoup(page3.text, 'html.parser')



