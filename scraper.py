import requests
from bs4 import BeautifulSoup
import time
from random import randint
import mysql.connector
import re

# -- for the database setup

HOST = "database ip address"
USERNAME = "my username"
PASSWORD = "the password used"
DATABASE = "the database used"
cnx = mysql.connector.connect(user=USERNAME, password=PASSWORD,
                              host=HOST,
                              database=DATABASE,
                              auth_plugin='mysql_native_password')  # auth_plugin might give you errors depending on how the password is setup around the database {google the error}
cursor = cnx.cursor()


# -- end database setup


# -- start database pass function
reviews_scraped = 0
links_passed = 0


def pass_to_database(current_link):
    global reviews_scraped
    session = requests.Session()
    # set username and password
    session.auth = ('user', 'pass')
    # update headers
    session.headers.update({'x-test': 'true'})
    # send the page request
    page = session.get(str(current_link), headers ={'x-test2': 'true'})
    # pause the loop - for human-like appearance
    time.sleep(randint(1, 10))
    soup4 = BeautifulSoup(page.text, 'html.parser')
    header = soup4.find_all('div', class_='serp-header')
    for rph in header:


        date = str(rph.find('div', class_='review-date'))
        review_title = str([x.get_text() for x in rph.select('#review-body > div > div > div.serp-row.serpHitundefined > article > div.serp-header > h1')])
        user_role_title = str(rph.find('div', class_='position'))
        company_industry = str(rph.find('span', class_='industry-type'))
        company_size = str(rph.find('span', class_='size'))
        score = str(rph.select('#review-body > div > div > div.serp-row.serpHitundefined > article > div.serp-header > div.trust-score > div:nth-child(1) > div.trust-score__score span')).replace('[<span class=\"strong\">Score </span>, <span>', '').replace(' out of 10</span>]', '')
        sql = """INSERT INTO header_columns( review_date, review_title, user_role_title, company_industry,
     company_size, score) VALUES (%s, %s, %s, %s, %s, %s)"""
        valso = [date, review_title, user_role_title, company_industry, company_size, score]
        cursor.execute(sql, valso)
        cnx.commit()
        reviews_scraped += .5

def pass_to_database_body(current_link):
    global reviews_scraped
    session = requests.Session()
    # set username and password
    session.auth = ('user', 'pass')
    # update headers
    session.headers.update({'x-test': 'true'})
    # send the page request
    page = session.get(str(current_link), headers ={'x-test2': 'true'})
    # pause the loop - for human-like appearance
    time.sleep(randint(1, 10))
    soup4 = BeautifulSoup(page.text, 'html.parser')
    body = soup4.findAll('section', {"class": "review-layout__review"})

    for bt in body:
        app_name = str([x.get_text() for x in bt.select('#review-body > section > div > div.section-block__header')][0])
        use_case_deployment_scope = str(bt.find('div', class_='response').text)
        pros = str(bt.find('ul', class_='pros'))
        cons = str(bt.find('ul', class_='cons'))
        roi = str([x.get_text(separator=' ') for x in bt.findAll('ul', {'data-slug': 'operational-benefits'})])
        competitors_considered = str(
            [x.get_text('----') for x in bt.findAll('div', {'class': 'not-edited question-competitors'})]).replace("[\"", '').replace("\"]", '')
        support_rating_usability_recommendation = str(
            [x.get_text('----') for x in bt.findAll('div', {'class': 'not-edited question-slider'})]).replace("[\"", '').replace("\"]", '')
        other_questions = str([x.get_text('----') for x in bt.findAll('div', {'class': 'js-multiquestion'})]).replace("[\"", '').replace("\"]", '')
        others_used = str([x.get_text('----') for x in bt.findAll('div', {'class': 'not-edited question-ratings'})]).replace("[\'", '').replace("\']", '')

        sql = """INSERT INTO body_columns(use_case_deployment_scope, pros, cons, roi, competitors_considered,
          support_rating_usability_recommendation, other_questions, others_used, app_name) VALUES (%s,
          %s, %s, %s, %s, %s, %s, %s, %s)"""

        valso = [use_case_deployment_scope, pros, cons, roi, competitors_considered,
                 support_rating_usability_recommendation, other_questions, others_used, app_name]

        cursor.execute(sql, valso)
        cnx.commit()
        reviews_scraped += .5


requestn = 0
start_time = time.time()

# -- scraper components

cat_links_list = ['human-resources']
    # 'customer-support', 'development', 'enterprise', 'finance-and-accounting', 'human-resources',
    #               'information-technology', 'marketing', 'professional-services', 'sales', 'vertical-specific']



def get_all_review_pages(url1):
    """ this function will go in and find the number of pages that are inside of each product page
    this allows the scraper to not miss any reviews """
    session = requests.Session()
    # set username and password
    session.auth = ('user', 'pass')
    # update headers
    session.headers.update({'x-test': 'true'})
    reviews_number = r'(?<=\s)\d*(?=\))'
    page111 = session.get(url1, headers ={'x-test2': 'true'})
    soup11 = BeautifulSoup(page111.text, 'html.parser')
    user_review_link = soup11.findAll('li', {'class': 'last'})
    front = url1.split('#products')[0]
    back = str()
    middle_range = int()
    add_this = int()
    for x in user_review_link:
        y = int(re.findall(r'(?<=\")\d*(?=\")', str(x))[0])
        for_range = round(y / 25)
        middle_range = (for_range*25)
        add_this = y - middle_range
    list_of_review_pages = []
    for i in range(0,(middle_range+add_this)+25, 25):
        urlback = '?f={numba}#products'.format(numba=i)
        back = str(urlback)
        final_link_concat = front + back
        print('Page Number: ', final_link_concat, '\n')
        list_of_review_pages.append(final_link_concat)
    return list_of_review_pages



def get_all_pages(url1):
    """ this function will go in and find the number of pages that are inside of each product page
    this allows the scraper to not miss any reviews """
    session = requests.Session()
    # set username and password
    session.auth = ('user', 'pass')
    # update headers
    session.headers.update({'x-test': 'true'})
    reviews_number = r'(?<=\s)\d*(?=\))'
    page111 = session.get(url1, headers ={'x-test2': 'true'})
    soup11 = BeautifulSoup(page111.text, 'html.parser')
    user_review_link = soup11.findAll('div', {'class': 'CategoryProducts_category-product-heading__1tKlO'})
    front = url1.split('#products')[0]
    back = str()
    middle_range = int()
    add_this = int()
    for x in user_review_link:
        y = str(x.find('p',{'class':'h4'}))
        product_listings = int(re.findall(reviews_number, y)[0])
        for_range = round(product_listings / 25)
        middle_range = (for_range*25)
        add_this = product_listings - middle_range
    list_of_review_pages = []
    for i in range(0,middle_range+add_this, 25):
        urlback = '?f={numba}#products'.format(numba=i)
        back = str(urlback)
        final_link_concat = front + back
        print('Page Number: ', final_link_concat, '\n')
        list_of_review_pages.append(final_link_concat)
    return list_of_review_pages

def third_layer_scrape(product_specific_reviews_url):

    """ this function is the same as clicking the reviews button on the product page and then finding the individual review links"""
    list_of_links = []
    """ goes down another layer and will retrieve the link to each user's reviews"""
    URL_3 = 'https://www.trustradius.com{sub_category}'.format(sub_category=product_specific_reviews_url)
    print('Use this to get review links list', URL_3)
    user_reviews_page_links = get_all_review_pages(URL_3)
    for urpl in user_reviews_page_links:
        session = requests.Session()
        # set username and password
        session.auth = ('user', 'pass')
        # update headers
        session.headers.update({'x-test': 'true'})
        page_3 = session.get(urpl, headers ={'x-test2': 'true'})
        # pause the loop - for human-like appearance
        time.sleep(randint(1, 5))
        # instantiate a BS object with our page content for it to parse over
        soup_3 = BeautifulSoup(page_3.text, 'html.parser')
        user_review_link = soup_3.find_all('a', class_='link-to-review-btn btn btn-block btn-primary')
        for c in user_review_link:
            ureview_link = c['href']
            list_of_links.append(ureview_link)
        print('Review Links Captured', len(list_of_links))

    counter_lol = 0
    for x in list_of_links:
        counter_lol += 1
        current_link = 'https://www.trustradius.com{linkn}'.format(linkn=x)
        pass_to_database(current_link)
        pass_to_database_body(current_link)
        print('Successfully Passed:', x, current_link, counter_lol)


def secondly_retrieve_product_info(subcat_type_url):

    counter_var = 0
    """ goes one layer deeper and will retrieve the links to each of the subcategories"""
    URL_2 = 'https://www.trustradius.com{sub_category}'.format(sub_category=subcat_type_url)
    all_URL_2_pages = get_all_pages(URL_2)
    for url in all_URL_2_pages:
        session = requests.Session()
        # set username and password
        session.auth = ('user', 'pass')
        # update headers
        session.headers.update({'x-test': 'true'})
        print('******** CURRENT SUB-CATEGORY ******','\n', str(url).replace('?f=0#products', '').replace('-', ' ').split('.com/')[1].upper(), '\n')
        page_2 = session.get(url, headers ={'x-test2': 'true'})
        # pause the loop - for human-like appearance
        time.sleep(randint(1, 8))
        soup_2 = BeautifulSoup(page_2.text, 'html.parser')
        # pulls the links for the reviews page for each product
        product_list_reviews_link = soup_2.select('.CategoryProduct_links__3hCwj a')
        for b in product_list_reviews_link:
            if b.get_text(strip=True) == 'Reviews':
                reviews_link = b['href']
                third_layer_scrape(reviews_link)
                counter_var += 1
                print('Platforms Located: ', counter_var, '\n', '\n')


def first_find_sub_cat_links(cat_links_list):
    """first layer which finds the subcategory links"""
    for sub in cat_links_list:
        # var the URL to scrape
        URL = 'https://www.trustradius.com/{sub_category}'.format(sub_category=sub)
        # send the page request
        session = requests.Session()
        # set username and password
        session.auth = ('user', 'pass')
        # update headers
        session.headers.update({'x-test': 'true'})
        page = session.get(URL, headers ={'x-test2': 'true'})
        # pause the loop - for human-like appearance
        time.sleep(randint(1, 4))
        # instantiate a BS object with our page content for it to parse over
        soup = BeautifulSoup(page.text, 'html.parser')
        # this will be the div we will be scraping 'within'
        subcat_link_css = soup.select('.link-list-columns a')
        for a in subcat_link_css:
            sub_cat_link_main = a['href']
            secondly_retrieve_product_info(sub_cat_link_main)





# -- end scraper components

test_1 = first_find_sub_cat_links(cat_links_list)





requestn += 1
elapsed_time = time.time() - start_time
print('Requests: ', requestn, 'Elapsed Time: ', elapsed_time/60, 'Reviews Scraped: ', reviews_scraped)


# -- closes the connection and the cursor when the full code is finished running
cursor.close()
cnx.close()
