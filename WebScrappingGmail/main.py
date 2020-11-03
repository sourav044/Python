from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

# https://gist.github.com/0xv/8aeb66c0f75e13a20eb148f86900e203

EMAIL = ' '
PASSWORD = ' '
CHROMEPATH = 'C:\\Users\\Sourav\\Downloads\\chromedriver_win32\\chromedriver.exe'
comment = 'wonderful movie'

youtube_links = ['https://www.youtube.com/watch?v=UcATP_YngfM&ab_channel=ShemarooMovies']

login_url = 'https://www.google.com/accounts/Login'
google_html = 'https://mail.google.com/mail/u/0/h/1pq68r75kzvdr/?v%3Dlui'


def youtube(browser):
    for domain in youtube_links:
        browser.get(domain)
        time.sleep(10)
        browser.execute_script("window.scrollTo(0, 500);")
        WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.ID, "comments")))
        time.sleep(4)
        # like
        like = browser.find_elements_by_xpath(
            "//div[@id='menu-container']/div/ytd-menu-renderer/div/ytd-toggle-button-renderer/a")
        ActionChains(browser).move_to_element(like[0]).click().perform()
        time.sleep(10)
        # Comment
        commentBox = browser.find_element_by_id("placeholder-area")
        time.sleep(10)
        ActionChains(browser).move_to_element(commentBox).perform()
        ActionChains(browser).move_to_element(commentBox).click().perform()
        time.sleep(2)
        box = browser.find_element_by_id("contenteditable-root")
        time.sleep(2)
        ActionChains(browser).move_to_element(box).perform()
        ActionChains(browser).move_to_element(box).click().perform()
        time.sleep(2)
        browser.find_element_by_id('contenteditable-root').send_keys(comment)
        time.sleep(2)
        submit = browser.find_element_by_id("submit-button")
        time.sleep(2)
        ActionChains(browser).move_to_element(submit).perform()
        ActionChains(browser).move_to_element(submit).click().perform()
        time.sleep(2)


def gmail_data():
    browser = webdriver.Chrome(CHROMEPATH)
    browser.set_window_position(0, 0)
    browser.get(login_url)
    time.sleep(2)
    browser.find_element_by_id("identifierId").send_keys(EMAIL)
    browser.find_element_by_id("identifierNext").click()
    time.sleep(2)
    browser.find_element_by_name("password").send_keys(PASSWORD)
    browser.find_element_by_id("passwordNext").click()
    time.sleep(2)

    # Google Emails
    browser.get(google_html)
    mytable = browser.find_element_by_class_name('th')
    for row in mytable.find_elements_by_css_selector('tr'):
        for cell in row.find_elements_by_tag_name('td'):
            print(cell.text)

    time.sleep(4)
    youtube(browser)
    # time.sleep(4)
    # print(soup.prettify())


if __name__ == '__main__':
    gmail_data()
