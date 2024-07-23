from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import csv
import time
import os
import openai
import re
from dotenv import load_dotenv, find_dotenv

# Loading AI features
load_dotenv(find_dotenv())

# Set the API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def launchBrowser():
    chrome_options = Options()
    chrome_options.add_argument("start-maximized")
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    url = "https://bottlebuzz.com/collections/rare"
    driver.get(url)
    return driver
def getDescription(product):
    response = openai.ChatCompletion.create(
    model="gpt-4-turbo",
    messages=[
        {
        "role": "system",
        "content": "You are a helpful assistant who will receive the full names of bottles with Names and you will give a description for this product to be placed on an e-commerce-optimized for the best SEO and I want it like this description for example - Size: 750 ml Proof: 95.6 Created: Buffalo Trace Distillery Aged 23 Years Vanilla Notes Rich And Complex Taste Pappy Van Winkle 23 Year Bourbon is part of a series of highly sought bourbons. This one has been aged for 23 years. This luxurious line of bourbon is named after the iconic Julian P. Van Winkle, Sr. who was called Pappy by family and friends who appreciated him dearly. It is produced at the award-winning Buffalo Trace Distillery. This marvelous blend is crafted from Mash Bill #2, like other great blends such as Blanton's, William Larue, and Elmer T. Lee's. It is produced with a sour mash bill of corn, malted barley, and extra rye. This bourbon is one of the most sought-after whiskies. It is rich and with vibrant aromas of ripe apples, cherries, caramel oak wood, and tobacco with a hint of chocolate. The finish is pleasant with a lot of wooden notes, and a nice sweet caramel finish. History Pappy Van Winkle 23 Year Bourbon is known as unicorn's in the bourbon community. It is an iconic bourbon created to commemorate the legacy of Julian P. Van Winkle, Sr. best known as Pappy by those who knew him and loved him. Pappy began working at W.L. Weller & Sons in 1893 as a traveling whiskey salesman and eventually became the President of Stitzel-Weller Distillery. During this time he sold Old Fitzgerald and earned an outstanding reputation for his commitment to a fine bourbon. His vow is still honored today. During this time bourbon wasn't selling and Pappy's son, Julian Jr., ended up selling the business in 1972. It was one of his biggest regrets so he formed a company called +J.P. Van Winkle and Son+ just in case his son wanted to continue the bourbon business. He got into marketing bourbon through many types of commemorative decanters and would bottle on the side. In 1981 Julian III took over the business after his father passed away. He was 32 years old and married with 4 children, one boy, and triplet girls. During this time, there wasn't a lot of demand for premium bourbon. But, he didn't let that discourage him. As his grandfather, Pappy started from the ground up, too. So, with the same passion in mind, Julian the Third purchased the Old Hoffman Distillery in Lawrenceburg, Kentucky, for barrel storage and bottling purposes. He couldn't afford to advertise, but the quality of Old Rip Van Winkle sold itself. In 1997 Julian III scored 99/100 from a beverage testing institute. Soon he was being contacted by people interested in carrying his product. When first approached by Buffalo Trace, Julian III was honored, but not immediately interested. He had enough bourbon aging, but the concern was for future reserves since bourbon was increasing in popularity. Buffalo Trace bought the W.L. Weller label in 1999 and has been making the bourbon with nearly the same recipe as Pappy. The transition was easy, and as of May 2002, Buffalo Trace has produced the Van Winkle bourbons, using Pappy's exact recipe. This change allows Old Rip Van Winkle to maintain their strict quality standards while producing more barrels for future enjoyment. Please drink responsibly, you must be at least 21 years of age to drink alcoholic beverages.- Dont copy this description this is the format you follow. Please search online for all information on these bottles"
        },
        {
        "role": "user",
        "content": product
        }
    ],
    temperature=0.88,
    max_tokens=4095,
    top_p=0.86,
    frequency_penalty=0.5,
    presence_penalty=0.9
    )
    # Extracting and printing the generated text
    generated_text = response['choices'][0]['message']['content'].strip()
    return generated_text

def slow_scroll(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

driver = launchBrowser()
time.sleep(15)

# Open a CSV file to store the scraped data
with open('products2.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Product Name', 'Brand', 'Price','Description','Category'])

    while True:
        slow_scroll(driver)
        products = driver.find_elements(By.CLASS_NAME, 'grid-item__meta')

        for product in products:
            raw_name = product.find_element(By.CLASS_NAME, 'grid-product__title').text
            # Remove all types of quotation marks
            name = re.sub(r'["\'\”\“\‘\’]', '', raw_name)
            if name != "":  # This checks if 'name' is not empty
                print(f"Name: {name}") 
                brand = product.find_element(By.CLASS_NAME, 'grid-product__vendor').text
                price_element = product.find_element(By.CSS_SELECTOR, '.grid-product__price--current .visually-hidden')
                price_text = price_element.text.strip().replace('$', '').replace(',', '')
                description = getDescription(name)
                writer.writerow([name, brand, price_text, description, 'Rare'])
            else:
                continue  # Skip this iteration if the name is empty
        try:
            next_page = driver.find_element(By.XPATH, "//a[@title='Next']")
            next_page.click()
            time.sleep(5)
        except NoSuchElementException:
            break

print("Scraping completed and data saved to CSV.")
driver.quit()
