import time
import requests
from bs4 import BeautifulSoup


def get_Data(url):
    # Gets html from url
    result = requests.get(url)
    doc = BeautifulSoup(result.text, "html.parser")

    # Finds the class containing the useful info
    keys = doc.find(class_="product-properties-details linear-gradient").find_all(class_="property-key")
    values = doc.find(class_="product-properties-details linear-gradient").find_all(class_="property-value")
    price = doc.find(class_="product-price").getText(strip=True)

    # Creates dictionary of (key, value) pairs
    information = {"Fiyat": price[:-3]}

    for i in range(len(values)):
        information[keys[i].getText(strip=True)] = values[i].getText(strip=True)

    # Removes unwanted word at the beginning of dictionary
    information["İlan No"] = information["İlan No"][10:]

    return information


def get_by_year(year):
    all_links = []
    data = []
    km_intervals = [0, 90000, 150000, 200000, 300000, 500000, 750000]

    for i in range(len(km_intervals) - 1):

        minkm = km_intervals[i] + 1
        maxkm = km_intervals[i + 1]
        print("checking interval:", minkm, "-", maxkm)

        # Gets html from url by indicating a km interval
        url = "https://www.arabam.com/ikinci-el/otomobil?maxYear=" + str(year) + "&maxkm=" + str(
            maxkm) + "&minYear=" + str(year) + "&minkm=" + str(minkm) + "&take=50"

        result = requests.get(url)
        soup = BeautifulSoup(result.text, "html.parser")

        # finds the div that state the total page
        pagination_div = soup.find('div', class_='listing-new-pagination cb tac mt16 pt16', id='pagination')
        #print(pagination_div)

        # gets the page number, if it is null set it to 1
        if pagination_div:
            page_number = pagination_div.find('span', id='js-hook-for-total-page-count').text.strip()
        else:
            page_number = 1

        print("total page number:", page_number)

        # for each page gets the data
        for page in range(1, int(page_number) + 1):
            print("page:", page)
            # gets the html from page url
            page_url = url + "&page=" + str(page)
            page_result = requests.get(page_url)
            page_soup = BeautifulSoup(page_result.text, "html.parser")

            # find the all parts that includes a link for unique adverts
            link_soup = page_soup.find_all('a', href=lambda href: href and href.startswith('/ilan'),
                                           class_='link-overlay')
            links = set([link["href"] for link in link_soup])  # extract just the links

            all_links.extend(links)

    print(len(all_links))
    i = 0
    failed_links = []
    for link in all_links:
        advert_url = "https://www.arabam.com" + link
        try:
            data.append(get_Data(advert_url))
            i += 1
            if i % 64 == 0:
                print(f"{i} data is processed")
            if i % 900 == 0:
                print("sleeping while processing")
                time.sleep(5)
        except:
            failed_links.append(advert_url)

    print(len(failed_links), "failed links in year", year)
    with open('failed_links.txt', 'a') as f:
        for link in failed_links:
            f.write(link + "\n")

    return data


def get_by_year_panelvan(year):
    all_links = []
    data = []
    km_intervals = [0, 90000, 150000, 200000, 300000, 500000, 750000]

    for i in range(len(km_intervals) - 1):

        minkm = km_intervals[i] + 1
        maxkm = km_intervals[i + 1]
        print("checking interval:", minkm, "-", maxkm)

        # Gets html from url by indicating a km interval
        url = "https://www.arabam.com/ikinci-el/minivan-van_panelvan?maxYear=" + str(year) + "&maxkm=" + str(
            maxkm) + "&minYear=" + str(year) + "&minkm=" + str(minkm) + "&take=50"

        result = requests.get(url)
        soup = BeautifulSoup(result.text, "html.parser")

        # finds the div that state the total page
        pagination_div = soup.find('div', class_='listing-new-pagination cb tac mt16 pt16', id='pagination')
        #print(pagination_div)

        # gets the page number, if it is null set it to 1
        if pagination_div:
            page_number = pagination_div.find('span', id='js-hook-for-total-page-count').text.strip()
        else:
            page_number = 1

        print("total page number:", page_number)

        # for each page gets the data
        for page in range(1, int(page_number) + 1):
            print("page:", page)
            # gets the html from page url
            page_url = url + "&page=" + str(page)
            page_result = requests.get(page_url)
            page_soup = BeautifulSoup(page_result.text, "html.parser")

            # find the all parts that includes a link for unique adverts
            link_soup = page_soup.find_all('a', href=lambda href: href and href.startswith('/ilan'),
                                           class_='link-overlay')
            links = set([link["href"] for link in link_soup])  # extract just the links

            all_links.extend(links)

    print(len(all_links))
    i = 0
    failed_links = []
    for link in all_links:
        advert_url = "https://www.arabam.com" + link
        try:
            data.append(get_Data(advert_url))
            i += 1
            if i % 64 == 0:
                print(f"{i} data is processed")
            if i % 900 == 0:
                print("sleeping while processing")
                time.sleep(5)
        except:
            failed_links.append(advert_url)

    print(len(failed_links), "failed links in year", year)
    with open('failed_links.txt', 'a') as f:
        for link in failed_links:
            f.write(link + "\n")

    return data

def get_by_year_suv(year):
    all_links = []
    data = []
    km_intervals = [0, 90000, 150000, 200000, 300000, 500000, 750000]

    for i in range(len(km_intervals) - 1):

        minkm = km_intervals[i] + 1
        maxkm = km_intervals[i + 1]
        print("checking interval:", minkm, "-", maxkm)

        # Gets html from url by indicating a km interval
        url = "https://www.arabam.com/ikinci-el/arazi-suv-pick-up?maxYear=" + str(year) + "&maxkm=" + str(
            maxkm) + "&minYear=" + str(year) + "&minkm=" + str(minkm) + "&take=50"

        result = requests.get(url)
        soup = BeautifulSoup(result.text, "html.parser")

        # finds the div that state the total page
        pagination_div = soup.find('div', class_='listing-new-pagination cb tac mt16 pt16', id='pagination')
        #print(pagination_div)

        # gets the page number, if it is null set it to 1
        if pagination_div:
            page_number = pagination_div.find('span', id='js-hook-for-total-page-count').text.strip()
        else:
            page_number = 1

        print("total page number:", page_number)

        # for each page gets the data
        for page in range(1, int(page_number) + 1):
            print("page:", page)
            # gets the html from page url
            page_url = url + "&page=" + str(page)
            page_result = requests.get(page_url)
            page_soup = BeautifulSoup(page_result.text, "html.parser")

            # find the all parts that includes a link for unique adverts
            link_soup = page_soup.find_all('a', href=lambda href: href and href.startswith('/ilan'),
                                           class_='link-overlay')
            links = set([link["href"] for link in link_soup])  # extract just the links

            all_links.extend(links)

    print(len(all_links))
    i = 0
    failed_links = []
    for link in all_links:
        advert_url = "https://www.arabam.com" + link
        try:
            data.append(get_Data(advert_url))
            i += 1
            if i % 64 == 0:
                print(f"{i} data is processed")
            if i % 900 == 0:
                print("sleeping while processing")
                time.sleep(5)
        except:
            failed_links.append(advert_url)

    print(len(failed_links), "failed links in year", year)
    with open('failed_links.txt', 'a') as f:
        for link in failed_links:
            f.write(link + "\n")

    return data

def get_all():
    data = []
    for year in range(1973, 2025):
        print(f"\n######################\nStarting {year}...\n######################\n")
        data.extend(get_by_year(year))
        data.extend(get_by_year_panelvan(year))

    return data


def write_to_csv(data):

    features = ['Fiyat', 'İlan No', 'İlan Tarihi', 'Marka', 'Seri', 'Model', 'Yıl', 'Kilometre', 'Vites Tipi',
                'Yakıt Tipi',
                'Kasa Tipi', 'Renk', 'Motor Hacmi', 'Motor Gücü', 'Çekiş', 'Araç Durumu', 'Ort. Yakıt Tüketimi',
                'Yakıt Deposu', 'Boya-değişen', 'Takasa Uygun', 'Kimden']

    with open('dataSet(backup1).csv', 'a') as f:

        # Write all the dictionary keys in a file with commas separated.
        #f.write(",".join(features))
        #f.write("\n")

        print("Writing is starting")

        for advert in data:
            for feature in features:
                # Write the values in a row.
                try:
                    if feature == "Kimden":
                        f.write(advert[feature].replace(",", " "))
                    else:
                        f.write(advert[feature].replace(",", " ") + ",")
                except KeyError:
                    if feature == "Kimden":
                        f.write("None")
                    else:
                        f.write("None" + ",")

            f.write('\n')


if __name__ == '__main__':

    # data = get_all()
    # write_to_csv(data)
    for i in range(2022,2025):
        print("year:",i)
        data = get_by_year_suv(i)
        write_to_csv(data)
        print("year",i,"is done")
        print("Sleeping...")
        time.sleep(10)


