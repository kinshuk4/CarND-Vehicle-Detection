import urllib
import zipfile
import os


def download_data_one(url, download_path):
    if not os.path.exists(download_path):
        os.mkdir(download_path)

    print("Downloading data from " + url)

    path = "./data/%s" % (url.split('/')[-1])
    urllib.request.urlretrieve(url, path)
    zipfile.ZipFile(path, 'r').extractall(download_path)


def download_data(urls, download_path):
    print("Downloading data...................")
    for url in urls:
        download_data_one(url, download_path)
