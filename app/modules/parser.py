# app/modules/parser.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io
import time
import requests
import logging

logger = logging.getLogger(__name__)

def get_selenium_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-images")  # ускоряет загрузку
    options.binary_location = "/usr/bin/chromium"
    service = Service(ChromeDriverManager(driver_version="142.0.7444.175").install())  # без указания версии
    return webdriver.Chrome(service=service, options=options)

def parse_marketplace(url: str):
    if "wildberries.ru" not in url and "ozon.ru" not in url:
        raise ValueError("Поддерживаются только wildberries.ru и ozon.ru")

    driver = get_selenium_driver()
    try:
        driver.get(url)
        time.sleep(5)  # даём время на полную загрузку

        if "wildberries.ru" in url:
            return _parse_wildberries(driver)
        elif "ozon.ru" in url:
            return _parse_ozon(driver)
    finally:
        driver.quit()

def _parse_wildberries(driver):
    wait = WebDriverWait(driver, 20)  # увеличьте таймаут

    # 🔹 Название: ищем <h3>, содержащий "productTitle" в class
    try:
        title_elem = wait.until(
            EC.presence_of_element_located((By.XPATH, "//h3[contains(@class, 'productTitle')]"))
        )
        title = title_elem.text.strip()
        if not title or len(title) < 5:
            # Резерв: любой h3 с длинным текстом
            title_elem = wait.until(
                EC.presence_of_element_located((By.XPATH, "//h3[string-length(text()) > 10]"))
            )
            title = title_elem.text.strip()
    except Exception as e:
        # Для отладки: сохраните HTML
        with open("/app/wb_debug.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        raise RuntimeError(f"Не удалось найти название на Wildberries: {str(e)}")

    # 🔹 Изображение: ищем <img> внутри блока с "imgContainer"
    try:
        img_elem = wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'imgContainer')]//img"))
        )
        img_url = img_elem.get_attribute("src") or img_elem.get_attribute("data-src")
        if img_url and not img_url.startswith("http"):
            img_url = "https:" + img_url
    except Exception as e:
        raise RuntimeError(f"Не удалось найти изображение на Wildberries: {str(e)}")

    return _download_image(img_url, title)

def _parse_ozon(driver):
    wait = WebDriverWait(driver, 20)

    try:
        title_elem = wait.until(
            EC.presence_of_element_located((By.XPATH, "//h1[contains(@class, 'tsHeadline')]"))
        )
        title = title_elem.text.strip()
    except Exception as e:
        with open("/app/ozon_debug.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        raise RuntimeError(f"Не удалось найти название на Ozon: {str(e)}")

    try:
        img_elem = wait.until(
            EC.presence_of_element_located((By.XPATH, "//img[contains(@src, 'multimedia')]"))
        )
        img_url = img_elem.get_attribute("src")
        if img_url and not img_url.startswith("http"):
            img_url = "https:" + img_url
    except Exception as e:
        raise RuntimeError(f"Не удалось найти изображение на Ozon: {str(e)}")

    return _download_image(img_url, title)

def _download_image(img_url: str, title: str):
    if not img_url:
        raise RuntimeError("Пустой URL изображения")

    # Очищаем URL от параметров и меняем .webp → .jpg
    if "?" in img_url:
        img_url = img_url.split("?")[0]
    if img_url.endswith(".webp"):
        img_url = img_url.replace(".webp", ".jpg")

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    response = requests.get(img_url, headers=headers, timeout=10)
    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image, title