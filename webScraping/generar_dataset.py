# webScrapping/generar_dataset.py
import bs4
import requests
import sys
import json
import os
from datetime import datetime
from shutil import copy
from crawlers.ElMundo import ElMundo
from crawlers.ElDiario import ElDiario
from crawlers.ElPais import ElPais
from crawlers.ElPublico import ElPublico
from crawlers.LaRazon import LaRazon
from crawlers.ElConfidencial import ElConfidencial
from crawlers.ABC import ABC
from crawlers.InfoLibre import InfoLibre
from crawlers.ElPlural import ElPlural
from crawlers.ElEspanol import ElEspanol


def manage_safety_copies(directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    date_str = datetime.now().strftime("%d%m%y-%H%M%S")
    safety_filename = os.path.join(directory, f"{date_str}.json")

    copy(filename, safety_filename)

    files = sorted(os.listdir(directory))
    if len(files) > 3:
        os.remove(os.path.join(directory, files[0]))


def manage_debug_files(directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    date_str = datetime.now().strftime("%d%m%y-%H%M%S")
    safety_filename = os.path.join(directory, f"{date_str}.json")

    copy(filename, safety_filename)

    files = sorted(os.listdir(directory))
    if len(files) > 3:
        os.remove(os.path.join(directory, files[0]))


def restore_dataset(safety_directory, dataset_filename):
    if not os.path.exists(safety_directory):
        print("No safety copies found.")
        return

    files = sorted(os.listdir(safety_directory))
    if not files:
        print("No safety copies found.")
        return

    latest_safety_copy = os.path.join(safety_directory, files[-1])
    if os.path.exists(dataset_filename):
        os.remove(dataset_filename)
    copy(latest_safety_copy, dataset_filename)
    print(f"Restored dataset from {latest_safety_copy}")


def main(args):
    if len(args) < 2:
        mode = "-h"
    else:
        mode = args[1]

    

    if mode == "-h":
        print("Uso: python generar_dataset.py [Opcion]")
        print("-h para ayuda")
        print("-s para iniciar el scrapping")
        print("-d para debug")
        print("-r para restaurar el dataset desde la última copia de seguridad")

    elif mode == "-s":
        # innit crawlers 
        crawlers = [
            ElMundo("https://www.elmundo.es"),
            ElDiario("https://www.eldiario.es"),
            ElPais("https://elpais.com"),
            ElPublico("https://www.publico.es"),
            LaRazon("https://www.larazon.es"),
            ElConfidencial("https://www.elconfidencial.com"),
            ABC("https://www.abc.es"),
            InfoLibre("https://www.infolibre.es"),
            ElPlural("https://www.elplural.com"),
            ElEspanol("https://www.elespanol.com/")

        ]
        # realizamos el scrapping
        result_data = []
        for crawler in crawlers:
            result_data.append(crawler.crawl())
            print(f"Finished crawling {crawler.newspaper} at {datetime.now().strftime('%H:%M:%S')} with {len(result_data[-1])} news")


        if os.path.exists("news_dataset.json"):
            # copias de seguridad para evitar sustos innecesarios
            manage_safety_copies("safety", "news_dataset.json")
            # añadir al dataset
            with open("news_dataset.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)  
        else:
            existing_data = []
        
        for result in result_data:
            existing_data.extend(result)
 

        with open("news_dataset.json", "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    elif mode == "-d":
        crawlers = [
            ElMundo("https://www.elmundo.es"),
            ElDiario("https://www.eldiario.es"),
            ElPais("https://elpais.com"),
            ElPublico("https://www.publico.es"),
            LaRazon("https://www.larazon.es"),
            ElConfidencial("https://www.elconfidencial.com"),
            ABC("https://www.abc.es"),
            InfoLibre("https://www.infolibre.es"),
            ElPlural("https://www.elplural.com")   
        ]
        result_data = []
        for crawler in crawlers:
            temp_res = crawler.crawl()
            if not temp_res:
                continue
            result_data.append(temp_res)
            print(f"Finished crawling {crawler.newspaper} at {datetime.now().strftime('%H:%M:%S')} with {len(result_data[-1])} news")
        with open("news_dataset_debug.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        manage_debug_files("debug", "news_dataset_debug.json")


    elif mode == "-r":
        restore_dataset("safety", "news_dataset.json")


if __name__ == "__main__":
    main(sys.argv)
