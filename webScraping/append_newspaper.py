# webScrapping/append_newspaper.py
import json

# Archivos de entrada y salida
input_file1 = "news_2.json"  # Archivo con el primer JSON
input_file2 = "news_cleaned.json"  # Archivo con el segundo JSON
output_file = "resultado.json"  # Archivo donde se guardará el resultado

# Leer el primer JSON desde el archivo
with open(input_file1, "r", encoding="utf-8") as file1:
    json1 = json.load(file1)

# Leer el segundo JSON desde el archivo
with open(input_file2, "r", encoding="utf-8") as file2:
    json2 = json.load(file2)

# Crear un diccionario para buscar rápidamente por ID en el segundo JSON
id_to_newspaper = {item["id"]: item["newspaper"] for item in json2}

# Agregar el campo "newspaper" al primer JSON
for item in json1:
    item["newspaper"] = id_to_newspaper.get(item["Id"], None)

# Guardar el resultado en un archivo de salida
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(json1, outfile, indent=4, ensure_ascii=False)

print(f"El resultado se ha guardado en '{output_file}'.")
