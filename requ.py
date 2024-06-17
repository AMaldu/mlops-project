# Ruta del archivo requirements.txt original
input_file = 'quirements.txt'
# Ruta del archivo requirements.txt formateado
output_file = 'requirements.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Dividir la línea en nombre del paquete y versión
        if '=' in line:
            parts = line.split('=')
            package = parts[0]
            version = parts[1]
            # Escribir la línea formateada en el nuevo archivo
            outfile.write(f"{package}=={version}\n")

print(f"Requisitos formateados guardados en {output_file}")
