import os
import sys

print("=== DIAGNÓSTICO ===")
print(f"Directorio actual: {os.getcwd()}")

# Verificar qué archivos hay
print("\nArchivos en carpeta actual:")
for archivo in os.listdir('.'):
    print(f"  - {archivo}")

# Verificar carpeta cellcounter
if os.path.exists('cellcounter'):
    print("\nArchivos en cellcounter/:")
    for archivo in os.listdir('cellcounter'):
        print(f"  - {archivo}")
else:
    print("\n❌ No existe carpeta 'cellcounter'")

# Intentar importar
try:
    import cellcounter
    print(f"\n✅ cellcounter importado desde: {cellcounter.__file__}")
    print(f"Funciones disponibles: {dir(cellcounter)}")
except Exception as e:
    print(f"\n❌ Error al importar: {e}")