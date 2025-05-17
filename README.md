# Introducción a los Sistemas Inteligentes - 2025-1

Este repositorio reúne todas las prácticas, talleres y laboratorios desarrollados durante el curso **Introducción a los Sistemas Inteligentes**, correspondiente al semestre **2025-1**.

## 📚 Contenido

El repositorio se encuentra organizado por el tipo de actividad, e incluye:

- 🧠 Prácticas asíncronicas
- 🧪 Laboratorios individuales
- 📄 Códigos auxiliares
- 🤖 Proyecto

## 🔑 Configuración de variables de entorno para Kaggle

Algunos laboratorios requieren acceso a la API de Kaggle para descargar datasets. Para esto, debes configurar una variable de entorno con la ruta a tu archivo `kaggle.json` (tu credencial secreta de Kaggle).

### Linux/MacOS
1. Descarga tu archivo `kaggle.json` desde [https://www.kaggle.com/settings](https://www.kaggle.com/settings) (sección "API").
2. Crea una carpeta `.kaggle` en tu directorio personal (si no existe):
   ```bash
   mkdir -p ~/.kaggle
   ```
3. Copia el archivo `kaggle.json` a esa carpeta:
   ```bash
   cp /ruta/al/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Crea un archivo `.env` en la raíz del repositorio (puedes usar `.env.example` como plantilla):
   ```env
   KAGGLE_CONFIG_DIR=/home/tu_usuario/.kaggle
   ```
   Cambia `/home/tu_usuario/` por tu ruta real de usuario.

### Windows
1. Descarga tu archivo `kaggle.json` desde [https://www.kaggle.com/settings](https://www.kaggle.com/settings) (sección "API").
2. Crea una carpeta `.kaggle` en tu carpeta de usuario (por ejemplo, `C:\Users\tu_usuario\.kaggle`).
3. Copia el archivo `kaggle.json` a esa carpeta.
4. Crea un archivo `.env` en la raíz del repositorio (puedes usar `.env.example` como plantilla):
   ```env
   KAGGLE_CONFIG_DIR=C:\\Users\\tu_usuario\\.kaggle
   ```
   Cambia `tu_usuario` por tu nombre de usuario real de Windows.

Esto permitirá que los scripts y notebooks accedan a la API de Kaggle de forma segura, sin exponer tu credencial en el código.

> **Nota:** ¡Nunca subas tu archivo `kaggle.json` ni tu `.env` real a un repositorio público!