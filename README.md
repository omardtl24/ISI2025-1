# Introducción a los Sistemas Inteligentes - 2025-1

Este repositorio reúne todas las prácticas, talleres y laboratorios desarrollados durante el curso **Introducción a los Sistemas Inteligentes**, correspondiente al semestre **2025-1**.

## 📚 Contenido

El repositorio se encuentra organizado por el tipo de actividad, e incluye:

- 🧠 Prácticas asíncronicas
- 🧪 Laboratorios individuales
- 📄 Códigos auxiliares
- 🤖 Proyecto

## 📁 Estructura del repositorio

El repositorio está organizado en las siguientes carpetas principales:

- `AI_algorithms/`: Implementaciones y utilidades de algoritmos de inteligencia artificial, aprendizaje supervisado/no supervisado y grafos de estados.
- `Laboratorios/`: Notebooks de laboratorio con ejercicios prácticos de Python, Numpy, Pandas, limpieza y construcción de datos, generación y evaluación de modelos, series de tiempo, clustering, redes neuronales, entre otros. Incluye datasets de ejemplo.
- `Practicas/`: Prácticas asíncronas para reforzar conceptos vistos en clase y laboratorios.
- `Repaso Quices/`: Notebooks de repaso y preparación para quices.

## 🐍 Entorno virtual y dependencias

Es fundamental trabajar en un entorno virtual para evitar conflictos de dependencias y asegurar la reproducibilidad. Puedes crear un entorno virtual con:

```bash
python3 -m venv ISI-venv
source ISI-venv/bin/activate
pip install -r requirements.txt
```

El archivo `requirements.txt` contiene todos los paquetes necesarios para el curso. Los principales paquetes utilizados son:

- **numpy** y **scipy**: Manipulación eficiente de arreglos y operaciones matemáticas avanzadas.
- **pandas**: Manejo y análisis de datos tabulares.
- **matplotlib** y **seaborn**: Visualización de datos y gráficos estadísticos.
- **scikit-learn**: Algoritmos de machine learning y herramientas de modelado.
- **kaggle**: Descarga de datasets y participación en competencias desde notebooks.
- **jupyter** y **notebook**: Ejecución interactiva de notebooks.
- **python-dotenv**: Manejo de variables de entorno para credenciales y configuraciones.
- **tensorflow, keras y dependencias nvidia**: Para el desarrollo y entrenamiento de modelos de deep learning. **Nota:** Las versiones incluidas están pensadas para equipos con GPU compatible (CUDA/cuDNN). Si usas CPU, revisa la documentación oficial para instalar versiones compatibles.

Estos paquetes permiten realizar desde análisis y limpieza de datos, hasta la implementación y evaluación de modelos de machine learning y deep learning, así como la visualización y documentación interactiva de los resultados.

> Recuerda activar siempre tu entorno virtual antes de trabajar y asegurarte de tener instaladas las dependencias con `pip install -r requirements.txt`.


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

## 🔒 Acceso a soluciones calificadas

Las soluciones de los ejercicios y laboratorios que son parte de la evaluación del curso están **encriptadas** por motivos de seguridad y honestidad académica. Solo pueden ser accedidas por los estudiantes autorizados mediante una contraseña segura proporcionada por el docente.

Para acceder a estas soluciones:
- Utiliza el script `decrypt.py` incluido en el repositorio.
- Ingresa la contraseña cuando se te solicite.
- El script desencriptará y extraerá los archivos protegidos en las carpetas correspondientes.

> **Importante:** Si no tienes la contraseña, no podrás acceder a las soluciones calificadas. El resto del contenido del repositorio (notebooks, prácticas, utilidades y ejemplos) está disponible libremente para consulta, aprendizaje y exploración.

