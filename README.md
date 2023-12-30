![STEAM](2.png)

# Proyecto MLOps: Sistema de Recomendación de Videojuegos para Usuarios de Steam

## Descripción del Proyecto:

Este proyecto se destaca como una parte esencial de la fase de LABS en el bootcamp de Henry, enfocándose en la aplicación práctica de habilidades técnicas y competencias blandas vitales en el mercado laboral. Exploramos un caso de negocio genuino mediante el uso de conjuntos de datos públicos de la industria de videojuegos, específicamente de la reconocida plataforma en línea STEAM.

## Objetivo:

El objetivo principal radica en la creación del primer modelo integral de Machine Learning (end to end) diseñado para abordar un desafío de negocio en Steam. Nuestro enfoque abarca tareas de Data Engineering (ETL, EDA, API) hasta la implementación del modelo de Machine Learning. Buscamos lograr un desarrollo ágil y obtener un Producto Mínimo Viable (MVP) de manera eficiente.

## Etapas del Proyecto

![sistema de recomendación](DiagramaConceptualDelFlujoDeProcesos.png)

1. __Ingeniería de Datos (ETL y API)__ 

1.1 __Proceso de Transformación de Datos:__ Inicialmente, me encontré con tres (3) archivos en formato JSON que estaban almacenados en la carpeta __"Dataset"__ de un repositorio público en [Google Drive](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj). Llevé esos archivos a una carpeta personal para facilitar mi trabajo. Puedes acceder a la carpeta en [Google Drive](https://drive.google.com/drive/folders/12GjkQx39pIPNU9mpOUL-Xt_MgovpnFkC?usp=sharing) 
. Realicé transformaciones esenciales para cargar los conjuntos de datos con el formato adecuado, con el objetivo de optimizar tanto el rendimiento de la API como el entrenamiento del modelo.

- [__australian_user_reviews.json:__](https://drive.google.com/file/d/1pLc9NEWje5NtPOWFGxsaqk-LSEtTge0B/view?usp=sharing)
 Contiene las reseñas de juegos específicamente realizadas por usuarios australianos. Se puede hacer referencia al notebook  [ETL_user_reviews ](ETL_user_reviews.ipynb)
para obtener más detalles sobre cómo se procesaron las reseñas dando como resultado un nuevo archivo con datos limpios, [user_reviews_cleaned.csv ](user_reviews_cleaned.csv)
- [__output_steam_games.json:__](https://drive.google.com/file/d/1XIz7LXIVt1fLnOtUr8wunN95G1oyuvQS/view?usp=drive_link) 
Proporciona información detallada sobre los juegos disponibles en la plataforma Steam, incluyendo géneros, etiquetas, especificaciones, desarrolladores, año de lanzamiento, precio y otros atributos relevantes. En el notebook  [ETL_steam_game ](ETL_steam_games.ipynb) puedes revisar el proceso de limpieza y transformación de datos que culmina con la creación de un nuevo archivo llamado [steam_games_cleaned.csv](steam_games_cleaned.csv)

- [__australian_users_items.json:__](https://drive.google.com/file/d/14KA5tzCUyneoVpOBKoeC0he9fuCCG5jd/view?usp=drive_link) 
Este archivo contiene información sobre los ítems relacionados con usuarios australianos y ha pasado por un proceso completo de Extracción, Transformación y Carga (ETL). Los detalles de este proceso se encuentran en el notebook [ETL_user_items](ETL_user_items.ipynb). Como resultado de este proceso, se generó un nuevo archivo  [user_items_cleaned.csv](user_items_cleaned.csv), facilitando así su manipulación y análisis para su integración en el modelo.

1.2 __Ingeniería de Características (Feature Engineering):__ Se Creó la columna de sentiment_analysis aplicando análisis de sentimiento a las reseñas de los usuarios. Opté por utilizar la biblioteca NLTK (Natural Language Toolkit) con el analizador de sentimientos de Vader. Este asigna una puntuación compuesta que se utiliza para clasificar la polaridad de las reseñas en negativas (valor '0'), neutrales (valor '1') o positivas (valor '2'). Las reseñas sin texto asignado se les dio el valor '1'. Puedes encontrar más detalles sobre el desarrollo en el notebook [ETL_user_reviews ](ETL_user_reviews.ipynb) y explorar el análisis en profundidad en el [EDA ](analisis_exploratorio_de_datos.ipynb).

1.3 __Desarrollo de API:__ Uno de los objetivos era implementar una API con FastAPI y la desplegué en Render. La API proporciona cinco (5) consultas sobre información de videojuegos. Puedes revisar el código en los notebooks [Funciones ](Funciones.ipynb) y [Consultas ](Arreglos_funciones.ipynb).

- Endpoint 1 (PlayTimeGenre): Devuelve año con mas horas jugadas para un género dado.
- Endpoint 2 (UserForGenre): Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
- Endpoint 3 (UsersRecommend): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
- Endpoint 4 (UsersWorstDeveloper): Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado.
- Endpoint 5 (sentiment_analysis): Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor.

Para acceder a la funcionalidad completa de la API y explorar las recomendaciones de juegos, visita este enlace a la  [URL de la API.](https://deploy-gabbriela07.onrender.com) En este sitio, encontrarás las diversas funciones desarrolladas. 

2. __Análisis Exploratorio de Datos (EDA)__

Investigé relaciones entre variables, identifiqué outliers y busqué patrones interesantes en los datos. El notebook  [CEDA_Análisis Exploratorio de Datos ](analisis_exploratorio_de_datos.ipynb)

3. __Modelo de Aprendizaje Automático__

Para la creación del sistema de recomendación, opté por implementar uno de los enfoques propuestos:

3.1 Sistema de Recomendación ítem-ítem:
Desarrollé un modelo que sugiere juegos similares basándose en un juego específico, empleando la técnica de similitud del coseno. Utilicé CountVectorizer para convertir los textos de la columna 'specs' en vectores numéricos, lo que facilitó el cálculo de la similitud del coseno.

La elección de la métrica de similitud del coseno fue estratégica, ya que mide el coseno del ángulo entre dos vectores. Cuanto más cercano a 1 sea el resultado, más similares son los vectores. Este enfoque resultó fundamental para determinar la semejanza entre los juegos. La similitud del coseno se emplea para generar recomendaciones, considerando que los juegos con vectores similares son tratados como sugerencias potenciales. Este método ha demostrado ser efectivo en la identificación de juegos con características comparables, fortaleciendo así la capacidad del sistema de recomendación.

4. __Implementación de MLOps__
Deploy del Modelo: Desplegué el modelo de recomendación como parte de la API, la cual puedes consultar acá: [URL de la API.](https://deploy-gabbriela07.onrender.com)


# Estructura del Repositorio

[__00 - DataSet:__ ](00 - DataSet) Almacena los datasets utilizados en una versión limpia y procesada de los mismos. Las fuentes de datos iniciales se encuentra almacenadas en la carpeta input en el siguiente repositorio [Google Drive](https://drive.google.com/drive/folders/12GjkQx39pIPNU9mpOUL-Xt_MgovpnFkC?usp=sharing) 

- __Archivos_API:__ Contiene los datasets en formato csv consumidos por la API.
- __Archivos_Limpios:__ Contiene los archivos depurados después de haber realizado el ETL.
- __Archivos_ML:__ Contiene los archivos consumidos por la API para hacer el sistema de recomendación.

[__01 - ETL:__ ](01 - ETL) Contiene los Jupyter Notebooks con el Código completo y bien comentado donde se realizaron las extracciones, transformaciones y carga de datos (ETL).

[__02 - DEF:__ ](02 - DEF) Contiene arreglos de archivos para las funciones y el Jupyter Notebooks de las funciones necesarias para la Api y ML.

[__03 - EDA:__ ](03 - EDA) Contiene el análisis exploratorio de los datos (EDA).

[__04 - ML:__ ](04 - ML) Contiene los Jupyter Notebooks con el Código completo de la recomendacion item_item.

[__05 - Agregados:__ ](04 - ML) Carpeta con imágenes y recursos utilizados en el desarrollo del proyecto.


# Autor:

## Barrionuevo Gabriela Soledad


