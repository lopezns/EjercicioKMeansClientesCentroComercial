import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Mall_Customers.csv"
IMG_DIR = BASE_DIR / "static" / "img"

VARIABLES_CLUSTERING = ["Annual Income (k$)", "Spending Score (1-100)"]
VARIABLES_TABLA = ["CustomerID", "Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]


def ObtenerDatos():
    datos = pd.read_csv(DATASET_PATH)
    datos = datos.drop_duplicates()

    for columna in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos = datos.dropna(subset=VARIABLES_CLUSTERING)
    return datos[VARIABLES_TABLA]


def ObtenerInformacionDataset():
    datos = ObtenerDatos()
    descripcion = datos[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].describe().round(2)

    return {
        "cantidad_registros": len(datos),
        "cantidad_columnas": len(datos.columns),
        "columnas": list(datos.columns),
        "nulos": datos.isna().sum().to_dict(),
        "primeros_registros": datos.head(10).to_dict(orient="records"),
        "descripcion": descripcion.reset_index().to_dict(orient="records"),
        "variables": [
            {
                "nombre": "Annual Income (k$)",
                "descripcion": "Ingreso anual del cliente expresado en miles de dólares.",
            },
            {
                "nombre": "Spending Score (1-100)",
                "descripcion": "Puntaje de gasto asignado al cliente según su comportamiento de compra.",
            },
        ],
    }


def CalcularMetodoCodo(max_k=10):
    datos = ObtenerDatos()
    x = datos[VARIABLES_CLUSTERING]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    inercias = []
    valores_k = list(range(1, max_k + 1))

    for k in valores_k:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(x_scaled)
        inercias.append(round(float(modelo.inertia_), 2))

    return valores_k, inercias


def _crear_grafica_codo(valores_k, inercias):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ruta = IMG_DIR / "metodo_codo.png"

    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, inercias, marker="o", color="#1f6f8b", linewidth=2)
    plt.title("Metodo del codo para seleccionar K")
    plt.xlabel("Numero de clusters (K)")
    plt.ylabel("Inercia")
    plt.xticks(valores_k)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ruta, dpi=140)
    plt.close()

    return "img/metodo_codo.png"


def _crear_grafica_clusters(resultados, centroides, nclusters):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ruta = IMG_DIR / f"clusters_k{nclusters}.png"

    plt.figure(figsize=(8, 5.6))
    dispersion = plt.scatter(
        resultados["Annual Income (k$)"],
        resultados["Spending Score (1-100)"],
        c=resultados["Cluster"],
        cmap="viridis",
        s=55,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    plt.scatter(
        centroides["Annual Income (k$)"],
        centroides["Spending Score (1-100)"],
        c="red",
        marker="X",
        s=230,
        label="Centroides",
        edgecolor="black",
        linewidth=0.8,
    )
    plt.title(f"Segmentacion de clientes con K-Means (K={nclusters})")
    plt.xlabel("Ingreso anual (k$)")
    plt.ylabel("Puntaje de gasto (1-100)")
    plt.colorbar(dispersion, label="Cluster")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ruta, dpi=140)
    plt.close()

    return f"img/clusters_k{nclusters}.png"


def _interpretar_cluster(ingreso, gasto):
    if ingreso >= 70 and gasto >= 60:
        return "Clientes de alto valor: ingresos altos y alto nivel de gasto."
    if ingreso >= 70 and gasto < 45:
        return "Clientes con potencial: ingresos altos, pero bajo gasto en el centro comercial."
    if ingreso < 45 and gasto >= 60:
        return "Clientes compradores frecuentes: ingresos bajos o medios y gasto alto."
    if ingreso < 45 and gasto < 45:
        return "Clientes conservadores: ingresos bajos y bajo nivel de gasto."
    return "Clientes promedio: comportamiento equilibrado de ingresos y gasto."


def RealizarClustering(nclusters=5):
    nclusters = max(2, min(int(nclusters), 10))
    datos = ObtenerDatos()

    x = datos[VARIABLES_CLUSTERING]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    modelo = KMeans(n_clusters=nclusters, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(x_scaled) + 1

    resultados = datos.copy()
    resultados["Cluster"] = etiquetas

    centroides_originales = scaler.inverse_transform(modelo.cluster_centers_)
    centroides = pd.DataFrame(centroides_originales, columns=VARIABLES_CLUSTERING)
    centroides["Cluster"] = range(1, nclusters + 1)
    centroides = centroides[["Cluster"] + VARIABLES_CLUSTERING].round(2)

    resumen = (
        resultados.groupby("Cluster")
        .agg(
            clientes=("CustomerID", "count"),
            edad_promedio=("Age", "mean"),
            ingreso_promedio=("Annual Income (k$)", "mean"),
            gasto_promedio=("Spending Score (1-100)", "mean"),
        )
        .round(2)
        .reset_index()
    )
    resumen["interpretacion"] = resumen.apply(
        lambda fila: _interpretar_cluster(fila["ingreso_promedio"], fila["gasto_promedio"]),
        axis=1,
    )

    valores_k, inercias = CalcularMetodoCodo()
    grafica_codo = _crear_grafica_codo(valores_k, inercias)
    grafica_clusters = _crear_grafica_clusters(resultados, centroides, nclusters)

    return {
        "resultados": resultados.to_dict(orient="records"),
        "resumenClusters": resumen.to_dict(orient="records"),
        "Centroides": centroides.to_dict(orient="records"),
        "valores_k": valores_k,
        "inercias": inercias,
        "grafica_codo": grafica_codo,
        "grafica_clusters": grafica_clusters,
        "variables_usadas": VARIABLES_CLUSTERING,
        "k_seleccionado": nclusters,
    }
