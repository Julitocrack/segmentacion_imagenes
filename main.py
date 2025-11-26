from algoritmo.ncuts import construir_grafo_rapido, ncuts
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def main():
    # RUTA DE TU IMAGEN
    ruta = 'sources/Chivas vs Gallos_247.jpg' 
    img = cv2.imread(ruta)
    
    if img is None:
        print(f"Error: No se encuentra la imagen en {ruta}")
        return

    # Aumentamos un poco la resolución para mejor detalle
    tano_target = (80, 80) 
    img_resized = cv2.resize(img, tano_target)
    
    # Convertir a LAB para mejor percepción de color
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    print("1. Construyendo Grafo...")
    # Estos parámetros son más tolerantes para evitar "ruido"
    W = construir_grafo_rapido(img_lab, sigma_color=40.0, sigma_espacial=15.0)
    
    # --- CONFIGURACIÓN DE SEGMENTOS ---
    # Usamos 4 para separar: Jugador A, Jugador B, Pasto, Fondo
    NUM_CLUSTERS = 4
    
    print("2. Resolviendo NCuts...")
    vectores_embedding = ncuts(W, NUM_CLUSTERS)
    
    print(f"3. Segmentando en {NUM_CLUSTERS} clusters...")
    # KMeans usa los vectores propios para agrupar los píxeles
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42).fit(vectores_embedding)
    labels = kmeans.labels_.reshape(tano_target[1], tano_target[0])
    
    # --- VISUALIZACIÓN ---
    plt.figure(figsize=(14, 5))
    
    # 1. Original
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("Imagen Original")
    plt.axis('off')
    
    # 2. Vector Fiedler (El más importante)
    plt.subplot(1, 3, 2)
    # Mostramos solo el primer vector informativo
    plt.imshow(vectores_embedding[:, 0].reshape(tano_target[1], tano_target[0]), cmap='jet')
    plt.title("Vector Fiedler (Estructura detectada)")
    plt.axis('off')
    
    # 3. Segmentación Final
    plt.subplot(1, 3, 3)
    plt.imshow(labels, cmap='tab10') # 'tab10' tiene colores distintivos
    plt.title(f"Segmentación ({NUM_CLUSTERS} regiones)")
    plt.axis('off')
    
    plt.tight_layout()
    print("Mostrando resultados...")
    plt.show()

if __name__ == "__main__":
    main()