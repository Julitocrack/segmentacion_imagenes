import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import eigsh
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import KMeans


# 1. CONSTRUCCIÓN DEL GRAFO 

def construir_grafo_rapido(imagen, sigma_color=40.0, sigma_espacial=15.0):
    """
    Construye la matriz de afinidad W usando operaciones vectorizadas.
    """
    alto, ancho, canales = imagen.shape
    
   
    # Pasamos solo el primer canal para definir la estructura de rejilla 2D.
   
    print("   -> Generando estructura de grafo (vectorizado)...")
    grafo = img_to_graph(imagen[:, :, 0])  
    
    grafo_coo = grafo.tocoo()
    src = grafo_coo.row
    dst = grafo_coo.col
    
    # Calcular Distancia de Color
    colores_flat = imagen.reshape(-1, canales)
    colores_src = colores_flat[src]
    colores_dst = colores_flat[dst]
    dist_color_sq = np.sum((colores_src - colores_dst) ** 2, axis=1)
    
    # Calcular Distancia Espacial
    y_coords, x_coords = np.indices((alto, ancho))
    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=1)
    coords_src = coords[src]
    coords_dst = coords[dst]
    dist_espacial_sq = np.sum((coords_src - coords_dst) ** 2, axis=1)
    
    # Calcular Pesos Finales (Kernel Gaussiano)
    print(f"   -> Calculando pesos (Sigma Color: {sigma_color}, Sigma Espacial: {sigma_espacial})...")
    pesos = np.exp(-dist_color_sq / (sigma_color**2)) * np.exp(-dist_espacial_sq / (sigma_espacial**2))
    
    grafo.data = pesos
    return grafo


# 2. SOLVER ROBUSTO (CORREGIDO PARA CONVERGENCIA)

def ncuts(W, num_segmentos):
    """
    Resuelve (D - W)x = lambda Dx usando Shift-Invert para estabilidad.
    """
    print("   -> Calculando Matriz Laplaciana...")
    
    # Grados con epsilon para seguridad
    grados = np.array(W.sum(axis=1)).flatten()
    grados[grados < 1e-10] = 1e-10 
    
    D_inv_sqrt = diags(1.0 / np.sqrt(grados))
    D = diags(grados)
    L = D - W
    
    # Laplaciano Normalizado: L_sym = D^-1/2 * L * D^-1/2
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    
    print(f"   -> Resolviendo {num_segmentos} vectores propios (Modo Shift-Invert)...")
    
   
    # sigma=0.001 + which='LM' 
    try:
        valores, vectores = eigsh(
            L_sym, 
            k=num_segmentos + 1, # Buscamos k+1 porque el primero se descarta
            sigma=0.001, 
            which='LM',
            maxiter=5000,
            ncv=20
        )
    except Exception as e:
        print(f"   ERROR: No convergió ({e}). Devolviendo ruido.")
        return np.random.rand(L_sym.shape[0], num_segmentos)

    # Recuperar solución original y descartar el primer vector (que es casi 0)
    # Devolvemos los vectores desde el índice 1 hasta el final
    vectores_embedding = D_inv_sqrt @ vectores[:, 1:]
    
    return vectores_embedding

