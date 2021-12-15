import numpy as np

def create_q(alpha, masks):
    q = alpha * (1 - masks)
    return q

def create_g(images, beta, N, H, W):
    g = np.zeros((N, N, H, W))
    
    for from_image_id in range(N):
        for to_image_id in range(N):
            for col_id in range(W-1):
                g[from_image_id, to_image_id, :, col_id] = beta * (np.linalg.norm((images[from_image_id, :, col_id] - images[to_image_id, :, col_id]), ord=1) + np.linalg.norm((images[from_image_id, :, col_id + 1] - images[to_image_id, :, col_id + 1]), ord=1))

    return g

def create_f(q, g, images, N, H, W):
    f = np.zeros((N, H, W))
    
    for i in range(H - 1, 0, -1):
        q_i = q[:, :, i]
        g_i = g[:, :, :, i]

        if i == H - 1:
            f[:, :, i] = np.min(q_i + g_i, axis=1)
        else:
            f[:, :, i] = np.min(q_i + g_i + f[:, :, i + 1], axis=1)
    return f
            
def calculate_k(q, g, f, images, H, W):
    k = np.zeros((H, W), dtype=int)
    
    for i in range(H):
        q_i = q[:, :, i]
        g_i = g[:, :, :, i]
        f_i = f[:, :, i]

        if i == 0:
            k[:, i] = np.argmin(q_i + f_i, axis=0)
        else:
            for row_id, row_mask in enumerate(k[:, i - 1]):
                k[row_id, i] = np.argmin(q_i[:, row_id] + g_i[row_mask, :, row_id] + f_i[:, row_id], axis=0)
    return k
    
def create_final_image(images, masks, alpha, beta):
    N = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    
    q = create_q(alpha, masks)
    g = create_g(images, beta, N, H, W)
    f = create_f(q, g, images, N, H, W)
    k = calculate_k(q, g, f, images, H, W)
    
    final_image = np.zeros_like(images[0])

    for row_id in range(H):
        for col_id in range(W):
            image_id = k[row_id, col_id]
            final_image[row_id, col_id, :] = images[image_id, row_id, col_id, :]

    return final_image