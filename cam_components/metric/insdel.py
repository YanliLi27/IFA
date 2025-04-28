import numpy as np
import torch
import numpy as np
import torch.nn.functional as F


def get_sorted_windows(saliency_map, window_size):
    """
    以窗口为单位排序显著图（降序），适用于单张图 [H, W]
    """
    H, W = saliency_map.shape
    h_steps = H // window_size
    w_steps = W // window_size

    window_scores = []
    windows = []

    for i in range(h_steps):
        for j in range(w_steps):
            h0, h1 = i * window_size, (i + 1) * window_size  # 
            w0, w1 = j * window_size, (j + 1) * window_size
            score = np.sum(saliency_map[h0:h1, w0:w1])
            window_scores.append(score)
            windows.append((h0, h1, w0, w1))

    sorted_windows = [w for _, w in sorted(zip(window_scores, windows), reverse=True)]
    return sorted_windows

def compute_auc(scores):
    x = np.linspace(0, 1, len(scores))
    return np.trapz(scores, x)

def evaluate_insertion_deletion_auc(
    images, cams, gt, model_fn, window_size=8, steps=None, mode='both', target_class=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Args:
        images: np.ndarray [B, C, H, W]
        cams: np.ndarray [B, 1, H, W]
        gt: np.array [B]
        model_fn: Callable, takes [B, C, H, W] and returns [B] or [B, num_classes]
        window_size: int, size of square window (e.g., 10)
        steps: int, optional, number of steps
        mode: 'insertion' | 'deletion' | 'both'
        target_class: int | None, if model returns logits for multiple classes

    Returns:
        dict with 'insertion_auc' and/or 'deletion_auc', each [B]
    """
    B, C, H, W = images.shape
    insertion_aucs = []
    deletion_aucs = []

    for b in range(B):
        image = images[b]  # [C,H,W]
        cam = cams[b, 0]  # [H,W]
        gt_class = gt[b]  # int

        sorted_windows = get_sorted_windows(cam, window_size)
        if steps is None:
            steps = len(sorted_windows)

        # Prepare insertion & deletion image sequences
        base_insert = np.zeros_like(image)
        base_delete = image.copy()

        insert_images = []
        delete_images = []

        cur_insert = base_insert.copy()
        cur_delete = base_delete.copy()

        for step in range(steps):
            h0, h1, w0, w1 = sorted_windows[step]
            cur_insert[:, h0:h1, w0:w1] = image[:, h0:h1, w0:w1]
            cur_delete[:, h0:h1, w0:w1] = 0

            if mode in ['insertion', 'both']:
                insert_images.append(cur_insert.copy())
            if mode in ['deletion', 'both']:
                delete_images.append(cur_delete.copy())


        if mode in ['insertion', 'both']:
            batch_insert = np.stack(insert_images)  # [S, C, H, W]
            preds = model_fn(torch.from_numpy(batch_insert).to(device))  # [S] or [S, logits(num_classes)]
            preds = F.softmax(preds, dim=1)  # [S, confidence(num_classes)]
            if target_class is not None:
                preds = preds[:, target_class]
            else:
                preds = preds[:, gt_class]
            insertion_aucs.append(compute_auc(preds.cpu().detach().numpy()))

        if mode in ['deletion', 'both']:
            batch_delete = np.stack(delete_images)  # [S, C, H, W]
            preds = model_fn(torch.from_numpy(batch_delete).to(device)) # [S] or [S, logits(num_classes)]
            preds = F.softmax(preds, dim=1)  # [S, confidence(num_classes)]
            if target_class is not None:
                preds = preds[:, target_class]
            else:
                preds = preds[:, gt_class]
            deletion_aucs.append(compute_auc(preds.cpu().detach().numpy()))

    return insertion_aucs, deletion_aucs  # list

# for test only
def generate_demo_batch(batch_size=4, height=64, width=64):
    images = np.random.rand(batch_size, 3, height, width).astype(np.float32)  # RGB
    cams = np.zeros((batch_size, 1, height, width), dtype=np.float32)

    for i in range(batch_size):
        # 中心区域是热点（高显著性）
        h_center, w_center = height // 2, width // 2
        h_radius, w_radius = 10, 10
        cams[i, 0, h_center - h_radius:h_center + h_radius,
                  w_center - w_radius:w_center + w_radius] = 1.0

        # 加点噪声模拟真实CAM
        cams[i] += 0.1 * np.random.randn(1, height, width)
        cams[i] = np.clip(cams[i], 0, 1)  # 保证在[0, 1]

    return images, cams

# for test only
class model_fn1:
    def __init__(self):
        pass
    def __call__(self, images):
        return images.mean(axis=(1, 2, 3))  # 返回每张图的均值


if __name__=="__main__":
    images, cams = generate_demo_batch(batch_size=2, height=256, width=256)
    print(images.shape)
    # [B, C, H, W]
    model_fn = model_fn1()

    insertion_list, deletion_list = evaluate_insertion_deletion_auc(images, cams, model_fn, window_size=8, target_class=None)

    print("Insertion AUCs:", insertion_list)
    print("Deletion AUCs:", deletion_list)