import numpy as np
from skimage.measure import label
from sklearn.metrics import confusion_matrix, f1_score
from glob import glob
from os.path import join, basename
from skimage.io import imread, imsave
import os
from skimage.segmentation import find_boundaries

def CheckOrCreate(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def add_contours(image, label, color="green"):
    """
    Adds contours to images.
    """
    
    mask = find_boundaries(label)
    res = np.array(image).copy()
    if color == "green":
        res[mask] = np.array([0, 255, 0])
    elif color == "yellow":
        res[mask] = np.array([255, 255, 0])
    return res

def f1__(G, S):
    s = S.copy()
    g = G.copy()
    g[g > 0] = 1
    s[s > 0] = 1
    return f1_score(g.flatten(), s.flatten(), pos_label=1)

def AJI_fast(G, S):
    """
    AJI as described in the paper, but a much faster implementation.
    """
    G = label(G, background=0)
    S = label(S, background=0)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0 
    USED = np.zeros(S.max())

    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()
        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]

            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union
        
        JI_ligne = map(h, range(1, S_max + 1))
        best_indice = np.argmax(JI_ligne) + 1
        C += cm[i, best_indice]
        U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
        USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U)  

def Generator_file(gt_path, pred_path, command):
    if command == "validation":
        files = glob(join(gt_path, "Slide_validation", "*.png"))
    elif command == "test":
        files = []
        for organ in ["testbreast", "testkidney", "testliver", "testprostate", \
                     "bladder", "colorectal", "stomach"]:
            files += glob(join(gt_path, "Slide_{}".format(organ), "*.png"))

    def pred_name(name):
        baseName = basename(name)
        return join(pred_path, baseName).replace(".png", "_mask.png")

    def mask_name(name):
        return name.replace("Slide", "GT")

    def return_organ(name):
        return name.split('/')[-2].split('_')[-1]

    def return_number(name):
        return int(name.split('_')[-1].split('.')[0])

    ll = [(f, mask_name(f), pred_name(f), return_organ(f), return_number(f)) for f in files]
    return ll


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN compute metrics')
    parser.add_argument('--GT', required=True,
                        metavar="/path/to/GT/",
                        help='path to GT')
    parser.add_argument('--PRED', required=True,
                        metavar="/path/to/Masks",
                        help="Path to Masks")
    parser.add_argument('--output_name', required=True,
                        metavar="output name",
                        help="Output name for the csv file")
    parser.add_argument("command",
                        metavar="<command>",
                        help="'test' or 'validation'")
    args = parser.parse_args()
    assert args.command in ["validation", "test"]
    gen_file = Generator_file(args.GT, args.PRED, args.command)
    
    aji_l = []
    f1_l = []

    fi = open(args.output_name, 'w')
    NAMES = ["NUMBER", "ORGAN", "F1", "AJI"]
    fi.write('{},{},{},{}\n'.format(*NAMES))
        
    for f, fmask, fpred, organ, number in gen_file:
        img = imread(f)
        mask = label(imread(fmask))
        pred = imread(fpred)
        aji = AJI_fast(mask, pred)
        f1 = f1__(mask, pred)
        fi.write('{},{},{},{}\n'.format(number, organ, f1, aji))
        aji_l.append(aji)
        f1_l.append(f1)
        if args.command == 'test':
            CheckOrCreate("./RCNN")
            organ_dir = os.path.join("./RCNN", organ)
            imsave(os.path.join(organ_dir, "rgb_{}.png").format(number), img)
            imsave(os.path.join(organ_dir, "pred_{}.png").format(number), pred)
            cont = add_contours(img, mask, "green")
            cont = add_contours(cont, pred, "yellow")
            imsave(os.path.join(organ_dir, "Contours_{}.png").format(number), cont)
            # TO DO check on test
    fi.write('{},{},{},{}\n'.format(0, "Mean", np.mean(f1_l), np.mean(aji_l)))
    fi.close()
    if args.command == 'validation':
        print np.mean(aji_l)
