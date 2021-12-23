from typing import List
import collections
import numpy as np
import re
from pathlib import Path


def seg_case_direct(
      root: str,
      image_dir: str,
      label_dir: str,
      *args
      ) -> List[List[Path]]:

    """
    - root
        - image_dir
        - label_dir
    """

    annotations = list()
    c_label, c_img = Path(root) / label_dir, Path(root) / image_dir
    labels = sorted(_get_img_files(c_label))
    imgs = [next(c_img.glob(f"{label.stem}.*")) for label in labels]
    annotations = list(zip(imgs, labels))
    return annotations


def seg_case_first_targets(
      root: str,
      image_dir: str,
      label_dir: str,
      target_dirs: List[str]
      ) -> List[List[Path]]:

    """
    caseごとにフォルダが作成されている場合
    - root
        - case_name
            - image_dir
            - label_dir
    target_dirs: [case_name1, case_name2, ...]
    """
    annos = list()
    for case_name in target_dirs:
        case_dir = Path(root) / case_name
        annos += seg_case_direct(str(case_dir), image_dir, label_dir)
    return annos


def seg_case_first_groups(
      root: str,
      image_dir: str,
      label_dir: str,
      group_dirs: List[str]
      ) -> List[List[Path]]:

    """
    caseごとのフォルダをさらにグループでまとめている場合
    - root
        - group_name
            - case_name
                - image_dir
                - label_dir
    group_dirs: [group_name1, group_name2, ...]
    """

    annos = list()
    for group_name in group_dirs:
        group_dir = Path(root) / group_name
        for case_dir in group_dir.iterdir():
            annos += seg_case_direct(str(case_dir), image_dir, label_dir)
    return annos


def _get_img_files(p_dir: Path) -> List[Path]:
    ImageEx = "jpg|jpeg|png|gif|bmp"
    img_files = [
        p for p in p_dir.glob('*') if re.search(rf'.*\.({ImageEx})', str(p))
    ]
    return img_files


def crossval(annos: List[List[Path]], cv_fold, depth):
    train_annos = list()
    val_annos = list()

    if depth == 0:
        for cv_i in range(cv_fold):
            nb_val = len(annos) // cv_fold
            train_annos.append(annos[:nb_val*cv_i] + annos[nb_val*(cv_i+1):])
            val_annos.append(annos[nb_val*cv_i:nb_val*(cv_i+1)])
    else:
        group_names = list()
        for img_file, _ in annos:
            for _ in range(depth+1):
                img_file = img_file.parent
            group_names.append(img_file.stem)
        group_counter = collections.Counter(group_names)
        cross_val_dirs = _cross_val_split(group_counter, cv_fold)

        for cv_i in range(cv_fold):
            val_dirs = cross_val_dirs[cv_i]
            train_dirs = list()
            for val_i in range(cv_fold):
                if val_i == cv_i:
                    continue
                train_dirs += cross_val_dirs[val_i]
            train_annos.append([
                anno for anno, group_name in zip(annos, group_names)
                if group_name in train_dirs
            ])
            val_annos.append([
                anno for anno, group_name in zip(annos, group_names)
                if group_name in val_dirs
            ])
    return train_annos, val_annos


def _cross_val_split(g_count, k=5, n_iter=15, mix_step=5):
    data = [dict(case=case, count=count) for case, count in g_count.items()]
    # sort raughly
    data = sorted(data, reverse=True, key=lambda x: x["count"])
    cross = [[v] for v in data[:k]]
    for v in data[k:]:
        sum_list = [sum([case_dic["count"] for case_dic in c]) for c in cross]
        min_sum_id = np.argmin(sum_list)
        cross[min_sum_id].append(v)

    # n swap sort
    swapping = False
    for step in range(n_iter):
        for i in range(k-1):
            for j in range(k-i):
                n = cross[-j-1]
                m = cross[i]
                n_sum = sum([case_dic["count"] for case_dic in n])
                m_sum = sum([case_dic["count"] for case_dic in m])
                if n_sum > m_sum:
                    large = n
                    small = m
                    diff = n_sum - m_sum
                else:
                    small = n
                    large = m
                    diff = m_sum - n_sum

                for i_l, v_l in enumerate(large):
                    for i_s, v_s in enumerate(small):
                        no_change_swap = int((step + 1) % mix_step == 0)
                        l_s_diff = v_l["count"] - v_s["count"]
                        if 0 < l_s_diff < diff + no_change_swap:
                            large[i_l] = v_s
                            small[i_s] = v_l
                            swapping = True
                            break
                    if swapping:
                        swapping = False
                        break

    cross_val_dirs = list()
    for case_list in cross:
        val_dirs = [d["case"] for d in case_list]
        cross_val_dirs.append(val_dirs)

    return cross_val_dirs
