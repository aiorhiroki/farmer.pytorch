import re
from typing import List
from pathlib import Path


def seg_case_first_targets(
      annotation_path: str,
      image_dir: str,
      label_dir: str,
      target_dirs: List[str]
      ) -> List[List[Path]]:

    """
    画像とアノテーションがセットになったリストを作る
    - annotation_path
        - case_name
            - image_dir
            - label_dir
    target_dirs: [case_name1, case_name2, ...]
    """
    annotations = list()
    for case_name in target_dirs:
        case_dir = Path(annotation_path) / case_name
        c_label = case_dir / label_dir
        c_img = case_dir / image_dir
        labels = sorted(_get_img_files(c_label))
        imgs = [next(c_img.glob(f"{label.stem}.*")) for label in labels]
        annotations += list(zip(imgs, labels))
    return annotations


def seg_case_first_groups(
      annotation_path: str,
      image_dir: str,
      label_dir: str,
      group_dirs: List[str]
      ) -> List[List[Path]]:

    """
    画像とアノテーションがセットになったリストを作る
    - annotation_path
        - group_name
            - case_name
                - image_dir
                - label_dir
    group_dirs: [group_name1, group_name2, ...]
    """

    annotations = list()
    for group_name in group_dirs:
        group_dir = Path(annotation_path) / group_name
        for case_dir in group_dir.iterdir():
            c_label = case_dir / label_dir
            c_img = case_dir / image_dir
            labels = sorted(_get_img_files(c_label))
            imgs = [next(c_img.glob(f"{label.stem}.*")) for label in labels]
            annotations += list(zip(imgs, labels))
    return annotations


def _get_img_files(p_dir: Path) -> List[Path]:
    ImageEx = "jpg|jpeg|png|gif|bmp"
    img_files = [
        p for p in p_dir.glob('*') if re.search(rf'.*\.({ImageEx})', str(p))
    ]
    return img_files
