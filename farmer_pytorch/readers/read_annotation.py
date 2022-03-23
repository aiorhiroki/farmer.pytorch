from typing import List
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


def seg_cases(
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


def seg_case_groups(
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
