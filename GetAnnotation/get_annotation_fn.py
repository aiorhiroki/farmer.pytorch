import re
from typing import List
from pathlib import Path


def seg_case_direct(
      root: str,
      image_dir: str,
      label_dir: str
      ) -> List[List[Path]]:

    """
    - root
        - image_dir
        - label_dir
    """

    annotations = list()
    case_dir = Path(root)
    c_label, c_img = case_dir / label_dir, case_dir / image_dir
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
        annos += seg_case_direct(f"{root}/{case_name}", image_dir, label_dir)
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
