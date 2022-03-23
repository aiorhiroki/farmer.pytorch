from typing import List
import re
from pathlib import Path


class CaseDirect:
    def __init__(self, image_dir: str, label_dir: str):
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __call__(self, root: str, *args) -> List[List[Path]]:
        return self.read_case_direct(root)

    def read_case_direct(self, root: str, *args) -> List[List[Path]]:
        """
        - root
            - image_dir
            - label_dir
        """
        annotations = list()
        c_label = Path(root) / self.label_dir
        c_img = Path(root) / self.image_dir
        labels = sorted(self._get_img_files(c_label))
        imgs = [next(c_img.glob(f"{label.stem}.*")) for label in labels]
        annotations = list(zip(imgs, labels))
        return annotations

    def _get_img_files(self, p_dir: Path) -> List[Path]:
        ImageEx = "jpg|jpeg|png|gif|bmp"
        img_files = [
            p for p in p_dir.glob('*')
            if re.search(rf'.*\.({ImageEx})', str(p))
        ]
        return img_files


class Cases(CaseDirect):
    def __init__(self, image_dir: str, label_dir: str):
        super().__init__(image_dir, label_dir)

    def __call__(self, root: str, target_dirs: List[str]) -> List[List[Path]]:
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
            annos += self.read_case_direct(
                str(case_dir), self.image_dir, self.label_dir)
        return annos


class CaseGroups(CaseDirect):
    def __init__(self, image_dir: str, label_dir: str):
        super().__init__(image_dir, label_dir)

    def __call__(self, root: str, group_dirs: List[str]) -> List[List[Path]]:
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
                annos += self.read_case_direct(
                    str(case_dir), self.image_dir, self.label_dir)
        return annos
