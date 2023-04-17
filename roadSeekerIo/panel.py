from pathlib import Path

from roadSeekerIo.paths import RESULT_FILE_NAME


class Panel:

    def __init__(self, file_path: str, upper_edge: (int, int), lower_edge: (int, int), score: float,
                 panel_type: int = 1):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.upper_edge = upper_edge
        self.lower_edge = lower_edge
        self.panel_type = panel_type
        self.score = round(score, 2)
        self._save_to_file(RESULT_FILE_NAME)

    def _to_string(self) -> str:
        return "{};{};{};{};{};{};{}".format(
            self.file_name,
            self.upper_edge[0],
            self.upper_edge[1],
            self.lower_edge[0],
            self.lower_edge[1],
            self.panel_type,
            self.score
        ) + '\n'

    def _save_to_file(self, file_name: str):
        with open(file_name, 'a+') as f:
            f.write(self._to_string())
