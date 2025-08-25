from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from matplotlib.figure import Figure

from calc import LOGGER


@dataclass
class SigmData:
    L: float
    x0: float
    k: float
    b: float
    max_der_x: float | None
    max_der_y: float | None
    min_sec_der_x: float
    min_sec_der_y: float
    message: str | None


class FileUnits:
    def __init__(self, file_name: str, cycles: list[int]):
        self._file_name = file_name
        self._cycles: list[int] = cycles
        self._data: dict[str, list[float]] = {}
        self._result: dict[str, tuple[SigmData | None, Figure | None] | None] = {}

    @property
    def file_name(self) -> str:
        return self._file_name

    @property
    def cycles(self) -> list[int]:
        return self._cycles

    def __len__(self):
        return self._data.__len__()

    def __setitem__(self, key: str, value: list[float]) -> None:
        if key not in self._data:
            self._data[key] = value
            self._result[key] = None
        else:
            counter = 1
            while f"{key}{counter}" in self._data:
                counter += 1
            self._data[f"{key}{counter}"] = value
            self._data[f"{key}{counter}"] = None

    def __getitem__(self, key) -> list[float]:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def set_result(self, key: str, sigm_data: SigmData, fig: Figure | None) -> None:
        self._result[key] = (sigm_data, fig)

    def get_result(self, key: str) -> tuple[SigmData | None, Figure | None]:
        result = self._result[key]
        if result is None:
            return (None, None)
        return result


class FileIO:
    def __init__(
        self,
        input_dir: Path = Path("data").absolute(),
        output_dir: Path = Path("out").absolute(),
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def read_data(self) -> Generator[FileUnits, Any, None]:
        for file in self.input_dir.iterdir():
            if file.is_file():
                try:
                    df = pd.read_excel(file, sheet_name=0, index_col="Cycle")
                except pd.errors.ParserError:
                    LOGGER.warning(f'"{file.stem}": file format is not excel. [skip]')
                except Exception as e:
                    LOGGER.warning(f"{file.stem}: {e}. [skip]")
                else:
                    input_file = FileUnits(
                        file_name=file.stem, cycles=df.index.astype(int).to_list()
                    )
                    for column in df.columns:
                        try:
                            input_file[column] = (
                                df[column].astype(dtype=float).to_list()
                            )
                        except Exception as e:
                            LOGGER.warning(f'"{file.stem}"."{column}": {e}. [skip]')
                    yield input_file

    def save_data(self, file_units: FileUnits) -> None:
        try:
            cur_dir = self.output_dir / file_units.file_name
            cur_dir.mkdir(exist_ok=True, parents=True)
            (cur_dir / "figs").mkdir(exist_ok=True, parents=True)
            result_data = []
            for key in file_units:
                sigm_data, fig = file_units.get_result(key)
                result_data.append(
                    [
                        key,
                        *(
                            astuple(sigm_data)
                            if sigm_data is not None
                            else [None] * len(fields(SigmData))
                        ),
                    ]
                )
                if fig is not None:
                    fig.savefig(cur_dir / "figs" / f"{key}.png")
            pd.DataFrame(
                columns=["name"] + [f.name for f in fields(SigmData)],
                data=result_data,
                index=None,
            ).to_excel(cur_dir / "result.xlsx", index=False, float_format="%.5f")
        except Exception as e:
            LOGGER.warning(f'"{file_units.file_name}": {e}. [skip save]')
