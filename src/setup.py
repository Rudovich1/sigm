import shutil
from pathlib import Path


def setup():
    input_path = Path("data").absolute()
    output_path = Path("out").absolute()

    try:
        shutil.rmtree(input_path)
        input_path.mkdir(parents=True)
        print(f"Всё содержимое папки '{input_path}' успешно удалено")
    except FileNotFoundError:
        input_path.mkdir(parents=True)
        print(f"Папка '{input_path}' успешно создана")
    except PermissionError:
        print(f"Нет прав для удаления папки '{input_path}'")
    except Exception as e:
        print(f"Ошибка при удалении папки: {e}")

    try:
        shutil.rmtree(output_path)
        print(f"Папка '{output_path}' и всё её содержимое успешно удалены")
    except FileNotFoundError:
        pass
    except PermissionError:
        print(f"Нет прав для удаления папки '{output_path}'")
    except Exception as e:
        print(f"Ошибка при удалении папки: {e}")


if __name__ == "__main__":
    setup()
