# Модуль содержит функции для работы с текстами


from openpyxl import load_workbook
from typing import List, Tuple
import xlrd, re, json
from typing import Any


def load_xlsx(input_file):
    # Загружаем книгу
    wb = load_workbook(input_file, data_only=True)   # data_only=True — берёт значения, а не формулы
    ws = wb.active  # берём первый (активный) лист

#    print(f"[INFO] Лист: {ws.title} | Строк: {ws.max_row} | Колонок: {ws.max_column}")

    lines = []

    # Проходим по всем строкам
    for row in ws.iter_rows(values_only=True):
        # Преобразуем каждую ячейку в строку
        row_values = []
        for cell in row:
            if cell is None:
                row_values.append("")
            else:
                val = str(cell)
                row_values.append(val)
        
        # Соединяем табуляцией
        line = "\t".join(row_values)
        lines.append(line)

    return "\n".join(lines)

def load_xls(input_file):
    # Открываем книгу
    wb = xlrd.open_workbook(input_file, on_demand=True)
    ws = wb.sheet_by_index(0)  # берём первый лист
    
    print(f"[INFO] Лист: {ws.name} | Строк: {ws.nrows} | Колонок: {ws.ncols}")

    lines = []

    # Проходим по всем строкам
    for row_idx in range(ws.nrows):
        row_values = []
        
        for col_idx in range(ws.ncols):
            cell = ws.cell_value(row_idx, col_idx)
            
            if cell is None or cell == "":
                row_values.append("")
            else:
                # xlrd возвращает разные типы: float, int, str, datetime и т.д.
                val = str(cell)
                row_values.append(val)
        
        line = "\t".join(row_values)
        lines.append(line)

    return "\n".join(lines)


def load_text(file_name: str):
    """
    Загружает текст из файла 
    """
    try:
        with open(file_name, "r", encoding="utf-8-sig") as f:
            text = f.read()
#            print(f"[OK] Файл успешно загружен: {file_name} ({len(self.text):,} символов)")
            return text
    except FileNotFoundError:
#        print(f"[ERROR] Файл не найден: {file_name}")
        return ""
    except Exception as e:
#        print(f"[ERROR] Ошибка при чтении файла {file_name}: {e}")
        return text
    


def load(file_name: str):
    """
    Загружает файл в разных форматах
    """

    #match = re.search(r'\.([a-zA-Z0-9]+)$', file_name.lower())
    match = re.search(r'(\.[a-z0-9]+)$', file_name)
    
    ext = match.group(1) if match else ""


    if ext == '.xlsx':
        return load_xlsx(file_name)
    elif ext == '.xls':
        return load_xls(file_name)
    else:
        return load_text(file_name)
