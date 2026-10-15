"""Create linked demographic and biological copies of the gallstone dataset."""

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


SOURCE = Path(__file__).with_name("dataset_gallstone.xlsx")
DEMOGRAPHIC_OUTPUT = Path(__file__).with_name("gallstone_demographic_clinical.csv")
BIOLOGICAL_OUTPUT = Path(__file__).with_name("gallstone_biological_body_composition.csv")

DEMOGRAPHIC_COLUMNS = [
    "gallstone_status",
    "age",
    "gender",
    "comorbidity",
    "cad",
    "hypothyroidism",
    "hyperlipidemia",
    "dm",
]

BIOLOGICAL_COLUMNS = [
    "gallstone_status",
    "height",
    "weight",
    "bmi",
    "tbw",
    "ecw",
    "icw",
    "ecf_tbw",
    "tbfr_pct",
    "lm_pct",
    "protein_pct",
    "vfr",
    "bm",
    "mm",
    "obesity_pct",
    "tfc",
    "vfa",
    "vma_kg",
    "hfa",
    "glucose",
    "tc",
    "ldl",
    "hdl",
    "triglyceride",
    "ast",
    "alt",
    "alp",
    "creatinine",
    "gfr",
    "crp",
    "hgb",
    "vitamin_d",
]

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def column_number(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group()
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def read_workbook(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.findall(".//x:t", NS))
            for item in shared_root.findall("x:si", NS)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row_node in sheet.findall(".//x:sheetData/x:row", NS):
        values = {}
        for cell in row_node.findall("x:c", NS):
            value_node = cell.find("x:v", NS)
            value = "" if value_node is None else value_node.text or ""
            if cell.get("t") == "s" and value:
                value = shared[int(value)]
            values[column_number(cell.get("r"))] = value
        width = max(values, default=-1) + 1
        rows.append([values.get(index, "") for index in range(width)])
    return rows


def write_subset(path: Path, records: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=["record_id", *columns])
        writer.writeheader()
        for record_id, record in enumerate(records, start=1):
            writer.writerow(
                {"record_id": record_id, **{column: record[column] for column in columns}}
            )


def main() -> None:
    rows = read_workbook(SOURCE)
    headers = rows[0]
    records = [dict(zip(headers, row)) for row in rows[1:]]
    write_subset(DEMOGRAPHIC_OUTPUT, records, DEMOGRAPHIC_COLUMNS)
    write_subset(BIOLOGICAL_OUTPUT, records, BIOLOGICAL_COLUMNS)
    print(f"Created {DEMOGRAPHIC_OUTPUT.name}: {len(records)} records")
    print(f"Created {BIOLOGICAL_OUTPUT.name}: {len(records)} records")


if __name__ == "__main__":
    main()
