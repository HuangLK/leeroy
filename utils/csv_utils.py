"""csv utils.
"""

import csv
from typing import Dict, List


def load_csv(data_file: str) -> List[Dict]:
    with open(data_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', escapechar='\\')
        headers = next(reader)
        for row in reader:
            yield { k: v for k, v in zip(headers, row) }

def dump_csv(out_file: str, rows: List, headers: List = None) -> None:
    assert len(rows) > 0, f'dumping an empty csv.'
    headers = rows[0].keys() if headers is None else headers
    with open(out_file, 'w', encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[k] for k in headers])
