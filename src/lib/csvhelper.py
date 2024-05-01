import csv


class CsvHelper:

    def write_data_to_csv(self, filename, fieldnames, rows):
        with open(filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
