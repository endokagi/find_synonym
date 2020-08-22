import csv
# Use for get any rows csv by all column
def get_data(path):
    rows = []
    with open(path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            rows.append(row)
    return rows
# Use for get any rows csv by only one column.
def get_data_by_one_column(path,column_name):
    rows = []
    with open(path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data_at_row = row[column_name]
            rows.append(data_at_row)
    return rows
# Use for get any rows csv by only one column.
def get_data_by_multi_columns(path,columns_name):
    rows = []
    with open(path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data_at_row = {}
            for column in columns_name:
                data_at_row.update({column:row[column]})
            rows.append(data_at_row)
    return rows
# Use for write a csv file by only one column.
def write_data_by_one_column(path,column_name,data):
    with open(path, mode='w', newline='', encoding='utf-8') as writefile:
        fieldnames = [column_name]
        writer = csv.DictWriter(writefile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data :
            writer.writerow({column_name:row})
# Use for write a csv file by more than one column.
def write_data_by_columns(path,fieldnames,data):
    with open(path, mode='w', newline='', encoding='utf-8') as writefile:
        writer = csv.DictWriter(writefile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data :
            writer.writerow(row)

def main():
    #test functions
    get_data("./max_range_3/tfidf_p75_t7.csv")

if __name__ == "__main__":
    main()