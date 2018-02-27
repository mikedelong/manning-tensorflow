import csv
import time


def read(filename, date_idx, date_parse, year, bucket=7):
    days_in_year = 365
    result = {}
    for period in range(0, int(days_in_year / bucket)):
        result[period] = 0
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        count = 0
        for row in csvreader:
            if row[date_idx] == '':
                continue
            t = time.strptime(row[date_idx], date_parse)
            if t.tm_year == year and t.tm_yday < (days_in_year - 1):
                result[int(t.tm_yday / bucket)] += 1
            count += 1
            if count % 1000 == 0 and count > 0:
                print(count)

    return result


freq = read('./input/311_Call_Center_Tracking_Data__Archived_.csv', 0, '%m/%d/%Y', 2014)
