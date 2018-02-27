import csv
import logging
import time

start_time = time.time()


def read(filename, date_idx, date_parse, year, bucket=7):
    days_in_year = 365
    result = {}
    for period in range(0, int(days_in_year / bucket)):
        result[period] = 0
    with open(filename, 'r') as input_file:
        reader = csv.reader(input_file)
        next(reader)
        count = 0
        for row in reader:
            if row[date_idx] == '':
                continue
            date = time.strptime(row[date_idx], date_parse)
            if date.tm_year == year and date.tm_yday < (days_in_year - 1):
                result[int(date.tm_yday / bucket)] += 1
            count += 1
            if count % 10000 == 0 and count > 0:
                logger.debug(count)

    return result


formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

frequencies = read('./input/311_Call_Center_Tracking_Data__Archived_.csv', 0, '%m/%d/%Y', 2014)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
