import csv
import gzip

with gzip.open('./data/psgs_w100.tsv.gz', 'rt') as f:
    tsv_reader = csv.reader(f, delimiter='\t')
    for row in tsv_reader:
        print(row)