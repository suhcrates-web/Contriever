# NSPLIT=128 #Must be larger than the number of processes used during training
# FILENAME=en_XX.txt
# FILENAME=data_test/data_test.txt
FILENAME=data_test/data_ko.txt

NSPLIT=$(wc -l < ${FILENAME})
NSPLIT=$((NSPLIT+1))
INFILE=./${FILENAME}
# TOKENIZER=bert-base-uncased
TOKENIZER=BM-K/KoSimCSE-roberta-multitask
#TOKENIZER=bert-base-multilingual-cased

SPLITDIR=./tmp-tokenization-kobert-${FILENAME}/
# SPLITDIR=./tmp-tokenization-${TOKENIZER}-${FILENAME}/
OUTDIR=./encoded-data/kobert/$(echo "$FILENAME" | cut -f 1 -d '.')
# OUTDIR=./encoded-data/${TOKENIZER}/$(echo "$FILENAME" | cut -f 1 -d '.')
NPROCESS=8

mkdir -p ${SPLITDIR}
echo ${INFILE}
split -a 3 -d -n l/${NSPLIT} ${INFILE} ${SPLITDIR}

# split : 쪼개라
# -a 3 :  파일 이름이 3자리.  000 001 ...
# -d  : suffix(파일 확장자?) 가 글자가 아닌 숫자.
# l/NSPLIT   :  line 베이스로 쪼개라.  NSPLIT 개로.
# INFILE : 쪼개는 대상
# SPLITDIR : 쪼개서 넣을 곳.