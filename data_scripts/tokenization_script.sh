## 이 파일은 일단 text파일을 쪼개서 'SPLITDIR' 에 담아놓고, 그걸 한개씩 preprocess.py 에 넣는 식으로 작업함.
## SPLITDIR 파일 개수는 128개로 설정돼있었으나, text 파일 line 수대로 만들도록 재설정함


NSPLIT=128 #Must be larger than the number of processes used during training
# FILENAME=en_XX.txt
# FILENAME=data_test/data_test.txt
FILENAME=data_test/data_ko.txt

##내가넣음.  
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

BASENAME=$(basename "$FILENAME" .txt)  # 내가넣음
# split -a 3 -d -n l/${NSPLIT} ${INFILE} ${SPLITDIR}
split -a 3 -d -n l/${NSPLIT} ${INFILE} ${SPLITDIR}/${BASENAME}  # 내가넣음

pids=()

for ((i=0;i<$NSPLIT;i++)); do
    num=$(printf "%03d\n" $i);
    # FILE=${SPLITDIR}${num};
    FILE=${SPLITDIR}${BASENAME}${num}; # 내가 넣음.

    #we used --normalize_text as an additional option for mContriever
    python3 preprocess.py --tokenizer ${TOKENIZER} --datapath ${FILE} --outdir ${OUTDIR} &
    pids+=($!);
    if (( $i % $NPROCESS == 0 ))
    then
        for pid in ${pids[@]}; do
            wait $pid
        done
    fi
done

for pid in ${pids[@]}; do
    wait $pid
done

echo ${SPLITDIR}

rm -r ${SPLITDIR}
