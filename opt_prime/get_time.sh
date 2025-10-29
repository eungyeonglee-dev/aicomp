#!/bin/bash

LOGFILE=$1

echo $LOGFILE
# avg=$(grep '^\[rank' "$LOGFILE" \
# avg=$(grep '^\[RANK[[:space:]]*7\]' "$LOGFILE" \
avg=$(grep '^| epoch   1 |' "$LOGFILE" \
    | grep -E '\s(5[1-9]|60)/' \
    | grep -oE 'ms/batch[[:space:]]+[0-9.]+' \
    | awk '{sum += $2; n++} END {if (n > 0) printf "%.3f\n", sum / n; else print "No matching lines found."}')
echo "===> Average ms/batch (51~60): $avg"
echo "===> Detailed lines:"
grep '^| epoch   1 |' "$LOGFILE" | grep -E '\s(5[1-9]|60)/' 
# grep '^\[RANK[[:space:]]*7\]' "$LOGFILE" | grep -E '\s(5[1-9]|60)/'
echo $avg > "${LOGFILE%.*}"_avg.txt