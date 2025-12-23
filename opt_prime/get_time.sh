#!/bin/bash

LOGFILE=$1

echo $LOGFILE
# Extract values and calculate average and standard deviation
result=$(grep '^| epoch   1 |' "$LOGFILE" \
    | grep -E '\s(5[1-9]|60)/' \
    | grep -oE 'ms/batch[[:space:]]+[0-9.]+' \
    | awk '{
        values[NR] = $2
        sum += $2
        n++
    } 
    END {
        if (n > 0) {
            avg = sum / n
            # Calculate variance
            variance_sum = 0
            for (i = 1; i <= n; i++) {
                variance_sum += (values[i] - avg) * (values[i] - avg)
            }
            variance = variance_sum / n
            stddev = sqrt(variance)
            printf "%.3f %.3f", avg, stddev
        } else {
            print "No matching lines found."
        }
    }')

# Extract average and standard deviation
avg=$(echo $result | awk '{print $1}')
stddev=$(echo $result | awk '{print $2}')

echo "===> Average ms/batch (51~60): $avg"
echo "===> Standard deviation ms/batch (51~60): $stddev"
echo "===> Detailed lines:"
grep '^| epoch   1 |' "$LOGFILE" | grep -E '\s(5[1-9]|60)/' 
# grep '^\[RANK[[:space:]]*7\]' "$LOGFILE" | grep -E '\s(5[1-9]|60)/'
echo $avg > "${LOGFILE%.*}"_avg.txt