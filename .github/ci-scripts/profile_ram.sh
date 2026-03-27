#!/bin/bash
LOG_FILE=$1
if [ -z "$LOG_FILE" ]; then
    echo "Usage: $0 <log_file>"
    exit 1
fi

echo "timestamp,mem_used_mb" > "$LOG_FILE"
while true; do
  # Get used memory in MB. Using /proc/meminfo or free -m
  # 'free -m' is standard on Ubuntu
  MEM=$(free -m | awk '/^Mem:/{print $3}')
  TIMESTAMP=$(date +%s)
  echo "$TIMESTAMP,$MEM" >> "$LOG_FILE"
  sleep 5
done
