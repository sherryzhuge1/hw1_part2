#!/bin/bash

mkdir -p ~/.kaggle/
echo '{"username":"sherryzhuge","key":"b7f2dff6b3efb31e22561699baaf0c72"}' > ~/.kaggle/kaggle.json

chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c 11785-spring-25-hw-1-p-2
unzip -qn 11785-spring-25-hw-1-p-2.zip

python main.py

# kaggle competitions submit -c 11785-spring-25-hw-1-p-2 -f ./submission.csv -m "Test Submission"