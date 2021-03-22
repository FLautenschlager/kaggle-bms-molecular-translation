#!/bin/bash

echo "install requirements"
pip install kaggle --upgrade
#pip install python-Levenshtein

echo ""
echo "input kaggle.json"
#read KAGGLE
#echo $KAGGLE > ~/.kaggle/kaggle.json
vim ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "Download dataset"
#mkdir /data
kaggle competitions download -c bms-molecular-translation -f data/bms.zip

7z x data/bms.zip -odata/

tmux new-session -s jupyter_session "jupyter lab ."
