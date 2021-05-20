#!/bin/bash

read -p "Enter directory name: " dirName

cd $dirName

for ((x=1;x<=250;x++))
do
    mv "${dirName}_${x}.jpg" "${dirName}${x}.jpg"
    mv "${dirName}_${x}.xml" "${dirName}${x}.xml"
done