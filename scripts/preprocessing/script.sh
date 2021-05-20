#!/bin/bash

# 1. Get directory name
# echo $1
read -p "Enter directory name: " dirName

# 2. Change directory
cd "${PWD}/$dirName/images"

# 3. Resize images using ImageMagick
    # Install ImageMagick: 
        # sudo apt-get update
        # sudo apt-get install imagemagick
mkdir images_resized
mogrify -resize 1080x900! -path ./images_resized *.jpg

# 4. Split images to train and test directory
# Image files name is a numeric so we use random to choose files and store it in an array
# Reference: https://www.programmersought.com/article/87001143286/
# Control the length of the array, and the remainder for RANDOM
arrLen=200
# Define an array to store the resulting random number
typeset RAND                 
# Generate $arrLen random numbers   
for ((i=0;i<$arrLen;i++))    
do
    Rnum=$[RANDOM%250+1]    
    
    # Extract the length of the array
    Length=${#RAND[@]}

    # if statement idea: first assign the first value to the array, then compare the random number generated later with the elements of the array
    # If the same regenerates a random number, if it is different, save it in an array
    if [ $Length -eq 0 ];then    
            RAND[$i]=$Rnum
    else
        for ((j=0;j<$Length;j++))
        do  
            if [ $Rnum != ${RAND[$j]} ];then    
                continue
            else
                Rnum=$[RANDOM%$arrLen+1]    
                j=-1    
            fi  
        done
        RAND[$i]=$Rnum
    fi  
done

# Move 200 images to train directory
cd images_resized
for ((x=0;x<$arrLen;x++))
do
    mv "./${RAND[$x]}.jpg" "../../train"
done

# Move the remaining images to test directory
mv *.jpg ../../test

cd ..
rm -rf images_resized

cd ..
mv ./images ./images_origin

cd ..
mkdir ${dirName,,}

cp -r "$dirName/images_origin" ${dirName,,}
cp -r "$dirName/train" ${dirName,,}
cp -r "$dirName/test" ${dirName,,}

rm -rf $dirName