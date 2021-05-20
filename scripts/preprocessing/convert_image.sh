# Convert png into jpg using ImageMagick 

# 1. Install ImageMagick: if you didn't install ImageMagick, please remove the comment in line 4 and 5
# sudo apt-get update
# sudo apt-get install imagemagick

# $ for img in *.png; do convert "$img" "$img.jpg" ; done

# 2. Convert image
read -p "Enter directory name: " dirName
read -p "Enter the image format that you want to convert (ex: png): " ext
read -p "Enter the image format that you convert to (ex: jpg): " ext_new
read -p "Start number: " start
read -p "End number: " end

cd $dirName

for ((x=${start};x<=${end};x++))
do
    convert "${dirName}_${x}.${ext}" "${dirName}_${x}.${ext_new}"
    rm "${dirName}_${x}.${ext}"
done