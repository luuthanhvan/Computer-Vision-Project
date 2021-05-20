count=1153
for f in *.jpg; do
    mv -- "$f" "com_tam_${count}.jpg"
    count=$(($count+1))
done

