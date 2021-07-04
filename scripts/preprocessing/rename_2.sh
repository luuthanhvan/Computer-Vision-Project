count=1153
for f in *.jpg; do
    mv -- "$f" "com_tam_${count}"
    count=$(($count+1))
done

