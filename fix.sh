i=1
for file in imgs/*; do
    if [ $i -gt 10000 ]; then 
        rm $file
    else 
        mv $file $i.png
    fi
    i=$(($i + 1))
done