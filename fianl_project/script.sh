cd /Users/maxwill/Documents/ML2018fall/Stockdata/
for f in *.csv;do iconv -f utf-8 -t ascii -c "$f" > /Users/maxwill/Documents/ML2018fall/dataset/"$f";done
cd /Users/maxwill/Documents/ML2018fall/dataset/
find . -size 0 -delete
sleep 3s
ls /Users/maxwill/Documents/ML2018fall/dataset/ > /Users/maxwill/Documents/ML2018fall/dataset/list.txt
sed -i '$d' /Users/maxwill/Documents/ML2018fall/dataset/list.txt
