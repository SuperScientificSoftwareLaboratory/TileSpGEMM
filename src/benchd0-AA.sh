input="squaremtx-0519.csv"

{
  read
  i=1
  while IFS=',' read -r mid group name rows cols nonzeros
  do
    echo "$mid $group $name $rows $cols $nonzeros"
    echo "~/ssget/MM/$group/$name.mtx"
    ./test -d 0 ~/ssget/MM/$group/$name.mtx
    i=`expr $i + 1`
  done 
} < "$input"
