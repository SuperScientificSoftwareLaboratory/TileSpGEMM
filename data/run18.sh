input="../datasets/mtx18.csv"

{
  read
  i=1
  while IFS=',' read -r Group Name rows cols nonzeros
  do
    echo "$Group $Name $rows $cols $nonzeros"
#    echo "../datasets/18mat/$Name/$Name.mtx"
    echo "/18mat/$Group/$Name/$Name.mtx"
#    ./../TileSpGEMM/test -d 0 -aat 0 ../datasets/18mat/$Name/$Name.mtx
    ./../TileSpGEMM/test -d 0 -aat 0 /home/ppopp22_test/MM/18mat/$Name.mtx
    i=`expr $i + 1`
  done 
} < "$input"
