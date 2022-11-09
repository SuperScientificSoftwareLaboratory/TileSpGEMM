input="../datasets/mat-142.csv"

{
  read
  i=1
  while IFS=',' read -r Group Name rows cols nonzeros
  do
    echo "$Group $Name $rows $cols $nonzeros"
#    echo "../datasets/142mat/$Name/$Name.mtx"
	echo "/142mat/$Group/$Name/$Name.mtx"
#    ./../TileSpGEMM/test -d 0 -aat 0 ../datasets/142mat/$Name/$Name.mtx
	./../TileSpGEMM/test -d 0 -aat 0 /home/ppopp22_test/MM/142mat/$Name.mtx
    i=`expr $i + 1`
  done 
} < "$input"
