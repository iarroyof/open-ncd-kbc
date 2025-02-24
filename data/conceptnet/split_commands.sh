$ output_file="output.tsv"
$ # Process the input file
$ awk -F'\t' '{
>     new_col = $3 " " $2 " " $4
>     print $1 "\t" new_col "\t" $2 "\t" $3 "\t" $4 "\t" $5
> }' OFS='\t' concepnet_shuffled.csv  > "$output_file"
$ wc -l $output_file
600000 output.tsv
$ head -n420000 $output_file > conceptnet_train.tsv
$ tail -n180000 $output_file > conceptnet_tv.tsv
$ head -n120000 conceptnet_tv.tsv > conceptnet_test.tsv
$ tail -n60000 conceptnet_tv.tsv > conceptnet_valid.tsv
