#AIgroKB/results/CSRncdKBC-transformer_epochs-2_seqlen-30_maxfeat-15000_batch-64_embdim-256_latent-2048_heads-8/predictions.csv | perl -pe 's/,\[start\] /\t/'| perl -pe 's/ \[end\]//' | sed '1!b;s/d,O/d\tO/' > objects_pairs.tsv
cut -d '\t' -f3,4 $1 | perl -pe 's/\[start\] //' | perl -pe 's/ \[end\],\[start\] /\t/'| perl -pe 's/ \[end\]//' | sed '1d' > $2
