$ wc -l oie_gp_shuffled_pmid.tsv
39166 oie_gp_shuffled_pmid.tsv
$ head -n31332 oie_gp_shuffled_pmid.tsv > oie_gp_train.tsv
$ tail -n7834 oie_gp_shuffled_pmid.tsv > oie_gp_valid.tsv
