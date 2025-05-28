#!/bin/bash
# python3 sts.py --input ../AIgroKB/results_final/attentionGRU-CSOIEGP_index-023_epochs-100_seqlen-10_maxfeat-15000_batch-128_modeldim-256_units-2048/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel local &
# python3 sts.py --input ../AIgroKB/results_final/attentionGRU-CSOIEGP_index-023_epochs-100_seqlen-10_maxfeat-15000_batch-128_modeldim-256_units-2048/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel local &
# python3 sts.py --input ../AIgroKB/results_final/attentionGRU-CSOIEGP_index-034_epochs-100_seqlen-30_maxfeat-15000_batch-32_modeldim-1024_units-1024/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel local &
# python3 sts.py --input ../AIgroKB/results_final/attentionGRU-CSOIEGP_index-034_epochs-100_seqlen-30_maxfeat-15000_batch-32_modeldim-1024_units-1024/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel local &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel local &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel local &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel local &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel local &

#==========================================================

# # 'Conceptnet+OIE-GP+NCD', 
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &


# # 'Conceptnet+NCD', 
# python3 sts.py --input ../AIgroKB/results_final/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &


# # 'OIE-GP+NCD', 
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &

#==========================================================

# # 'openNCDKB',
# python3 sts.py --input ../AIgroKB/results_final/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &


# # 'Conceptnet+OIE-GP+NCD,(N=2)', 
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-conceptnet-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-conceptnet-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-conceptnet-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-conceptnet-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &


# # 'Conceptnet+NCD,(N=2)', 
# python3 sts.py --input ../AIgroKB/results_final/ncd-conceptnet-transformer_epochs-40_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-conceptnet-transformer_epochs-40_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-conceptnet-transformer_epochs-40_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-conceptnet-transformer_epochs-40_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &

# #==========================================================

# # 'OIE-GP+NCD,(N=2)', 
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-gp-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &


# # 'openNCDKB,(N=2)',
# python3 sts.py --input ../AIgroKB/results_final/ncd-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/ncd-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &

# # 'CSRncdKBC-attentionGRU_epochs-40_seqlen-10',
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &

# #==========================================================

# # 'CSRncdKBC-attentionGRU_epochs-40_seqlen-30',
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-40_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &

# # 'CSRncdKBC-attentionGRU_epochs-100_seqlen-10',
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
# python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &

# 'CSRncdKBC-attentionGRU_epochs-100_seqlen-30',
python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-100_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-100_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs_val.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-100_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &
python3 sts.py --input ../AIgroKB/results_final/CSRncdKBC-attentionGRU_epochs-100_seqlen-30_maxfeat-15000_batch-32_embdim-1024_steps-1024/object_pairs_val_random.tsv --embedmodel fstx_300d_indexed/ --format wisse --idfmodel none &