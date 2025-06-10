import pandas as pd

"""
This script is used to get the prediction pairs obtained from the val set (). It will perform a clean up and verify that they are valid (based on length)
Afterwards it will generate a tsv with those object pairs. 
"""

# csv_files  = [
#     'results_final/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
#     'results_final/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
#     'results_final/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
#     'results_final/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/'
# ]

csv_files  = [
    'results_final/ncd-gp-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-gp-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-gp-conceptnet-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-conceptnet-transformer_epochs-40_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-gp-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/',
    'results_final/ncd-transformer_epochs-100_stackSize-2_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8/'
]



for num,csv_item in enumerate(csv_files):
    object_pairs  = list()
    inferences_item = pd.read_csv(csv_item+'val_predictions.csv',
            header=0,
            index_col=0)
    for idx,inference in inferences_item.iterrows():
        # print('&', csv_names[num]+ ' & ', inference['Obj'], '\\\\' )
        inference_obj = str(inference['Obj']).replace('[start] ','').replace(' [end]','')
        inference_obj_true = str(inference['Obj_true']).replace('[start] ','').replace(' [end]','')
        if(len(inference_obj)> 0 and len(inference_obj_true)>0):
            object_pairs.append((inference_obj,
                                inference_obj_true))    
    pd.DataFrame(object_pairs).to_csv(csv_item+'object_pairs_val.tsv', sep='\t',header=None,index=None)

