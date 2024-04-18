mkdir neg_aug_large/

CUDA_VISIBLE_DEVICES=1 python predict_ours_batch_prediction.py --function_sets database_msa_sampled_100_neg_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_neg_aug/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_neg_aug.pth  --queue_size 1024
mv ours_rnk_*.csv neg_aug_large/


