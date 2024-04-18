
CUDA_VISIBLE_DEVICES=3 nohup python predict_ours_batch_prediction.py --function_sets database_msa_sampled_100_neg_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_neg_aug/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_neg_aug.pth  --queue_size 1024 --output_dir neg_aug_large > neg_aug_large.out &

CUDA_VISIBLE_DEVICES=1 nohup python predict_ours_batch_prediction.py --function_sets database_msa_sampled_100_pos_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_pos_aug/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_pos_aug.pth  --queue_size 1024 --output_dir pos_aug_large > pos_aug_large.out &

 
CUDA_VISIBLE_DEVICES=2 nohup python predict_ours_batch_prediction.py --function_sets database_msa_sampled_100_weak_neg_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_weak_neg_aug/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_weak_neg_aug.pth  --queue_size 1024 --output_dir weak_neg_aug_large > weak_neg_aug_large.out &