mkdir database_msa_sampled_20
CUDA_VISIBLE_DEVICES=2 nohup python create_reference_database_ours.py data/csa_functional_sites.csv database_msa_sampled_20/csa_site_db_nn.pkl database_msa_sampled_20/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_sampled_20.pth &


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_msa_sampled_20/csa_function_sets_nn.pkl --db database_msa_sampled_20/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_20.pth  --queue_size 1024


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_msa_sampled_20/csa_function_sets_nn.pkl --db database_msa_sampled_20/csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_20.pth  --queue_size 1024

python calculate_metrics.py --rank_file ours_rnk_AF-G8BPC4-F1-model_v4_msa_20.csv
python calculate_metrics.py --rank_file ours_rnk_1fcb_msa_20.csv


mkdir database_sampled_pairs_small_pos_neg_aug
CUDA_VISIBLE_DEVICES=2 nohup python create_reference_database_ours.py data/csa_functional_sites.csv database_sampled_pairs_small_pos_neg_aug/csa_site_db_nn.pkl database_sampled_pairs_small_pos_neg_aug/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint sampled_pairs_small_pos_neg_aug.pth &

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_sampled_pairs_small_pos_neg_aug/csa_function_sets_nn.pkl --db database_sampled_pairs_small_pos_neg_aug/csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model MoCo --checkpoint sampled_pairs_small_pos_neg_aug.pth  --queue_size 1024

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_sampled_pairs_small_pos_neg_aug/csa_function_sets_nn.pkl --db database_sampled_pairs_small_pos_neg_aug/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint sampled_pairs_small_pos_neg_aug.pth  --queue_size 1024

python calculate_metrics.py --rank_file ours_rnk_AF-G8BPC4-F1-model_v4.csv
python calculate_metrics.py --rank_file ours_rnk_1fcb.csv

mv ours_rnk_AF-G8BPC4-F1-model_v4.csv ours_rnk_AF-G8BPC4-F1-model_v4_pos_neg_aug.csv
mv ours_rnk_1fcb.csv ours_rnk_1fcb_pos_neg_aug.csv


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_sampled_pairs_small_pos_neg_aug/csa_function_sets_nn.pkl --db database_sampled_pairs_small_pos_neg_aug/csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model MoCo --checkpoint sampled_pairs_small_pos_neg_aug.pth  --queue_size 1024

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_sampled_pairs_small_pos_neg_aug/csa_function_sets_nn.pkl --db database_sampled_pairs_small_pos_neg_aug/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint sampled_pairs_small_pos_neg_aug.pth  --queue_size 1024


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_moco_mix_small_neg_aug/csa_function_sets_nn.pkl --db database_moco_mix_small_neg_aug/csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model MoCo --checkpoint moco_mix_small_neg_aug.pth --queue_size 1024

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_moco_mix_small_neg_aug/csa_function_sets_nn.pkl --db database_moco_mix_small_neg_aug/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint moco_mix_small_neg_aug.pth --queue_size 1024

mv ours_rnk_AF-G8BPC4-F1-model_v4.csv ours_rnk_AF-G8BPC4-F1-model_v4_neg_aug.csv
mv ours_rnk_1fcb.csv ours_rnk_1fcb_neg_aug.csv


python calculate_metrics.py --rank_file ours_rnk_AF-G8BPC4-F1-model_v4_neg_aug.csv
python calculate_metrics.py --rank_file ours_rnk_AF-G8BPC4-F1-model_v4_pos_neg_aug.csv