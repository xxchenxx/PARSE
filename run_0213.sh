mkdir database_moco_mixed_no_weights
CUDA_VISIBLE_DEVICES=0 python create_reference_database_ours.py data/csa_functional_sites.csv database_moco_mixed_no_weights/csa_site_db_nn.pkl database_moco_mixed_no_weights/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint moco_pair.pth 

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_moco_mixed_no_weights/csa_function_sets_nn.pkl --db database_moco_mixed_no_weights/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_full_filtered_sampled_lr1e-3_wd1e-3_ep1000_1706309206.266638_checkpoints.pth.tar --queue_size 1000


mkdir database_moco_mixed_weights
CUDA_VISIBLE_DEVICES=0 python create_reference_database_ours.py data/csa_functional_sites.csv database_moco_mixed_weights/csa_site_db_nn.pkl database_moco_mixed_weights/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint moco_pair_weights.pth 

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_moco_mixed_no_weights/csa_function_sets_nn.pkl --db database_moco_mixed_no_weights/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint moco_pair.pth  --queue_size 1024

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_moco_mixed_weights/csa_function_sets_nn.pkl --db database_moco_mixed_weights/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint moco_pair_weights.pth  --queue_size 1024


mkdir database_moco_pair_weights_ep100
CUDA_VISIBLE_DEVICES=0 python create_reference_database_ours.py data/csa_functional_sites.csv database_moco_pair_weights_ep100/csa_site_db_nn.pkl database_moco_pair_weights_ep100/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint moco_pair_weights_ep100.pth

mkdir database_moco_pair_weights_ep100_weights_100
CUDA_VISIBLE_DEVICES=1 python create_reference_database_ours.py data/csa_functional_sites.csv database_moco_pair_weights_ep100_weights_100/csa_site_db_nn.pkl database_moco_pair_weights_ep100_weights_100/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint moco_pair_weights_ep100_weights_100.pth


mkdir database_moco_pair_weights_ep100_weights_100_queue_4096
CUDA_VISIBLE_DEVICES=3 python create_reference_database_ours.py data/csa_functional_sites.csv database_moco_pair_weights_ep100_weights_100_queue_4096/csa_site_db_nn.pkl database_moco_pair_weights_ep100_weights_100_queue_4096/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint moco_pair_weights_ep100_weights_100_queue_4096.pth --queue_size 4096


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_moco_pair_weights_ep100/csa_function_sets_nn.pkl --db database_moco_pair_weights_ep100/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint moco_pair_weights_ep100.pth  --queue_size 1024


mkdir database_InfoNCE_small_neg_aug
CUDA_VISIBLE_DEVICES=3 python create_reference_database_ours.py data/csa_functional_sites.csv database_InfoNCE_small_neg_aug/csa_site_db_nn.pkl database_InfoNCE_small_neg_aug/csa_function_sets_nn.pkl --pdb_dir pdb/ --model Triplet --checkpoint InfoNCE_small_neg_aug.pth

mkdir database_moco_mix_small_neg_aug
CUDA_VISIBLE_DEVICES=1 python create_reference_database_ours.py data/csa_functional_sites.csv database_moco_mix_small_neg_aug/csa_site_db_nn.pkl database_moco_mix_small_neg_aug/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint moco_mix_small_neg_aug.pth --queue_size 1024


CUDA_VISIBLE_DEVICES=1 python alanine_scan.py --function_sets database_moco_mix_small_neg_aug/csa_function_sets_nn.pkl --db database_moco_mix_small_neg_aug/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint moco_mix_small_neg_aug.pth --queue_size 1024


CUDA_VISIBLE_DEVICES=3 nohup python alanine_scan.py --function_sets database_InfoNCE_small_neg_aug/csa_function_sets_nn.pkl --db database_InfoNCE_small_neg_aug/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model Triplet --checkpoint InfoNCE_small_neg_aug.pth --queue_size 1024 --output_name alanine_scan_InfoNCE_small_neg_aug.txt &




mkdir database_triplet_small_neg_aug_only
CUDA_VISIBLE_DEVICES=3 python create_reference_database_ours.py data/csa_functional_sites.csv database_triplet_small_neg_aug_only/csa_site_db_nn.pkl database_triplet_small_neg_aug_only/csa_function_sets_nn.pkl --pdb_dir pdb/ --model Triplet --checkpoint triplet_small_neg_aug_only.pth


CUDA_VISIBLE_DEVICES=3 nohup python alanine_scan.py --function_sets database_triplet_small_neg_aug_only/csa_function_sets_nn.pkl --db database_triplet_small_neg_aug_only/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model Triplet --checkpoint triplet_small_neg_aug_only.pth --queue_size 1024 --output_name alanine_scan_triplet_small_neg_aug_only.txt &


mkdir database_moco_mix_small_pos_aug
CUDA_VISIBLE_DEVICES=1 python create_reference_database_ours.py data/csa_functional_sites.csv database_moco_mix_small_pos_aug/csa_site_db_nn.pkl database_moco_mix_small_pos_aug/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint moco_mix_small_pos_aug.pth --queue_size 1024


CUDA_VISIBLE_DEVICES=0 python predict_ours.py --function_sets database_moco_mix_small_neg_aug/csa_function_sets_nn.pkl --db database_moco_mix_small_neg_aug/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint moco_mix_small_neg_aug.pth --queue_size 1024

CUDA_VISIBLE_DEVICES=0 python predict_ours.py --function_sets database_moco_mix_small_pos_aug/csa_function_sets_nn.pkl --db database_moco_mix_small_pos_aug/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint moco_mix_small_pos_aug.pth --queue_size 1024