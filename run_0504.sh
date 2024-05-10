mkdir msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x
CUDA_VISIBLE_DEVICES=0 python create_reference_database_ours.py data/csa_functional_sites.csv msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x/csa_site_db_nn.pkl msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x.pth --hidden_dim 1024 --out_dim 256


mkdir msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x
CUDA_VISIBLE_DEVICES=2 python create_reference_database_ours.py data/csa_functional_sites.csv msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x/csa_site_db_nn.pkl msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x.pth --hidden_dim 2048 --out_dim 512

mkdir msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x
CUDA_VISIBLE_DEVICES=1 python create_reference_database_ours.py data/csa_functional_sites.csv msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x/csa_site_db_nn.pkl msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x.pth --hidden_dim 4096 --out_dim 1024

mkdir pos_weak_neg_aug_ep100_version2_16000_hard_2x_2
CUDA_VISIBLE_DEVICES=0 nohup python predict_ours_batch_prediction.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x/csa_site_db_nn.pkl --pdb_dir test_pdb_hard --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000_hard_2x_2 --hidden_dim 1024 --out_dim 256 > pos_weak_neg_aug_16000.out &

mkdir pos_weak_neg_aug_ep100_version2_16000_hard_4x_2
CUDA_VISIBLE_DEVICES=1 nohup python predict_ours_batch_prediction.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x/csa_site_db_nn.pkl --pdb_dir test_pdb_hard --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000_hard_4x_2 --hidden_dim 2048 --out_dim 512 > pos_weak_neg_aug_16000.out &

mkdir pos_weak_neg_aug_ep100_version2_16000_hard_8x_2
CUDA_VISIBLE_DEVICES=2 nohup python predict_ours_batch_prediction.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x/csa_site_db_nn.pkl --pdb_dir test_pdb_hard --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000_hard_8x_2 --hidden_dim 4096 --out_dim 1024 > pos_weak_neg_aug_16000.out &


nohup python summarize_new.py --dir pos_weak_neg_aug_ep100_version2_16000_hard_2x_2 --mapping test_pdb_hard/ids.csv --output pos_weak_neg_aug_ep100_version2_16000_hard_2x_2 &

nohup python summarize_new.py --dir pos_weak_neg_aug_ep100_version2_16000_hard_4x_2 --mapping test_pdb_hard/ids.csv --output pos_weak_neg_aug_ep100_version2_16000_hard_4x_2 &

nohup python summarize_new.py --dir pos_weak_neg_aug_ep100_version2_16000_hard_8x_2 --mapping test_pdb_hard/ids.csv --output pos_weak_neg_aug_ep100_version2_16000_hard_8x_2 &


CUDA_VISIBLE_DEVICES=0 nohup python alanine_scan.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_2x.pth --queue_size 1024 --output_name pos_weak_neg_aug_ep100_version2_16000_2x.txt --hidden_dim 1024 --out_dim 256 &

CUDA_VISIBLE_DEVICES=1 nohup python alanine_scan.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_4x.pth --queue_size 1024 --output_name pos_weak_neg_aug_ep100_version2_16000_4x.txt --hidden_dim 2048 --out_dim 512 &

CUDA_VISIBLE_DEVICES=2 nohup python alanine_scan.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000_8x.pth --queue_size 1024 --output_name pos_weak_neg_aug_ep100_version2_16000_8x.txt --hidden_dim 4096 --out_dim 1024 &

