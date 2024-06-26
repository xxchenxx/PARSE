mkdir msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000
CUDA_VISIBLE_DEVICES=0 python create_reference_database_ours.py data/csa_functional_sites.csv msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth 


mkdir pos_weak_neg_aug_ep100_version2_16000
CUDA_VISIBLE_DEVICES=0 nohup python predict_ours_batch_prediction.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000 > pos_weak_neg_aug_16000.out &

CUDA_VISIBLE_DEVICES=3 nohup python alanine_scan.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth --queue_size 1024 --output_name pos_weak_neg_aug_ep100_version2_16000.txt &


mkdir pos_weak_neg_aug_ep100_version2_16000_hard
CUDA_VISIBLE_DEVICES=0 nohup python predict_ours_batch_prediction.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir test_pdb_hard --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000_hard > pos_weak_neg_aug_16000.out &

CUDA_VISIBLE_DEVICES=1 nohup python predict_ours_batch_prediction.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_500 > pos_weak_neg_aug.out &

CUDA_VISIBLE_DEVICES=2 nohup python predict_ours_batch_prediction.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500/csa_site_db_nn.pkl --pdb_dir test_pdb_hard --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_500_hard > pos_weak_neg_aug.out &