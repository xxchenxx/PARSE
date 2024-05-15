python embed_pdb_dataset.py test_pdb_hard test_pdb_hard_embed --filetype=pdb 

python run_parse_lmdb.py --dataset=test_pdb_hard_embed

CUDA_VISIBLE_DEVICES=2 nohup python alanine_scan_CLEAN.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth --queue_size 1024 --output_name clean.txt > clean.out &


