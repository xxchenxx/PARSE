CUDA_VISIBLE_DEVICES=3 nohup python alanine_scan.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth --queue_size 1024 --output_name pos_weak_neg_aug_ep100_version2_16000.txt &



