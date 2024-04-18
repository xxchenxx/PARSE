for i in $(ls dms | grep csv); do 
CUDA_VISIBLE_DEVICES=0 nohup python -u dms_scan.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_500.pth --queue_size 1024 --dms_file dms/${i}
done 

