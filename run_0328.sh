# mkdir database_msa_sampled_100
CUDA_VISIBLE_DEVICES=2 python create_reference_database_ours.py data/csa_functional_sites.csv database_msa_sampled_100/csa_site_db_nn.pkl database_msa_sampled_100/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_mcsa_rcsb_sampled100K.pth 

CUDA_VISIBLE_DEVICES=2 nohup python predict_ours_batch_prediction.py --function_sets database_msa_sampled_100/csa_function_sets_nn.pkl --db database_msa_sampled_100/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_sampled100K.pth   --queue_size 1024 --output_dir normal > normal.out &

CUDA_VISIBLE_DEVICES=3 nohup python alanine_scan.py --function_sets database_msa_sampled_100/csa_function_sets_nn.pkl --db database_msa_sampled_100/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_sampled100K.pth --queue_size 1024 --output_name alanine_scan_msa_sampled_100.txt &