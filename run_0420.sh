mkdir pos_weak_neg_aug_ep100_version2_16000
CUDA_VISIBLE_DEVICES=0 nohup python extract.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000 > pos_weak_neg_aug_16000.out &



CUDA_VISIBLE_DEVICES=2 python extract.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000 --csv_file /datastor1/deep-proteins/peepin/data/jeff/query_residues_240419.csv --pdb_dir /datastor1/deep-proteins/peepin/data/cifs --output_dir query_residues_240419


CUDA_VISIBLE_DEVICES=2 nohup python extract.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000 --csv_file /datastor1/deep-proteins/peepin/data/jeff/test_residues_240419.csv --pdb_dir /datastor1/deep-proteins/peepin/data/cifs --output_dir test_residues_240419 > test.out &


CUDA_VISIBLE_DEVICES=2 nohup python extract.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000 --csv_file /datastor1/deep-proteins/peepin/data/jeff/test_residues_240423.csv --pdb_dir /datastor1/deep-proteins/peepin/data/cifs --output_dir test_residues_240423 > test.out &

CUDA_VISIBLE_DEVICES=2 nohup python extract.py --function_sets msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_function_sets_nn.pkl --db msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl --pdb_dir test_pdb --cutoff 1e-6 --model MoCo --checkpoint msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000.pth   --queue_size 1024 --output_dir pos_weak_neg_aug_ep100_version2_16000 --csv_file /datastor1/deep-proteins/peepin/data/jeff/query_residues_240423.csv --pdb_dir /datastor1/deep-proteins/peepin/data/cifs --output_dir query_residues_240423 > query.out &


