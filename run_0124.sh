python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb price_pdb/pdb/1ea9.pdb --cutoff 1e-6

python predict.py --pdb price_pdb/pdb/1ea9.pdb --cutoff 1e-1

# remove len==1 function


['66', '17', '17', '217', '14'] -> debug ()

# esm embedding to residue distance 



python predict.py --pdb pdb/1fcb.pdb --cutoff 1e-1

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --background function_score_dists_new.pkl

CUDA_VISIBLE_DEVICES=3 python create_reference_database_ours.py data/csa_functional_sites.csv database_medium/csa_site_db_nn.pkl database_medium/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint best_demo_1702870147.059707_checkpoints.pth.tar --queue_size 1024

CUDA_VISIBLE_DEVICES=3 python predict_ours.py --function_sets database_medium/csa_function_sets_nn.pkl --db database_medium/csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model MoCo --checkpoint best_demo_1702870147.059707_checkpoints.pth.tar --queue_size 1024

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000

CUDA_VISIBLE_DEVICES=3 nohup python create_reference_database_ours.py data/csa_functional_sites.csv database_large/csa_site_db_nn.pkl database_large/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000 &

CUDA_VISIBLE_DEVICES=3 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model MoCo --checkpoint best_demo_1702870147.059707_checkpoints.pth.tar --queue_size 1024

CUDA_VISIBLE_DEVICES=3 nohup python create_reference_database_ours.py data/csa_functional_sites.csv database_large_positive_only/csa_site_db_nn.pkl database_large_positive_only/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo_positive_only --checkpoint best_large_lr5e-4_wd1e-3_ep10_positive_only_1705959574.9200642_checkpoints.pth.tar &

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large_positive_only/csa_function_sets_nn.pkl --db database_large_positive_only/csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model MoCo_positive_only --queue_size 1024 --checkpoint best_large_lr5e-4_wd1e-3_ep10_positive_only_1705959574.9200642_checkpoints.pth.tar

CUDA_VISIBLE_DEVICES=3 nohup python create_reference_database_ours.py data/csa_functional_sites.csv database_large_simsiam/csa_site_db_nn.pkl database_large_simsiam/csa_function_sets_nn.pkl --pdb_dir pdb/ --model SimSiam --checkpoint best_large_lr5e-4_wd1e-3_ep10_simsiam_1705962703.2396321_checkpoints.pth.tar &

CUDA_VISIBLE_DEVICES=3 python predict_ours.py --function_sets database_large_simsiam/csa_function_sets_nn.pkl --db database_large_simsiam/csa_site_db_nn.pkl --pdb pdb/1fcb.pdb --cutoff 1e-6 --model SimSiam --queue_size 1024 --checkpoint best_large_lr5e-4_wd1e-3_ep10_simsiam_1705962703.2396321_checkpoints.pth.tar



CUDA_VISIBLE_DEVICES=3 python predict_ours.py --function_sets database_large_simsiam/csa_function_sets_nn.pkl --db database_large_simsiam/csa_site_db_nn.pkl --pdb P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model SimSiam --queue_size 1024 --checkpoint best_large_lr5e-4_wd1e-3_ep10_simsiam_1705962703.2396321_checkpoints.pth.tar

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000