CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P00175_homologs/AF-C7GME9-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P00480_homologs/AF-G3R7S1-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P00480_homologs_0.5/AF-Q9YHY9-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P35505_homologs/AF-A0A6J0EAD3-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P35505_homologs_0.5/AF-A0A384P5L6-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000



CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb pdb/2gsa.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000


CUDA_VISIBLE_DEVICES=3 python predict_ours_input_with_uniprot_1\ copy.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb pdb/2gsa.pdb --cutoff 1e-6 

CUDA_VISIBLE_DEVICES=3 python predict_ours_input_with_uniprot_2\ copy.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb pdb/2gsa.pdb --cutoff 1e-6 

CUDA_VISIBLE_DEVICES=3 python predict_ours_input_with_uniprot_3\ copy.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb pdb/2gsa.pdb --cutoff 1e-6 