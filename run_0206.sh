for name in triplet_pair_trivial; do 
mkdir database_${name}
CUDA_VISIBLE_DEVICES=0 python create_reference_database_ours.py data/csa_functional_sites.csv database_${name}/csa_site_db_nn.pkl database_${name}/csa_function_sets_nn.pkl --pdb_dir pdb/ --model Triplet --checkpoint ${name}.pth
done


CUDA_VISIBLE_DEVICES=1 python alanine_scan.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large/csa_function_sets_nn.pkl --db database_large/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model Triplet --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000

mkdir database_large_new

CUDA_VISIBLE_DEVICES=2 python create_reference_database_ours.py data/csa_functional_sites.csv database_large_new/csa_site_db_nn.pkl database_large_new/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000 

mkdir database_full_subsampled
CUDA_VISIBLE_DEVICES=3 python create_reference_database_ours.py data/csa_functional_sites.csv database_full_subsampled/csa_site_db_nn.pkl database_full_subsampled/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint best_full_filtered_sampled_lr1e-3_wd1e-3_ep1000_1706309206.266638_checkpoints.pth.tar --queue_size 1000 



CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_large_new/csa_function_sets_nn.pkl --db database_large_new/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar --queue_size 1000 --prefix HHHHHH


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_triplet_pair_trivial/csa_function_sets_nn.pkl --db database_triplet_pair_trivial/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model Triplet --checkpoint triplet_pair_trivial.pth --queue_size 1000

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_triplet_pair/csa_function_sets_nn.pkl --db database_triplet_pair/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model Triplet --checkpoint triplet_pair.pth --queue_size 1000


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_full_subsampled/csa_function_sets_nn.pkl --db database_full_subsampled/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model Triplet --checkpoint best_full_filtered_sampled_lr1e-3_wd1e-3_ep1000_1706309206.266638_checkpoints.pth.tar --queue_size 1000


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_full_subsampled/csa_function_sets_nn.pkl --db database_full_subsampled/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_full_filtered_sampled_lr1e-3_wd1e-3_ep1000_1706309206.266638_checkpoints.pth.tar --queue_size 1000



CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_full_subsampled/csa_function_sets_nn.pkl --db database_full_subsampled/csa_site_db_nn.pkl --pdb results/P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint best_full_filtered_sampled_lr1e-3_wd1e-3_ep1000_1706309206.266638_checkpoints.pth.tar --queue_size 1000 --prefix HHHHHH