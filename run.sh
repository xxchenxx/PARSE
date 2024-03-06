CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb AF-A9SIZ6-F1-model_v4.pdb --cutoff 1e-6 

CUDA_VISIBLE_DEVICES=1 python predict.py --pdb AF-A9SIZ6-F1-model_v4.pdb --cutoff 1e-1 


mkdir P00175_homologs

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb P00175_homologs/AF-C7GME9-F1-model_v4.pdb --cutoff 1e-6 

CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P00175_homologs/AF-C7GME9-F1-model_v4.pdb --cutoff 1e-1 
mv *C7GME9* P00175_homologs


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb AF-A0A6C1DXI9-F1-model_v4.pdb --cutoff 1e-6 

CUDA_VISIBLE_DEVICES=1 python predict.py --pdb AF-A0A6C1DXI9-F1-model_v4.pdb --cutoff 1e-1 
mv *A0A6C1DXI9* P00175_homologs

mkdir P00175_homologs_0.5
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 

CUDA_VISIBLE_DEVICES=1 python predict.py --pdb AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-1 
mv *G8BPC4* P00175_homologs_0.5


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb AF-G0WAH2-F1-model_v4.pdb --cutoff 1e-6 

CUDA_VISIBLE_DEVICES=1 python predict.py --pdb AF-G0WAH2-F1-model_v4.pdb --cutoff 1e-1 
mv *G0WAH2* P00175_homologs_0.5


mkdir P00175_homologs_with_neighbors
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb P00175_homologs/AF-C7GME9-F1-model_v4.pdb --cutoff 1e-6 
mv *C7GME9* P00175_homologs_with_neighbors

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb P00175_homologs/AF-A0A6C1DXI9-F1-model_v4.pdb --cutoff 1e-6 
mv *A0A6C1DXI9* P00175_homologs_with_neighbors

mkdir P00175_homologs_0.5_with_neighbors
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb P00175_homologs_0.5/AF-G8BPC4-F1-model_v4.pdb --cutoff 1e-6 
mv *G8BPC4* P00175_homologs_0.5_with_neighbors

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb P00175_homologs_0.5/AF-G0WAH2-F1-model_v4.pdb --cutoff 1e-6 
mv *G0WAH2* P00175_homologs_0.5_with_neighbors



CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb Q2RSB2_homologs_0.5_with_neighbors/AF-A0A0B4Y0Q1-F1-model_v4.pdb --cutoff 1e-6 
mv *A0A0B4Y0Q1* Q2RSB2_homologs_0.5_with_neighbors/

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb Q2RSB2_homologs_0.5_with_neighbors/AF-A0A143DE76-F1-model_v4.pdb --cutoff 1e-6 
mv *A0A143DE76* Q2RSB2_homologs_0.5_with_neighbors/

CUDA_VISIBLE_DEVICES=1 python predict.py --pdb Q2RSB2_homologs_0.5_with_neighbors/AF-A0A0B4Y0Q1-F1-model_v4.pdb --cutoff 1e-1 
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb Q2RSB2_homologs_0.5_with_neighbors/AF-A0A143DE76-F1-model_v4.pdb --cutoff 1e-1 
mv *A0A0B4Y0Q1* Q2RSB2_homologs_0.5_with_neighbors/
mv *A0A143DE76* Q2RSB2_homologs_0.5_with_neighbors/


CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb Q2RSB2_homologs_0.5/AF-A0A0B4Y0Q1-F1-model_v4.pdb --cutoff 1e-6 
mv *A0A0B4Y0Q1* Q2RSB2_homologs_0.5/

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb Q2RSB2_homologs_0.5/AF-A0A143DE76-F1-model_v4.pdb --cutoff 1e-6 
mv *A0A143DE76* Q2RSB2_homologs_0.5/

CUDA_VISIBLE_DEVICES=1 python predict.py --pdb Q2RSB2_homologs_0.5/AF-A0A0B4Y0Q1-F1-model_v4.pdb --cutoff 1e-1 --function_sets database_without_neighbors/csa_function_sets_nn.pkl --db database_without_neighbors/csa_site_db_nn.pkl
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb Q2RSB2_homologs_0.5/AF-A0A143DE76-F1-model_v4.pdb --cutoff 1e-1 --function_sets database_without_neighbors/csa_function_sets_nn.pkl --db database_without_neighbors/csa_site_db_nn.pkl
mv *A0A0B4Y0Q1* Q2RSB2_homologs_0.5/
mv *A0A143DE76* Q2RSB2_homologs_0.5/



