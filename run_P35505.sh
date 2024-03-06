for name in A0A6J0EAD3 A0A0U2MLH5; do 
mkdir P35505_homologs_with_neighbors
mkdir P35505_homologs
wget https://alphafold.ebi.ac.uk/files/AF-${name}-F1-model_v4.pdb
cp AF-${name}-F1-model_v4.pdb P35505_homologs_with_neighbors
cp AF-${name}-F1-model_v4.pdb P35505_homologs
rm AF-${name}-F1-model_v4.pdb
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb P35505_homologs_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-6 
mv *${name}* P35505_homologs_with_neighbors/
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P35505_homologs_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-1 
mv *${name}* P35505_homologs_with_neighbors/

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb P35505_homologs_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-6 
mv *${name}* P35505_homologs/
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P35505_homologs_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-1  --function_sets database_without_neighbors/csa_function_sets_nn.pkl --db database_without_neighbors/csa_site_db_nn.pkl
mv *${name}* P35505_homologs/
done 


for name in A0A384P5L6 H0YLC7; do 
wget https://alphafold.ebi.ac.uk/files/AF-${name}-F1-model_v4.pdb
mkdir P35505_homologs_0.5_with_neighbors
mkdir P35505_homologs_0.5
cp AF-${name}-F1-model_v4.pdb P35505_homologs_0.5_with_neighbors
cp AF-${name}-F1-model_v4.pdb P35505_homologs_0.5
rm AF-${name}-F1-model_v4.pdb
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb P35505_homologs_0.5_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-6 
mv *${name}* P35505_homologs_0.5_with_neighbors/
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P35505_homologs_0.5_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-1 
mv *${name}* P35505_homologs_0.5_with_neighbors/

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb P35505_homologs_0.5_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-6 
mv *${name}* P35505_homologs_0.5/
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P35505_homologs_0.5_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-1  --function_sets database_without_neighbors/csa_function_sets_nn.pkl --db database_without_neighbors/csa_site_db_nn.pkl
mv *${name}* P35505_homologs_0.5/
done 



