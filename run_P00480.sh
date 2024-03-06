for name in G3R7S1 A0A5N4C2U4; do 
wget https://alphafold.ebi.ac.uk/files/AF-${name}-F1-model_v4.pdb
cp AF-${name}-F1-model_v4.pdb P00480_homologs_with_neighbors
cp AF-${name}-F1-model_v4.pdb P00480_homologs
rm AF-${name}-F1-model_v4.pdb
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb P00480_homologs_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-6 
mv *${name}* P00480_homologs_with_neighbors/
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P00480_homologs_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-1 
mv *${name}* P00480_homologs_with_neighbors/

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb P00480_homologs_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-6 
mv *${name}* P00480_homologs/
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P00480_homologs_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-1  --function_sets database_without_neighbors/csa_function_sets_nn.pkl --db database_without_neighbors/csa_site_db_nn.pkl
mv *${name}* P00480_homologs/
done 


for name in Q9YHY9 P84010; do 
wget https://alphafold.ebi.ac.uk/files/AF-${name}-F1-model_v4.pdb
cp AF-${name}-F1-model_v4.pdb P00480_homologs_with_neighbors
cp AF-${name}-F1-model_v4.pdb P00480_homologs
rm AF-${name}-F1-model_v4.pdb
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_with_neighbors/csa_function_sets_nn.pkl --db database_with_neighbors/csa_site_db_nn.pkl --pdb P00480_homologs_0.5_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-6 
mv *${name}* P00480_homologs_0.5_with_neighbors/
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P00480_homologs_0.5_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-1 
mv *${name}* P00480_homologs_0.5_with_neighbors/

CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets csa_function_sets_nn.pkl --db csa_site_db_nn.pkl --pdb P00480_homologs_0.5_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-6 
mv *${name}* P00480_homologs_0.5/
CUDA_VISIBLE_DEVICES=1 python predict.py --pdb P00480_homologs_0.5_with_neighbors/AF-${name}-F1-model_v4.pdb --cutoff 1e-1  --function_sets database_without_neighbors/csa_function_sets_nn.pkl --db database_without_neighbors/csa_site_db_nn.pkl
mv *${name}* P00480_homologs_0.5/
done 



