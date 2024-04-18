msa_sampled_100_weak_neg_aug.pth

mkdir database_msa_sampled_100_weak_neg_aug
CUDA_VISIBLE_DEVICES=2 nohup python create_reference_database_ours.py data/csa_functional_sites.csv database_msa_sampled_100_weak_neg_aug/csa_site_db_nn.pkl database_msa_sampled_100_weak_neg_aug/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_sampled_100_weak_neg_aug.pth &

mkdir weak_neg_aug/
for name in A0A6C1DXI9 C7GME9 G0WAH2 G8BPC4 A0A0U2MLH5 A0A6J0EAD3 A0A384P5L6 A0A0B4Y0Q1 A0A143DE76 A0A5N4C2U4 G3R7S1 P84010 Q9YHY9; do 
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_msa_sampled_100_weak_neg_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_weak_neg_aug/csa_site_db_nn.pkl --pdb test_fasta/AF-$name-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_weak_neg_aug.pth  --queue_size 1024
done 
mv ours_rnk_*.csv weak_neg_aug/



mkdir database_msa_sampled_100_neg_aug
CUDA_VISIBLE_DEVICES=2 nohup python create_reference_database_ours.py data/csa_functional_sites.csv database_msa_sampled_100_neg_aug/csa_site_db_nn.pkl database_msa_sampled_100_neg_aug/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_sampled_100_neg_aug.pth &


mkdir neg_aug/
for name in A0A6C1DXI9 C7GME9 G0WAH2 G8BPC4 A0A0U2MLH5 A0A6J0EAD3 A0A384P5L6 A0A0B4Y0Q1 A0A143DE76 A0A5N4C2U4 G3R7S1 P84010 Q9YHY9; do 
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_msa_sampled_100_neg_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_neg_aug/csa_site_db_nn.pkl --pdb test_fasta/AF-$name-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_neg_aug.pth  --queue_size 1024
done 
mv ours_rnk_*.csv neg_aug/
python summarize.py --dir neg_aug



CUDA_VISIBLE_DEVICES=3 nohup python alanine_scan.py --function_sets database_msa_sampled_100_neg_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_neg_aug/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_neg_aug.pth --queue_size 1024 --output_name alanine_scan_msa_sampled_100_neg_aug.txt &


CUDA_VISIBLE_DEVICES=2 nohup python alanine_scan.py --function_sets database_msa_sampled_100_weak_neg_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_weak_neg_aug/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_weak_neg_aug.pth --queue_size 1024 --output_name alanine_scan_msa_sampled_100_weak_neg_aug.txt &





mkdir database_msa_sampled_100_pos_aug
CUDA_VISIBLE_DEVICES=2 nohup python create_reference_database_ours.py data/csa_functional_sites.csv database_msa_sampled_100_pos_aug/csa_site_db_nn.pkl database_msa_sampled_100_pos_aug/csa_function_sets_nn.pkl --pdb_dir pdb/ --model MoCo --checkpoint msa_sampled_100_pos_aug.pth &
mkdir pos_aug/
for name in A0A6C1DXI9 C7GME9 G0WAH2 G8BPC4 A0A0U2MLH5 A0A6J0EAD3 A0A384P5L6 A0A0B4Y0Q1 A0A143DE76 A0A5N4C2U4 G3R7S1 P84010 Q9YHY9; do 
CUDA_VISIBLE_DEVICES=1 python predict_ours.py --function_sets database_msa_sampled_100_pos_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_pos_aug/csa_site_db_nn.pkl --pdb test_fasta/AF-$name-F1-model_v4.pdb --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_pos_aug.pth  --queue_size 1024
done 
mv ours_rnk_*.csv pos_aug/
python summarize.py --dir pos_aug


CUDA_VISIBLE_DEVICES=3 nohup python alanine_scan.py --function_sets database_msa_sampled_100_pos_aug/csa_function_sets_nn.pkl --db database_msa_sampled_100_pos_aug/csa_site_db_nn.pkl --pdb_dir pdb/ --cutoff 1e-6 --model MoCo --checkpoint msa_sampled_100_pos_aug.pth --queue_size 1024 --output_name alanine_scan_msa_sampled_100_pos_aug.txt &