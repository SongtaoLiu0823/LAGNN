python lagcn_arxiv.py --concat 1 --samples 2 --num_layers 3 --hidden_channels 128 --lr 0.01 --epochs 500 --lam 1 --tem 0.5
python lagcn_proteins.py --concat 1 --samples 2 --num_layers 3 --hidden_channels 128 --lr 0.01 --epochs 1000
python lagcn_products_average.py --num_layers 3 --hidden_channels 256 --lr 0.01 --epochs 300

python lasage_arxiv.py --concat 1 --samples 4 --num_layers 3 --hidden_channels 128 --lr 0.01 --epochs 500 --lam 1 --tem 0.5
python lasage_proteins.py --concat 1 --samples 2 --num_layers 3 --hidden_channels 128 --lr 0.01 --epochs 1000
python lasage_products_average.py --concat 1 --samples 4 --hidden_channels 256 --tem 0.5 --lam 1

python lagat_arxiv.py --concat 1 --samples 2 --hidden_channels 125 --num_layers 3 --lam 1 --tem 0.5 --epochs 2000 --use-norm --use-labels --no-attn-dst --edge-drop 0 --input-drop 0
python lagat_products_average.py --concat 1 --samples 2
