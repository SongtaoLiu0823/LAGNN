python lagcn.py --dataset cora --epochs 2000 --hidden 8 --concat 1 --samples 4 --lr 0.01
python lagcn.py --dataset citeseer --epochs 2000 --hidden 8 --concat 1 --samples 4 --lr 0.01
python lagcn.py --dataset pubmed --epochs 300 --hidden 4 --concat 3 --samples 4 --lr 0.02

python lagat.py --dataset cora --epochs 1000 --hidden 8 --concat 1 --samples 4 --lr 0.01 --weight_decay 5e-4 --nb_heads 4 --nb_heads_2 1 --dropout 0.6 --alpha 0.2
python lagat.py --dataset citeseer --epochs 1000 --hidden 8 --concat 1 --samples 4 --lr 0.01 --weight_decay 5e-4 --nb_heads 4 --nb_heads_2 1 --dropout 0.6 --alpha 0.2
python lagat.py --dataset pubmed --epochs 1000 --hidden 8 --concat 3 --samples 4 --lr 0.01 --weight_decay 5e-4 --nb_heads 2 --nb_heads_2 1 --dropout 0.6 --alpha 0.2

python lagcnii.py --dataset cora --epochs 1000 --hidden 32 --concat 1 --sample 4 --lr 0.01 --wd1 0.01 --wd2 5e-4 --layer 64 --dropout 0.6 --alpha 0.1 --lamda 0.5 --patience 200
python lagcnii.py --dataset citeseer --epochs 1000 --hidden 128 --concat 1 --sample 4 --lr 0.01 --wd1 0.01 --wd2 5e-4 --layer 32 --dropout 0.7 --alpha 0.1 --lamda 0.6 --patience 200 --consistency
python lagcnii.py --dataset pubmed --epochs 1000 --hidden 64 --concat 3 --sample 4 --lr 0.01 --wd1 5e-4 --wd2 5e-4 --layer 16 --dropout 0.5 --alpha 0.1 --lamda 0.4 --patience 200 --consistency

python lagrand.py --dataset cora --input_droprate 0.5 --hidden_droprate 0.5 --dropnode_rate 0.5 --order 8 --sample 4 --tem 0.5 --lam 1.0 --lr 1e-2 --patience 200
python lagrand.py --dataset citeseer --input_droprate 0.2 --hidden_droprate 0.1 --dropnode_rate 0 --order 2 --sample 4 --tem 0.2 --lam 0.7 --lr 1e-2 --patience 200
python lagrand.py --dataset pubmed --input_droprate 0.7 --hidden_droprate 0.8 --dropnode_rate 0.7 --order 5 --sample 4 --tem 0.2 --lam 1.2 --use_bn --lr 2e-1 --patience 200
