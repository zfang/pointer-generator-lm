MODEL=$1
shift
CUDA=$1
shift
CUDA_VISIBLE_DEVICES=$CUDA python train_abstractor.py --path=pretrained/$MODEL --w2v=vanilla-word2vec/word2vec.128d.226k.bin --lm=elmo --lr_p=5 --patience=10 $@
