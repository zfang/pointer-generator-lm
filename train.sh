LM_COEF=$1
shift
CUDA=$1
shift
CUDA_VISIBLE_DEVICES=$CUDA python train_abstractor.py --path=pretrained/lm_coef_$LM_COEF --w2v=vanilla-word2vec/word2vec.128d.226k.bin --lm=elmo --lm-coef=$LM_COEF --lr_p=10 --patience=20 $@
