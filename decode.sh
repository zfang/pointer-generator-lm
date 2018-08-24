MODEL=$1
shift
MODE=$1
shift
CUDA=$1
shift
CUDA_VISIBLE_DEVICES=$CUDA python decode.py --path=.decode/$MODEL/$MODE --model_dir=pretrained/$MODEL --$MODE $@
