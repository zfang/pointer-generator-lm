MODEL=$1
shift
MODE=$1
shift
python evaluate_model.py --rouge --decode_dir=.decode/$MODEL/$MODE -f $@
python evaluate_model.py --meteor --decode_dir=.decode/$MODEL/$MODE  -f $@
