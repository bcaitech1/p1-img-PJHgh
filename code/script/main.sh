# version_0
python mk_labeling_info.py

python training.py --task "mask"\
                   --USE_DATA_SPLIT "False"\
                   --epochs 1\
                   --name "final_version"
                   
python training.py --task "gender"\
                   --USE_DATA_SPLIT "False"\
                   --epochs 1\
                   --name "final_version"
                   
python training.py --task "age"\
                   --USE_DATA_SPLIT "False"\
                   --epochs 1\
                   --name "final_version"
                   
python mk_submission.py --version "final_version"