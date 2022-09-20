



Example arguments

#For the Gaussian Model
#no auto tuning
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --no-render --no-automatic_entropy_tuning --check_pt_name alt-mod-no-auto --logs_dir run1_logs --train --QNet other

#auto tune alpha 
python3 main.py --env-name Humanoid-v2 --alpha 0.1 --cuda --no-render --automatic_entropy_tuning --check_pt_name alt-mod-auto --logs_dir run1_at_logs --train --Qnet other

#these need to be updated after testing
#For the Deterministic Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True --policy Determinitst

#For the Deterministic Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True --policy Determinitst --tau 1 --target_update_interval 1000

#run trained network
python3 main.py --env-name Humanoid-v2 --alpha 0.1 --render --cuda --no-train --Qnet other --check_pt_name path

#use the tensorboard
tensorboard --logdir=runs

http://localhost:6006/
