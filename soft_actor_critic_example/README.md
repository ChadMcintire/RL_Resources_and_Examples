



Example arguments

#For the Gaussian Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --no-render --no-automatic_entropy_tuning --check_pt_name ch_pt --logs_dir run1_logs --train --QNet other

auto tune alpha 
python3 main.py --env-name Humanoid-v2 --alpha 0.1 --cuda --no-render --automatic_entropy_tuning --check_pt_name ch_pt_at_a.01 --logs_dir run1_at_logs --train --Qnet other

#these need to be updated after testing
#For the Deterministic Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True --policy Determinitst

#For the Deterministic Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True --policy Determinitst --tau 1 --target_update_interval 1000

#use the tensorboard
tensorboard --logdir=runs

http://localhost:6006/
