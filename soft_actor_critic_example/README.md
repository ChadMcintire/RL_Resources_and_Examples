



Example arguments

#For the Gaussian Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True

#For the Deterministic Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True --policy Determinitst

#For the Deterministic Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True --policy Determinitst --tau 1 --target_update_interval 1000
