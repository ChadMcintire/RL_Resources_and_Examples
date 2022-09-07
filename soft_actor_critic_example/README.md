



Example arguments

#For the Gaussian Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --no-render --no-automatic_entropy_tuning


#these need to be updated after testing
#For the Deterministic Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True --policy Determinitst

#For the Deterministic Model
python3 main.py --env-name Humanoid-v2 --alpha 0.05 --cuda --render True --policy Determinitst --tau 1 --target_update_interval 1000
