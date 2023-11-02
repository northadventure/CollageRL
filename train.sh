python train.py --goal imagenet --source dtd --data_path ~/Datasets \
 --scale small --target_width 64 --target_height 64 \
 --wmin 0.1 --wmax 1.0 --hmin 0.1 --hmax 1.0 \
 --algo sac --model_based --noop \
 --replay_size 80000 --num_steps 10 --automatic_entropy_tuning