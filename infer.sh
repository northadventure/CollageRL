### Model configuration
SCALE='small'
WMIN=0.1
WMAX=1.0
HMIN=0.1
HMAX=1.0
LEARNED_MAX_STEPS=10
WEIGHT_PATH='weights/agents/dtd2imgnet10S_random0110'
ALGO=sac

### Goal image configuration
GOAL_PATH='samples/goals/boat.jpg'
GOAL_RESOLUTION=512
GOAL_RESOLUTION_FIT=horizontal

### Material images configuration
SOURCE_DIR='samples/materials/newspaper'
SOURCE_RESOLUTION_RATIO=0.6
SOURCE_LOAD_LIMIT=100
SOURCE_SAMPLE_SIZE=100
MIN_SOURCE_COMPLEXITY=20

### Multi-scale collage configuration
SCALE_ORDER="512 256 128 64 32"
NUM_CYCLES=8
WINDOW_RATIO=0.5
MIN_SCRAP_SIZE=0.01
SENSITIVITY=3.0
FIXED_T=9

### Seuqnce video configuration
FPS=30

python infer.py --scale $SCALE --weight_path $WEIGHT_PATH --num_multi_env 1 \
 --wmin $WMIN --wmax $WMAX --hmin $HMIN --hmax $HMAX --learned_max_steps $LEARNED_MAX_STEPS\
 --goal $GOAL_PATH --goal_resolution $GOAL_RESOLUTION --goal_resolution_fit $GOAL_RESOLUTION_FIT \
 --source_dir $SOURCE_DIR --source_resolution_ratio $SOURCE_RESOLUTION_RATIO \
 --source_load_limit $SOURCE_LOAD_LIMIT --source_sample_size $SOURCE_SAMPLE_SIZE --min_source_complexity $MIN_SOURCE_COMPLEXITY \
 --model_based --algo $ALGO --noop \
 --scale_order $SCALE_ORDER --num_cycles $NUM_CYCLES --window_ratio $WINDOW_RATIO --min_scrap_size $MIN_SCRAP_SIZE \
 --complexity_aware --sensitivity $SENSITIVITY --fixed_t $FIXED_T \
 --skip_negative_reward \
 --paper_like \
 --disallow_duplicate \
 --video_fps $FPS \