#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

BASE_PORT=3001
BASE_TM_PORT=5001
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=team_code/sparsedrive_b2d_agent.py
TEAM_CONFIG=projects/configs/sparsedrive_stage2.py+ckpt/sparsedrive_small_b2d_stage2.pth
BASE_CHECKPOINT_ENDPOINT=close_loop_log/result/bench2drive
SAVE_PATH=close_loop_log/result/save
PLANNER_TYPE=only_traj
GPU_RANK=0

PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"

bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
