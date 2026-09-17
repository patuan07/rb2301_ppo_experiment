#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
source .venv/bin/activate
source install/setup.bash
set -u

run_dir="${1:-runs/imitation_$(date +%Y%m%d_%H%M%S)}"
expert_episodes="${EXPERT_EPISODES:-500}"
dagger_episodes="${DAGGER_EPISODES:-100}"
bc_epochs="${BC_EPOCHS:-50}"
dagger_epochs="${DAGGER_EPOCHS:-15}"
maze_difficulty="${MAZE_DIFFICULTY:-medium}"
collection_envs="${COLLECTION_ENVS:-4}"
collection_seed="${COLLECTION_SEED:-12301}"

mkdir -p "$run_dir/data" "$run_dir/models" "$run_dir/simulator_logs"

python -m rb2301_ca1.collect_demonstrations \
  --output "$run_dir/data/expert.npz" \
  --episodes "$expert_episodes" \
  --num-envs "$collection_envs" \
  --seed "$collection_seed" \
  --maze-difficulty "$maze_difficulty" \
  --run-dir "$run_dir/simulator_logs/expert"

python -m rb2301_ca1.train_imitation \
  --dataset "$run_dir/data/expert.npz" \
  --output "$run_dir/models/bc_model.zip" \
  --epochs "$bc_epochs" \
  --action-std 0.30

previous_dataset="$run_dir/data/expert.npz"
previous_model="$run_dir/models/bc_model.zip"
probabilities=(0.35 0.70 1.00)

for index in 1 2 3; do
  probability="${probabilities[$((index - 1))]}"
  next_dataset="$run_dir/data/dagger_${index}.npz"
  next_model="$run_dir/models/dagger_${index}_model.zip"
  python -m rb2301_ca1.collect_demonstrations \
    --output "$next_dataset" \
    --append "$previous_dataset" \
    --episodes "$dagger_episodes" \
    --num-envs "$collection_envs" \
    --seed "$((collection_seed + index * 100000))" \
    --maze-difficulty "$maze_difficulty" \
    --student-model "$previous_model" \
    --student-control-probability "$probability" \
    --no-successful-only \
    --run-dir "$run_dir/simulator_logs/dagger_${index}"
  python -m rb2301_ca1.train_imitation \
    --dataset "$next_dataset" \
    --initial-model "$previous_model" \
    --output "$next_model" \
    --epochs "$dagger_epochs" \
    --action-std 0.30
  previous_dataset="$next_dataset"
  previous_model="$next_model"
done

printf 'DAgger model ready: %s\n' "$previous_model"
printf 'Fine-tune conservatively with: ./train_rl.sh --algorithm ppo --resume %s --teacher-model %s --teacher-anchor-strength 0.02 --learning-rate 0.000025 --ent-coef 0.0 --action-std 0.18 --ppo-clip-range 0.10 --ppo-n-epochs 5 --ppo-target-kl 0.01 --timesteps 100000 --num-envs %s --device cpu\n' "$previous_model" "$previous_model" "$collection_envs"
