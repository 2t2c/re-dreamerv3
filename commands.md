### Training Script

python dreamerv3/main.py \
    --logdir $HOME/logdir/dreamer/{timestamp} \
    --configs atari100k \
    --run.train_ratio 32 \
    --jax.platform cpu \
    --batch_size 1

### View Results
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000