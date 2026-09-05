"""Hyperparameters, kept in one place and matching the values used for the
run reported in the README (test MAE 2.83 m). Changing anything here changes
what the published numbers mean, so they are defined once rather than
scattered across the training and evaluation code.
"""

IMG_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42

# The backbone is frozen for this many epochs so the randomly-initialised
# regression head can settle before gradients start moving the pretrained
# features.
FREEZE_BACKBONE_EPOCHS = 5

# Huber rather than MSE: building heights are long-tailed (median ~6 m, max
# ~307 m), so squared error would let a handful of skyscrapers dominate the
# gradient. delta=3.0 m is roughly the error scale we care about.
HUBER_DELTA = 3.0

# Epochs without validation improvement before stopping.
PATIENCE = 15

# Height buckets used in the evaluation breakdown, in metres.
HEIGHT_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 200)]
