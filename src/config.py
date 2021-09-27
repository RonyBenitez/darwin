SAMPLING_RATIO=0.4

EPOCHS=20
BATCH_SIZE=8

IMAGE_SIZE=(224,224)

IMG_PATH="../input/after_4_bis/*/*.jpg"
EMBED_IMG_PATH="../../input/reference_images/*.jpg"
LOG_DIR="logdir_train"
LR=1e-3

DEVICE="cuda"
MODEL_SAVEPATH="./modelsave"