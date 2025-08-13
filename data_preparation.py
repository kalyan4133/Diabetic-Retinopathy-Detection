import tensorflow as tf

# Load images from the directory
dataset_path = "K:/diabetic_retinopathy/data/images"

# Set image parameters
img_size = (224, 224)
batch_size = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',  # Since it's multi-class classification
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',
    validation_split=0.2,
    subset="validation",
    seed=123
)
 
