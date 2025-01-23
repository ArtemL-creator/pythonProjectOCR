import pickle
import json
import keras
from keras.layers import StringLookup
from keras import ops
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path

print("Using device:", tf.config.list_physical_devices('CPU'))

# Параметры
validation_split = 0.2  # Доля данных для тестовой выборки

# Путь к JSON-файлу
json_path = Path("resources/data/train_images/labels.json")

# Чтение и парсинг JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Извлечение данных
image_paths = [entry["image"] for entry in data]
labels = [entry["label"] for entry in data]

# Перемешивание данных
np.random.seed(42)
indices = np.arange(len(image_paths))
np.random.shuffle(indices)

image_paths = np.array(image_paths)[indices]
labels = np.array(labels)[indices]

# Определение индексов для разделения
split_idx = int(len(image_paths) * (1 - validation_split))
train_img_paths = image_paths[:split_idx]
train_labels = labels[:split_idx]
test_img_paths = image_paths[split_idx:]
test_labels = labels[split_idx:]

val_split_idx = int(0.5 * len(test_img_paths))
validation_img_paths = test_img_paths[:val_split_idx]
validation_labels = test_labels[:val_split_idx]

test_img_paths = test_img_paths[val_split_idx:]
test_labels = test_labels[val_split_idx:]

assert len(image_paths) == len(train_img_paths) + len(validation_img_paths) + len(test_img_paths)

print(f"Total training samples: {len(train_img_paths)}")
print(f"Total validation samples: {len(validation_img_paths)}")
print(f"Total test samples: {len(test_img_paths)}")

# Создание маппинга символов
alphabet = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    " ,.!?-—;:"
)

char_to_num = StringLookup(vocabulary=list(alphabet), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - ops.shape(image)[0]
    pad_width = w - ops.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = ops.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image


max_len = max(len(label) for label in labels)

# Преобразование изображений
batch_size = 16
padding_token = 99
# image_width = 128
# image_height = 32
image_width = 256
image_height = 64


def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = ops.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = ops.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)


train_ds = prepare_dataset(train_img_paths, train_labels)
validation_ds = prepare_dataset(validation_img_paths, validation_labels)
test_ds = prepare_dataset(test_img_paths, test_labels)

for data in train_ds.take(1):
    images, labels = data["image"], data["label"]

    for i in range(len(images)):
        # Восстанавливаем путь к изображению
        image_path = train_img_paths[i]  # Путь к текущему изображению

        # Восстанавливаем метку, убирая padding
        label = ''.join(
            [num_to_char(ch).numpy().decode('utf-8') for ch in labels[i] if ch != 0]
        )

        print(f"Image Path: {image_path}")
        print(f"Label: {label}")

for data in train_ds.take(1):
    images, labels = data["image"], data["label"]

    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    for i in range(16):
        img = images[i]
        img = tf.image.flip_left_right(img)
        img = ops.transpose(img, (1, 0, 2))
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        # Gather indices where label!= padding_token.
        label = labels[i]
        indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
        # Convert to string.
        label = tf.strings.reduce_join(num_to_char(indices))
        label = label.numpy().decode("utf-8")

        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")

    plt.show()


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model():
    # Inputs to the model
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    new_shape = ((image_width // 4), (image_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(
        len(alphabet) + 2, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model


# Get the model.
model = build_model()
model.summary()

validation_images = []
validation_labels = []

# Собираем изображения и метки из validation_ds
for batch in validation_ds:
    validation_images.append(batch["image"])
    validation_labels.append(batch["label"])


def calculate_edit_distance(labels, predictions):
    # Преобразуем метки в sparse формат
    sparse_labels = ops.cast(tf.sparse.from_dense(labels), dtype=tf.int64)
    # Декодируем предсказания
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    decoded_predictions = tf.keras.backend.ctc_decode(
        predictions, input_length=input_len
    )[0][0][:, :max_len]

    # Преобразуем предсказания в sparse формат
    sparse_predictions = ops.cast(tf.sparse.from_dense(decoded_predictions), dtype=tf.int64)

    # Вычисляем расстояние редактирования
    edit_distances = tf.edit_distance(sparse_predictions, sparse_labels, normalize=False)
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )


epochs = 50  # To get good results this should be at least 50.

model = build_model()
prediction_model = keras.models.Model(
    model.get_layer(name="image").output, model.get_layer(name="dense2").output
)
edit_distance_callback = EditDistanceCallback(prediction_model)

# Train the model.
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[edit_distance_callback],
)

model.save("ocr_train_model_ctc.keras")
with open("alphabet.pkl", "wb") as f:
    pickle.dump(alphabet, f)
