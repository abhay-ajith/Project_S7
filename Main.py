import numpy as np
import os
import cv2
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 29

train_path='Dataset\cityscapes_data\train'
val_path='Dataset\cityscapes_data\val'
train_images=[]
train_masks=[]
val_images=[]
val_masks=[]


color_palette= np.array([
    [0, 0, 0],
    [111, 74, 0],
    [81, 0, 81],
    [128, 64, 128],
    [244, 35, 232],
    [250, 170, 160],
    [230, 150, 140],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [180, 165, 180],
    [150, 100, 100],
    [150, 120, 90],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 0, 90],
    [0, 0, 110],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    ])


def getlabel(img):
    """
    turn a 3 channel RGB image to 1 channel index image
    """
    height, width, ch = img.shape
    m_lable = np.zeros((height, width, 1), dtype=np.uint8)
    for w in range(width):
        for h in range(height):
            b,g,r=img[h, w, :]
            m_lable[h,w,:]=np.argmin(np.linalg.norm(np.array([r,g,b])-color_palette,axis=1),axis=0)
    return m_lable

    return one_hot_mask.numpy()


def load_images(path,MAX):
    temp_img,temp_masks=[],[]
    images=glob(os.path.join(path,'*.jpg'))
    count = 0
    for i in tqdm(images):
        if count<MAX:
            i = cv2.imread(i)
            img = i[:, :256]
            img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            msk = i[:, 256:]
            label = getlabel(msk)
            temp_masks.append(label)
            temp_img.append(img)
            count+=1
    return np.array(temp_img),np.array(temp_masks)


def data_generator(image, mask):
    dataset = tf.data.Dataset.from_tensor_slices((image, mask))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

train_images,train_masks=load_images(train_path,MAX=500)
val_images,val_masks=load_images(val_path,MAX=50)


train_dataset=data_generator(train_images,train_masks)
val_dataset=data_generator(val_images,val_masks)
print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)

history = model.fit(train_dataset,validation_data=val_dataset, epochs=5)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.savefig("accuracy.png")

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.savefig("val_loss.png")

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.savefig("val_accuracy.png")
model.save("my_model.h5")