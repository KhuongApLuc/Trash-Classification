from Library import *
from Load_data import *
from Img_generator import *

# Hàm xây dựng Connector 1
def connector_1(input_tensor):
    x = layers.Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding="same")(input_tensor)
    x = layers.AveragePooling2D(pool_size=(1, 1))(input_tensor)
    x = layers.Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([input_tensor, x])

# Hàm xây dựng Connector 2
def connector_2(input_tensor):
    x = layers.Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding="same")(input_tensor)
    x = layers.Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([input_tensor, x])

# Feature Extraction Module
def feature_extraction_module(input_tensor):
    #Block1
    x = layers.Conv2D(16, (7, 7), padding='same', activation=None)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(16, (7, 7), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.AveragePooling2D((2, 2))(x)  # 224x224 -> 112x112
    #Block2
    x = layers.Conv2D(32, (5, 5), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.AveragePooling2D((2, 2))(x)
    
    x = connector_1(x)  # Thêm connector 1
    #Block3
    x = layers.Conv2D(64, (4, 4), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.AveragePooling2D((2, 2))(x)
    
    x = connector_1(x)  # Thêm connector 1
    #Block4
    x = layers.Conv2D(128, (3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.AveragePooling2D((2, 2))(x)
    
    x = connector_1(x)  # Thêm connector 1
    #Block5
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(6, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = connector_2(x)  # Thêm connector 2

    return x

# Classification Module
def classification_module(input_tensor, num_classes=6):
    x = layers.GlobalAveragePooling2D()(input_tensor)
    x = layers.Dense(num_classes, activation="softmax")(x)
    return x

# Xây dựng mô hình tổng thể
def build_model(input_shape=(224, 224, 3), num_classes=6):
    inputs = layers.Input(shape=input_shape)
    x = feature_extraction_module(inputs)
    outputs = classification_module(x, num_classes=num_classes)
    model = models.Model(inputs, outputs)
    return model

# Tạo và kiểm tra mô hình
model = build_model()
model.summary()