from keras.models import Model, Input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenetv2 import MobileNetV2


def create_model():
    mobilenet_model = MobileNetV2(weights='imagenet', include_top=False)
    input_tensor = Input((224, 224, 3))
    tensor = mobilenet_model(input_tensor)
    tensor = GlobalAveragePooling2D()(tensor)
    tensor = Dense(512, activation="relu")(tensor)
    output_tensor = Dense(5, activation="sigmoid")(tensor)
    model = Model(input_tensor, output_tensor)
    return model
