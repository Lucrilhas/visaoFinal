### Bibs
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def rodar_tudo(indice):
    # Configs
    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    execs = {
        1: ["mobilev2_boot", "Dataset", "Sem", 0],
        2: ["mobilev2_ft1", "Dataset", "Sem", 1],
        3: ["mobilev2_ft3", "Dataset", "Sem", 3],
        4: ["mobilev2_ft5", "Dataset", "Sem", 5],

        5: ["mobilev2_boot_contraste", "Dataset_Contraste", "Contraste", 0],
        6: ["mobilev2_boot_noskin", "Dataset_Noskin", "SkinTresh", 0],
        7: ["mobilev2_boot_bordas", "Dataset_Bordas", "Bordas", 0],
        8: ["mobilev2_boot_contrasteBordas", "Dataset_Contraste_Noskin", "Contraste + SkinTresh", 0],
        9: ["mobilev2_boot_contrasteNoskin", "Dataset_Contraste_Bordas", "Contraste + Bordas", 0],

        10: ["mobilev2_ft1_contraste", "Dataset_Contraste", "Contraste", 1],
        11: ["mobilev2_ft1_noskin", "Dataset_Noskin", "SkinTresh", 1],
        12: ["mobilev2_ft1_bordas", "Dataset_Bordas", "Bordas", 1],
        13: ["mobilev2_ft1_contrasteBordas", "Dataset_Contraste_Noskin", "Contraste + SkinTresh", 1],
        14: ["mobilev2_ft1_contrasteNoskin", "Dataset_Contraste_Bordas", "Contraste + Bordas", 1],
    }

    name, ds, pre, fine_tunning = execs[indice]
    print(f"\n\n\t\t -- {execs[indice]} --\n\n")

    lr = 0.001
    n_epochs = 40
    img_width, img_height = 224, 224
    batch_size = 32
    input_shape = (img_width, img_height, 3)

    ds = ['Dataset', 'Dataset_Bordas', 'Dataset_Contraste', 'Dataset_Noskin', 'Dataset_Contraste_Bordas', 'Dataset_Contraste_Noskin'][0]
    train_data_dir = ds + "/train"
    validation_data_dir = ds + "/test"
    test_data_dir = ds + "/valid"

    # Dados
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode="categorical", shuffle=False)
    validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode="categorical", shuffle=False)
    test_generator = test_datagen.flow_from_directory( test_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode="categorical", shuffle=False)

    class_names = list(train_generator.class_indices.keys())
    # Execução
    conv_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    if fine_tunning > 0:
        for layer in conv_base.layers[:-fine_tunning]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(7, activation='softmax')(top_model)

    modelo = Model(name=name, inputs=conv_base.input, outputs=output_layer)

    modelo.compile( optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy',metrics=['accuracy'])

    hist = modelo.fit(
        train_generator,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=validation_generator,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min'), 
            ModelCheckpoint(filepath=f"weights/{name}.keras",save_best_only=True,verbose=1)
        ],
        verbose=1
    )
    pd.DataFrame(hist.history).to_csv(f'weights/hist_{name}.csv', index=False)

    predictions = modelo.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    cm = confusion_matrix(true_classes, predicted_classes)
    loss, accuracy = modelo.evaluate(test_generator, verbose=0)

    respostas = {
        'Pré': pre,
        'Fine-Tunning': fine_tunning,
        'Treino Acurácia': hist.history["accuracy"][-1], 
        'Treino Loss': hist.history["loss"][-1], 
        'Valid Acurácia': hist.history["val_accuracy"][-1], 
        'Valid Loss': hist.history["val_loss"][-1], 
        'Test Acurácia': accuracy, 
        'Test Loss': loss,
    }
    pd.DataFrame(respostas, index=[0]).to_csv(f'weights/res_{name}.csv', index=False)
    pd.DataFrame(
        cm,
        index=['Actual: {}'.format(c) for c in class_names],
        columns=['Predicted: {}'.format(c) for c in class_names]
    ).to_csv(f'weights/cm_{name}.csv')

    print("##########################################################")
    print(f"\n\t\t\tFinalizou --- {indice}\n")
    print("##########################################################")
    
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Run rodar_tudo function with an index.')
    
    # Add an argument for the index
    parser.add_argument('indice', type=int, help='Index to pass to the function.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided index
    rodar_tudo(args.indice)