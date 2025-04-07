from Library import *
from Load_data import *
from Img_generator import *
from Model import *

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') and logs.get('accuracy')>=0.90) :
            print('\n reached 90 % accuracy so counceling training')
            self.model.stop_training = True

callback = MyCallback()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(train_generator, epochs=300,batch_size = 16, validation_data=test_generator)#, callbacks=[callback])
model.save("/kaggle/working/save_model.h5")