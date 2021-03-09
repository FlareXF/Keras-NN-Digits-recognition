import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam


def cnn_model_init():
	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=(5,5), activation = 'relu',padding = 'same' ,input_shape = (28,28,1)))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Conv2D(filters=64, kernel_size=(3,3), activation = 'relu',padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(filters=64, kernel_size=(3,3), activation = 'relu',padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(128,activation='relu'))
	model.add(Dense(10,activation='softmax'))
	model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
	return model
def cnn_model_train(model):
	(num_mod_train,lab_mod_train), (num_mod_test,lab_mod_test) = mnist.load_data()
	train_data = np.reshape(num_mod_train,(num_mod_train.shape[0],28,28,1))
	train_data = train_data.astype('float32')/255
	train_labels_cat = keras.utils.to_categorical(lab_mod_train,10)
	val_data = np.reshape(num_mod_test,(num_mod_test.shape[0],28,28,1))
	val_data = val_data.astype('float32')/255
	val_labels_cat = keras.utils.to_categorical(lab_mod_test,10)
	model.fit(train_data, train_labels_cat, epochs=2, batch_size=64, validation_data=(val_data,val_labels_cat))
	return model


model = cnn_model_init()
cnn_model_train(model)
model.save('cnn_digs_ts.h5')