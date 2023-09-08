import tensorflow as tf
import numpy as np
from PIL import Image

img=Image.open("8.png")
img=img.convert('L')
img.show()
img=np.array(img)
img.reshape(28,28)
print(img.shape)

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# img=x_test[59]
# img=img.reshape(28,28)
# img_show(img)


x_train,x_test=x_train/255.0,x_test/255.0

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam",loss=loss_fn,metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=8)
#model.save_weights('myMnisetWeights.h5')
#model.evaluate(x_test,y_test,verbose=2)

model.load_weights('myMnisetWeights.h5')


img = np.expand_dims(img, 0)
predictions = model(img).numpy()
#res=tf.nn.softmax(predictions).numpy()
print(np.argmax(predictions))



