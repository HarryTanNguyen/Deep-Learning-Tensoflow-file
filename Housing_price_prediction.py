import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keral.layers import Dense
from sklearn.preprocessing import MinMaxScaler

#split data to training dataset and test set
train_df = pd.read_csv('https://firebasestorage.googleapis.com/v0/b/bible-project-2365c.appspot.com/o/train.csv?alt=media&token=9c5d17c2-0589-43ea-b992-e7c2ad02d714', index_col='ID')
train_df.head()
test_df = pd.read_csv('https://firebasestorage.googleapis.com/v0/b/bible-project-2365c.appspot.com/o/test.csv?alt=media&token=99688b27-9fdb-4ac3-93b8-fa0e0f4d7540', index_col='ID')
test_df.head()

#Normalize our data
scaler = MinMaxScaler(feature_range=(0, 1))
# Scale both the training inputs and outputs
scaled_train = scaler.fit_transform(train_df)
#scaled_test = scaler.transform(test_df)
multiplied_by = scaler.scale_[13]
added = scaler.min_[13]
scaled_train_df = pd.DataFrame(scaled_train, columns=train_df.columns.values)


#build our model
model = Sequential([tf.keras.Dense(50, activation='relu'),
                    tf.keras.Dense(100, activation='relu'),
                    tf.keras.Dense(50, activation='relu'),
                    tf.keras.Dense(1)])

model.compile(loss='mean_squared_error', optimizer='adam')

X = scaled_train_df.drop(target, axis=1).values
Y = scaled_train_df[[target]].values

# Train the model
model.fit(
    X[10:],
    Y[10:],
    epochs=50,
    shuffle=True,
    verbose=2
)

#prediction
prediction = model.predict(X[:1])
y_0 = prediction[0][0]
print('Prediction with scaling - {}',format(y_0))
y_0 -= added
y_0 /= multiplied_by
print("Housing Price Prediction  - ${}".format(y_0))
