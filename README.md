# Yapay-Zeka-Deep-Learning
Yapay Zeka -  Deep learning using Keras, Tensorflow

-----------------------------------------------------------

(students-grade-prediction-with-keras-sequential-m.ipynb)

First, we start by importing "os","numpy" and "pandas".

Secondly, we use for loop to traverse every file under the /kaggle/input folder and prints its full path. It’s a quick way to discover what data files (CSVs, images, JSONs, etc.) are available in Kaggle environment. 

Then, we Identify the CSV file and load it with pandas (pd): df = pd.read_csv('')

After that, we use "df.head()" to see the first 5 rows of the DataFrame(df). This lets you verify that your data loaded correctly and inspect its structure at a glance.

Finally, we can start with deep learning. y = df['Scores'] y is being assigned the contents of "Scores" column. y is the target variable(what you’re trying to predict)
Moving your target column (y) into its own variable is a common step in supervised learning. y will serve as the ground truth for model training and evaluation.
Keeping features (X) and target (y) separate makes it easy to split data into training and test sets.


Next, x = df.drop(columns = ['Scores','Grade']) we create a new DataFrame x by dropping the columns Scores and Grade from your original DataFrame (df). You end up with only the predictor variables (features) in x, leaving out the target (Scores) and any other non-features (Grade). X now holds DataFrame without 'Scores' and 'Grade', making it your feature set.

y = y.values.astype('float32') transforms your pandas Series y into a raw NumPy array of type float32. Keras (and many other deep-learning frameworks) expects its inputs and targets to be in contiguous NumPy arrays with a specific numeric dtype for optimal performance. Deep-learning libraries like TensorFlow default to 32-bit floats for a balance of precision and speed.


Deep-learning libraries like TensorFlow default to 32-bit floats for a balance of precision and speed.

x = x.values.astype('float32') "X.values()" extracts the raw NumPy array from the DataFrame. This gives you a plain array of shape (n_samples, n_features)

from keras.models import Sequential (Imports the Sequential class, which lets you stack layers one after another in a linear graph.)
from keras.layers import Dense   (Brings in the Dense layer type—every neuron in one layer is connected to every neuron in the next)
model = Sequential()             (Lastly, create a brand-new empty model instance ready to have layers added.)

model.add(Dense(16)) -This first dense layer has 16 neurons and uses a linear activation. It performs a weighted sum of inputs plus a bias term.

model.add(Dense(32 , activation = "relu"))
model.add(Dense(32 , activation = "relu")) -These two layers each have 32 neurons with ReLU activation, introducing non-linearity and helping the network learn complex relationships.

model.add(Dense(1)) -The final layer outputs a single continuous value. It also defaults to a linear activation.



model.compile(optimizer = "adam" , loss = "mse") -Adam optimizer: adapts learning rates per parameter using estimates of first and second moments (momentum + RMSProp).
-MSE loss: measures average squared difference between predictions and targets—standard for regression.

model.fit(x, y, epochs=128, batch_size=16) -epochs=128: full passes through your dataset. batch_size=16: number of samples processed before the model updates weights.


model.predict(np.array([[5, 1, 0.5, 4, 4.2,]])) - Creates a 2D NumPy array of shape (1, 5), matching the five input features your model expects. Feeds that single-sample array through the network and returns a NumPy array of predictions with shape (1, 1).


------------------------------------------------------------------------


# anemia-type-classification-keras-sequential-model (2).ipynb

df = pd.read_csv('/kaggle/input/anemia-types-classification/diagnosed_cbc_data_v4.csv') - We read the CSV file and turn it into a DataFrame (df).


y=df['Diagnosis'] Accesses the column named "Diagnosis" from the DataFrame (df). This column likely contains the anemia type labels (e.g., Iron Deficiency Anemia etc.).

Assigns this column to the variable y, which is typically used as the target in machine learning tasks. y contains the labels (anemia types)


x=df.drop('Diagnosis',axis=1) Removes the "Diagnosis" column from the DataFrame (df). axis=1 means you're dropping a column (not a row).

Assigns the remaining columns to the variable x, which will be used as the input features.


from sklearn.preprocessing import LabelEncoder **LabelEncoder is used to convert categorical labels (like strings) into numeric values.**

>Machine learning models (especially neural networks) require numerical inputs. So if your target variable y contains string labels, you need to encode them using LabelEncoder.


le=LabelEncoder() **Creates an instance of the LabelEncoder class.**

y=le.fit_transform(y) **Learns the unique categories in y and converts those categories into numbers.**

le.inverse_transform([4]) **Reverse label encoding — converts numeric label back to its original categorical form.**


from tensorflow.keras.utils import to_categorical

**Imports the to_categorical function from Keras, which is used to convert integer labels into one-hot encoded vectors.**

y=to_categorical(y) -Convert your target labels y into one-hot encoded format, which is essential for multi-class classification tasks using neural network.

**After using LabelEncoder, your target labels (y_encoded) are integers like 0, 1, 2. But neural networks often perform better when targets are in one-hot format. That's why it is important to turn it into the one-hot format. y_categorical is can be fed into a Keras model with "categorical_crossentropy" loss.**


-If you did everything right, the output of y[0] should be looking like this:


>array([0., 0., 0., 0., 0., 1., 0., 0., 0.])


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=43)

**Splits your dataset into training and testing sets.**

**y: Target labels (anemia types)**

**x: Features**

**train_size=0.75: 75% of the data goes to training, 25% to testing**

**random_state=43: Ensures reproducibility — the same split every time you run it.**

**x_train, y_train: Used to train your model.**

**x_test, y_test: Used to evaluate model performance**

from tensorflow.keras.models import Sequential **Initializes a Keras Sequential model, which is a linear stack of layers — perfect for building feedforward neural networks.**

model=Sequential() **Sequential(): Creates an empty model where you’ll add layers one by one.**


x.columns **This will return a list of all the column names from your original DataFrame, excluding "Diagnosis" since you dropped it earlier.**


len(x.columns)  📊 Returns the number of columns.

from tensorflow.keras.layers import Dense **Imports the Dense layer from Keras, which is a fully connected neural network layer (essential for building Sequential model.)**

**Each neuron in a Dense layer receives input from every neuron in the previous layer.**

model.add(Dense(64,activation='relu',input_dim=len(x.columns))) **Adding the first hidden layer to neural network.**

>64 units: Number of neurons
>
>activation: Activation function (e.g., 'relu')
>
>input_dim: Shape of input data. It tells Keras how many features to expect in each input sample.

***The first Dense layer processes that input and applies weights, biases, and an activation function (like 'relu').***

**This first layer will learn patterns from the data to help distinguish between anemia types.**

model.add(Dense(32, activation='relu'))  **Adds a hidden layer with 32 neurons and ReLU activation.**

model.add(Dense(32, activation='relu')) **This layer learns intermediate features from the previous layer.**

model.add(Dense(9, activation='softmax')) **Output layer with 9 neurons (for 9 anemia types).**


**'softmax' activation converts outputs into probabilities summing to 1.**


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) -Prepares your Keras model for training by specifying how it should learn and evaluate performance.

>optimizer='adam': Uses the Adam optimizer — fast, adaptive, and great for most tasks.
>
>loss='categorical_crossentropy': Ideal for multi-class classification when your labels are one-hot encoded.
>
>metrics=['accuracy']: Tracks accuracy during training and evaluation.



model.compile() step tells Keras:

How to adjust weights **(optimizer)**

What error to minimize **(loss)**

What metric to report **(accuracy)**

**Specifies the loss function for multi-class classification when your labels are one-hot encoded.**

**Adam optimizer adapts learning rates during training for faster convergence. It’s a great default choice.**



model.fit(x_train,y_train,epochs=128,batch_size=32,validation_split=0.24)  **.fit() is training Keras model on the anemia dataset.**

model.fit()**This method trains your neural network using the training data (x_train, y_train). It adjusts the model’s weights to minimize the loss function you defined earlier.**


--------------------------------------------------------------------



