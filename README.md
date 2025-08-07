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


anemia-type-classification-keras-sequential-model (2).ipynb

df = pd.read_csv('/kaggle/input/anemia-types-classification/diagnosed_cbc_data_v4.csv') - We read the CSV file and turn it into a DataFrame (df).


y=df['Diagnosis'] Accesses the column named "Diagnosis" from the DataFrame (df). This column likely contains the anemia type labels (e.g., Iron Deficiency Anemia etc.).

Assigns this column to the variable y, which is typically used as the target in machine learning tasks. y contains the labels (anemia types)


x=df.drop('Diagnosis',axis=1) Removes the "Diagnosis" column from the DataFrame (df). axis=1 means you're dropping a column (not a row).

Assigns the remaining columns to the variable x, which will be used as the input features.


from sklearn.preprocessing import LabelEncoder **LabelEncoder is used to convert categorical labels (like strings) into numeric values.**

>Machine learning models (especially neural networks) require numerical inputs. So if your target variable y contains string labels, you need to encode them using LabelEncoder.






