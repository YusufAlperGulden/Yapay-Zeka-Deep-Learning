# Yapay-Zeka-Deep-Learning
Yapay Zeka -  Deep learning using Keras, Tensorflow

First, we start by importing "os","numpy" and "pandas".

Secondly, we use for loop to traverse every file under the /kaggle/input folder and prints its full path. It’s a quick way to discover what data files (CSVs, images, JSONs, etc.) are available in Kaggle environment. 

Then, we Identify the CSV file and load it with pandas (pd): df = pd.read_csv('')

After that, we use "df.head()" to see the first 5 rows of the DataFrame(df). This lets you verify that your data loaded correctly and inspect its structure at a glance.

Finally, we can start with deep learning. y = df['Scores']
