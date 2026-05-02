import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import knn

# Membaca dataset dari file Excel
df = pd.read_excel('Feature.xlsx')

# Memilih fitur dan label
X = df[['Average Red', 'Average Green', 'Average Blue']].values
y = df['Label'].values

# Mengonversi label kategori ke nilai numerik
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Membagi dataset menjadi training set dan tst set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1234)

# Visualisasi dataset (opsional, untuk 2D scatter plot, kita gunakan dua fitur pertama)
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # Warna yang sesuai untuk label numerik
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap=cmap, edgecolor='k', s=20)
plt.xlabel('Average Red')
plt.ylabel('Average Green')
plt.title('Scatter plot of the dataset')
plt.show()

# Inisialisasi dan pelatihan model KNN
clf = knn(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Menampilkan prediksi
print("Predictions:", predictions)

# Menghitung dan menampilkan akurasi
acc = np.sum(predictions == y_test) / len(y_test)
print("Accuracy:", acc)
