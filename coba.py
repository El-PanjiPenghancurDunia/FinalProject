import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "CarPrice_Assignment.csv"

# Membaca data
try:
    data = pd.read_csv(file_path)
    print("Dataset berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat dataset dari URL: {e}")


# Tampilkan 5 baris pertama dan info data
print("\n--- 5 Baris Pertama Data ---")
print(data.head())

print("\n--- Informasi Data ---")
data.info()

# Cek nilai yang hilang
print("\n--- Jumlah Nilai Hilang per Kolom ---")
print(data.isnull().sum())
# Jika ada nilai yang hilang, kita dapat mengatasinya, misalnya mengisi dengan rata-rata (imputation)
# Untuk dataset ini, mungkin tidak ada nilai hilang (sesuai info() Anda)

# Mengganti beberapa nama merek yang salah ketik (Contoh dari lab Anda)
data['brand'] = data['CarName'].apply(lambda x: x.split(' ')[0].lower())
data['brand'] = data['brand'].replace(['vw', 'vokswagen'], 'volkswagen')
data['brand'] = data['brand'].replace(['porsche', 'porcshce'], 'porsche')
data.drop(['CarName', 'car_ID'], axis=1, inplace=True) # Hapus kolom yang tidak relevan

print("\nData dibersihkan dan siap untuk EDA...")

# 📈 Visualisasi Distribusi Harga (Sebelum Transformasi)
plt.figure(figsize=(6, 4))
sns.histplot(data['price'], kde=True)
plt.title('Distribusi Harga Jual (Skewed)')
plt.show()

# Transformasi Log pada Variabel Target 'price'
data['price_log'] = np.log(data['price'])

# 📈 Visualisasi Distribusi Harga (Setelah Transformasi Log)
plt.figure(figsize=(6, 4))
sns.histplot(data['price_log'], kde=True)
plt.title('Distribusi Harga Jual (Log-Transformed, Lebih Normal)')
plt.show()

# Hapus kolom 'price' asli dan gunakan 'price_log' sebagai variabel target (y)
data.drop('price', axis=1, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Tentukan X (fitur) dan y (target)
X = data.drop('price_log', axis=1)
y = data['price_log']

# 2. Identifikasi Kolom Numerik dan Kategori
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# 3. Buat Transformer untuk Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # handle_unknown='ignore' menghindari error pada test set

# 4. Gabungkan Transformer dengan ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nUkuran Training Set: {X_train.shape}")
print(f"Ukuran Testing Set: {X_test.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Buat Model Regresi Linear
linear_model = LinearRegression()

# 2. Buat Pipeline: Preprocessing -> Model
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', linear_model)])

# 3. Latih Model (Fit)
# Pipeline akan menjalankan StandardScaler dan OneHotEncoder secara otomatis,
# lalu melatih model regresi pada data yang sudah diproses.
full_pipeline.fit(X_train, y_train)

# 4. Lakukan Prediksi pada Data Test
y_pred = full_pipeline.predict(X_test)

# 5. Evaluasi Model
# Karena kita menggunakan log(price) sebagai y, metriknya juga dalam skala log.

# a. Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE) [log scale]: {mse:.4f}")

# b. Root Mean Squared Error (RMSE) [dapat diinterpretasikan sebagai rata-rata error]
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE) [log scale]: {rmse:.4f}")

# c. R-squared (Koefisien Determinasi)
r2 = r2_score(y_test, y_pred)
print(f"R-squared Score: {r2:.4f}")

print("\n--- Interpretasi Hasil ---")
if r2 > 0.8:
    print(f"Model memiliki R-squared tinggi ({r2:.4f}), menunjukkan model menjelaskan sebagian besar variasi harga mobil.")
else:
    print(f"R-squared ({r2:.4f}) masih dapat ditingkatkan. Perlu eksplorasi fitur dan model lain.")