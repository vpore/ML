import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'marks 1': [85, 76, 90, 65, 88, None, 78, 92, 85, 7000],
    'marks 2': [88, None, 94, 80, 87, 79, None, 98, 91, 7500],
    'grade': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'A', 'A', 'B']
}

df = pd.DataFrame(data)

imp = SimpleImputer(strategy='median')
df[['marks 1', 'marks 2']] = imp.fit_transform(df[['marks 1', 'marks 2']])
print("\nImputation - \n", df)

outlier_detector = IsolationForest(contamination=0.05)
df['is_outlier'] = outlier_detector.fit_predict(df[['marks 1', 'marks 2']])
print("\nAnomaly Detection - \n", df)
df = df[df['is_outlier'] == 1] 
df.drop(columns=['is_outlier'], inplace=True)

scaler = StandardScaler()
df[['marks 1', 'marks 2']] = scaler.fit_transform(df[['marks 1', 'marks 2']])
print("\nStandardization - \n", df)

scaler = MinMaxScaler()
df[['marks 1', 'marks 2']] = scaler.fit_transform(df[['marks 1', 'marks 2']])
print("\nNormalization - \n", df)

encoder = OneHotEncoder()
encoded_features = pd.DataFrame(encoder.fit_transform(df[['grade']]).toarray())
df = pd.concat([df, encoded_features], axis=1)
print("\nEncoding - \n", df)