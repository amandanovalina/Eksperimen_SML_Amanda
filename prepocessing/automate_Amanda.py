import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop kolom dengan missing value tinggi
    df = df.drop(columns=['Cabin'])

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encoding
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Drop kolom tidak relevan
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])

    # Scaling
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df['Survived'] = y.values

    processed_df.to_csv(output_path, index=False)