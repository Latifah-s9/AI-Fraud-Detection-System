from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
df = fetch_openml('creditcard', version=1, as_frame=True).frame
df['Class'] = df['Class'].astype(int)
X = df.drop(columns=['Class'])
y = df['Class']
if 'Time' in X.columns:
    scaler = StandardScaler()
    X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])
else:
    scaler = StandardScaler()
    X[['Amount']] = scaler.fit_transform(X[['Amount']].values.reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pos = (y_train==1).sum()
neg = (y_train==0).sum()
spw = neg / max(1, pos)
model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', tree_method='hist', n_jobs=-1, scale_pos_weight=spw, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')
print("Saved model.pkl")
