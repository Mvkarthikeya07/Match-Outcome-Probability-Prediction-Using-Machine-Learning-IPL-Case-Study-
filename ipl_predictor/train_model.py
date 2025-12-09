# train_model.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/matches.csv"   # update path if needed
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def try_read_csv_with_encodings(path, encodings=("utf-8","latin1","cp1252")):
    """Try reading CSV file with several encodings. Return DataFrame or raise last exception."""
    last_exc = None
    for enc in encodings:
        try:
            print(f"Trying to read CSV with encoding={enc} ...")
            df = pd.read_csv(path, encoding=enc)
            print(f"Read succeeded with encoding={enc}")
            return df
        except Exception as e:
            last_exc = e
            print(f"Failed with encoding={enc}: {e}")
    raise last_exc

def load_matches(path):
    """
    Robust loader: try CSV with multiple encodings, then fall back to Excel.
    Raises a friendly error if all attempts fail.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    # If file is actually an Excel workbook but named .csv, try reading as CSV first then Excel.
    try:
        df = try_read_csv_with_encodings(path, encodings=("utf-8","latin1","cp1252"))
        return df
    except Exception as csv_exc:
        print("CSV read failed, trying to read as Excel file (.xls/.xlsx)...")
        try:
            df = pd.read_excel(path)
            print("Read succeeded as Excel file.")
            return df
        except Exception as excel_exc:
            # provide helpful combined error
            raise RuntimeError(
                "Failed to read data file as CSV (utf-8/latin1/cp1252) and as Excel.\n"
                f"CSV error: {csv_exc}\nExcel error: {excel_exc}\n"
                "Possible causes: file is binary/not a table, wrong path, or unusual encoding.\n"
                "Solutions: (1) open file in Excel and save as CSV UTF-8, (2) pass correct encoding, "
                "(3) move file to data/matches.csv and re-run."
            )

def clean_and_prepare(df):
    # show columns to help debug
    print("Columns found in file:", list(df.columns))
    # normalize expected columns
    for c in ["team1","team2","toss_winner","winner","city","venue","toss_decision"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # Ensure toss_decision exists
    if "toss_decision" not in df.columns:
        df["toss_decision"] = "field"
    df["toss_decision"] = df["toss_decision"].str.lower().map({"bat":1,"field":0}).fillna(0).astype(int)

    # Filter rows where winner is one of playing teams (reduce noise)
    try:
        df = df[df["winner"].isin(pd.concat([df["team1"], df["team2"]]).unique())]
    except Exception:
        # if team1/team2/winner missing or malformed, just dropna on essentials
        df = df.dropna(subset=["team1","team2","winner"])

    return df

def create_features_and_encoders(df):
    features = ["team1","team2","toss_winner","toss_decision","city"]
    # ensure feature columns exist
    for f in features:
        if f not in df.columns:
            df[f] = "Unknown"

    X = df[features].copy()
    y = df["winner"].astype(str).copy()

    encoders = {}
    # encode categorical features
    for col in ["team1","team2","toss_winner","city"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y)
    encoders["label"] = label_enc

    return X, y_enc, encoders

def train_and_save(df):
    X, y, encoders = create_features_and_encoders(df)

    # Small datasets may cause stratify issues; use a simple split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(clf, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    print("Saved model and encoders to", MODEL_DIR)

if __name__ == "__main__":
    print("Loading data from:", DATA_PATH)
    df = load_matches(DATA_PATH)
    print("Loaded rows:", len(df))
    df = clean_and_prepare(df)
    print("After cleaning, rows:", len(df))
    train_and_save(df)
