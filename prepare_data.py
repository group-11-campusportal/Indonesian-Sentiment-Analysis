import pandas as pd
import os 

BASE_URL = "https://raw.githubusercontent.com/indobenchmark/indonlu/master/dataset/smsa_doc-sentiment-prosa"
TRAIN_URL = f"{BASE_URL}/train_preprocess.tsv"
VALID_URL = f"{BASE_URL}/valid_preprocess.tsv"
TEST_URL  = f"{BASE_URL}/test_preprocess.tsv"

print("Downloading and loading IndoNLU SmSA dataset...")

train_df = pd.read_csv(TRAIN_URL, sep="\t", header=None, names=["text", "label"])
val_df   = pd.read_csv(VALID_URL, sep="\t", header=None, names=["text", "label"])
test_df  = pd.read_csv(TEST_URL,  sep="\t", header=None, names=["text", "label"])

# Clean outer edge whitespace while preserving internal casing and words
train_df["text"] = train_df["text"].astype(str).str.strip()
val_df["text"]   = val_df["text"].astype(str).str.strip()
test_df["text"]  = test_df["text"].astype(str).str.strip()

#automated leakage prevention (test set is protected)

test_set =set(test_df["text"])
val_set = set(val_df["text"])

#identify my training samples that match test or validation samples
leaked_in_train = train_df[train_df["text"].isin(test_set) | train_df["text"].isin(val_set)]

if len(leaked_in_train) >0: 
    print(f">> Detected {len(leaked_in_train)} overlapping samples in Train. Automatically removing them...")
    train_df = train_df[~train_df["text"].isin(test_set) &  ~train_df ["text"].isin(val_set)].reset_index(drop=True)
    print(f">> Cleaned Train count : {len(train_df)}")
else: 
     print(">> Audit Passed: 0 overlapping samples. No deduplication required.")

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

dist_table = pd.DataFrame({
    "Train Count": train_df["label"].value_counts(),
    "Train (%)":   (train_df["label"].value_counts(normalize=True) * 100).round(2),
    "Val Count":   val_df["label"].value_counts(),
    "Val (%)":     (val_df["label"].value_counts(normalize=True) * 100).round(2),
    "Test Count":  test_df["label"].value_counts(),
    "Test (%)":    (test_df["label"].value_counts(normalize=True) * 100).round(2),
})
print(dist_table)

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)
test_df.to_csv("data/test_split.csv", index=False)
dist_table.to_csv("data/class_distribution.csv")
print("Saved split CSV files to 'data/' folder successfully!")

print("\nTrain sample preview:")
print(train_df.head(3))

print("\nClass distribution in Train:")
print(train_df["label"].value_counts())