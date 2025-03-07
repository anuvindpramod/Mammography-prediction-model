import pandas as pd
from dataset_2 import MammoDataset
from config import Image_dir,CSV_dir

# Quick validation test
test_df = pd.read_csv(f'{CSV_dir}/mass_case_description_train_set.csv').head(5)
dataset = MammoDataset(test_df, None, Image_dir)

print(f"Valid samples: {len(dataset)}")
for i in range(3):
    img, label = dataset[i]
    print(f"Sample {i}: Shape {img.shape}, Label {label}")
