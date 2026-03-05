# 📊 Dataset Instructions

This repository contains dummy sample files `Fake.csv` and `True.csv` in this folder to demonstrate the project's structure without bloating the git history with large data files.

To train the model on the full, real-world dataset, please follow these steps:

1. Go to the [ISOT Fake News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
2. Download the archive.
3. Extract `Fake.csv` and `True.csv` into this `dataset/` directory, overwriting the dummy files.
4. Run the training script from the root directory:
   ```bash
   python src/train_model.py
   ```

The `Fake.csv` and `True.csv` datasets contain thousands of news articles (both title and text) labeled appropriately for Fake News Detection.
