# Data Folder

This project uses the **Instacart Market Basket Analysis** dataset from Kaggle:
https://www.kaggle.com/c/instacart-market-basket-analysis/data

Place the following files directly inside this `data/` folder (do not rename them):

- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

## Notes

- These files are **not included** in this repository because they are large and
  because Kaggle's license does not allow redistribution.
- After downloading and unzipping, just copy the CSV files into this folder.
- `data/*.csv` is already listed in `.gitignore`, so these files will never be
  accidentally committed to Git.
- Once the files are in place, run the project with:

```bash
python main.py
```

The code automatically checks which files are present and will tell you
clearly if any are missing.
