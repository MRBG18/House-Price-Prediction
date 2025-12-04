import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

# Base features
sqft = np.random.normal(1500, 300, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 5, n)
age = np.random.randint(0, 40, n)
floors = np.random.randint(1, 3, n)
lot_size = np.random.normal(4000, 800, n)
garage = np.random.randint(0, 3, n)
distance_city = np.random.uniform(1, 30, n)
crime_index = np.random.uniform(1, 10, n)
school_rating = np.random.uniform(1, 10, n)
public_transport = np.random.randint(0, 2, n)

# Linear relationship
price = (
    200 * sqft
    + 15000 * bedrooms
    + 12000 * bathrooms
    - 800 * age
    + 5000 * floors
    + 3 * lot_size
    + 10000 * garage
    - 3000 * distance_city
    - 2000 * crime_index
    + 4000 * school_rating
    + 500 * public_transport
    + np.random.normal(0, 10000, n)
)

# Introduce nulls randomly
df = pd.DataFrame({
    "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms, "age": age,
    "floors": floors, "lot_size": lot_size, "garage": garage,
    "distance_city": distance_city, "crime_index": crime_index,
    "school_rating": school_rating, "public_transport": public_transport,
    "price": price
})

for col in df.columns:
    df.loc[df.sample(frac=0.02).index, col] = np.nan  # 2% nulls

# Introduce outliers
df.loc[df.sample(20).index, "sqft"] *= 3
df.loc[df.sample(20).index, "price"] *= 3

# Save CSV
path = "dataset.csv"
df.to_csv(path, index=False)

path
