import pandas as pd
from datetime import date

# Path of the file to read
iowa_file_path = 'resources/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Print summary statistics in next line
print(home_data.describe())

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = round(home_data["LotArea"].mean())
print(avg_lot_size)

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = (date.today().year - home_data["YearBuilt"]).min()
print(newest_home_age)
