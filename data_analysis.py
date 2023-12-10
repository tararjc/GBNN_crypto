# %%
import pandas as pd
import glob
import matplotlib.pyplot as plt


file_names = glob.glob("Dataset/*csv")


rows = []

for file_name in file_names:
    df = pd.read_csv(file_name)

    # Extract the close price of the first date and last date
    first_date_close = df.loc[0, "Close"]
    last_date_close = df.loc[df.index[-1], "Close"]

    # Create a dictionary with the data
    data = {
        "First_Date_Close": first_date_close,
        "Last_Date_Close": last_date_close,
        "File_Name": file_name,
    }

    rows.append(data)

result_df = pd.DataFrame(rows)


df = pd.read_csv("Dataset_des.csv")


df.set_index("Cryptocurrency", inplace=True)

plt.figure(figsize=(20, 8))
df.plot(kind="bar", logy=True)

# Add labels and title
plt.title("Cryptocurrency Comparison (Logarithmic Scale)")
plt.xlabel("Cryptocurrency")
plt.ylabel("Close Price")
plt.tight_layout()
plt.savefig("barplot.png", dpi=700)
