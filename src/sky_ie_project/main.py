import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Plotting the line of total demand over time regardless of the product and the city
def total_demand_plot():
    # Connect to the SQLite database
    conn = sqlite3.connect("data/demand_train.db")
    
    # Query to aggregate total demand by date from the database
    query = """SELECT date, SUM(demand) as total_demand
    FROM demand_train
    GROUP BY date
    ORDER BY date ASC
    """
    
    # Read the query result into a DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Close the database connection
    conn.close()
    
    # Convert 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Plotting the demand data
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["total_demand"], color="blue", linewidth=1)

    # Formatting the x-axis for better readability
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    # Adding labels and title
    plt.xlabel("Date")
    plt.ylabel("Total Demand")
    plt.title("Monthly Total Demand")
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Plotting the line of total demand over time of the city regardless of the product
def city_demand_plot(which_city):
    # Connect to the SQLite database
    conn = sqlite3.connect("data/demand_train.db")
    
    # Query to aggregate total demand by date from the database
    query = """SELECT date, SUM(demand) as total_demand
    FROM demand_train
    WHERE city = "Montreal"
    GROUP BY date
    ORDER BY date ASC
    """
    
    # Read the query result into a DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Close the database connection
    conn.close()
    
    # Convert 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Plotting the demand data
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["total_demand"], color="blue", linewidth=1)

    # Formatting the x-axis for better readability
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    # Adding labels and title
    plt.xlabel("Date")
    plt.ylabel("Total Demand")
    plt.title("Monthly Total Demand")
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()



def main():
    #total_demand_plot()
    city_demand_plot("Montreal")
    


if __name__ == "__main__":
    main()
