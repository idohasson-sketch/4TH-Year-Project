import pandas as pd

# Load the CSV file you already generated
csv_filename = 'mock_wildlife_data.csv'
df = pd.read_csv(csv_filename)

# Start building the SQL script with correct double quotes for case-sensitive table and columns
sql_script = 'INSERT INTO "Observations" (\n'
sql_script += '    "observations_id", "row_created_at", "birds species", "location", "amount", "healthy", "eventtime"\n'
sql_script += ') VALUES\n'

rows = []
for _, row in df.iterrows():
    # Convert healthy boolean to SQL standard TRUE/FALSE strings
    healthy_val = 'TRUE' if str(row['healthy']).lower() in ['true', '1', 't'] else 'FALSE'
    
    # Escape single quotes in location text just in case
    loc_escaped = str(row['location']).replace("'", "''")
    species_escaped = str(row['birds species']).replace("'", "''")
    
    row_str = (
        f"    ({row['observations_id']}, '{row['row_created_at']}', '{species_escaped}', "
        f"'{loc_escaped}', {row['amount']}, {healthy_val}, '{row['eventtime']}')"
    )
    rows.append(row_str)

# Join all rows with commas and close with a semicolon
sql_script += ",\n".join(rows) + ";"

# Save to a text file
output_sql_file = 'insert_direct_to_db.sql'
with open(output_sql_file, 'w', encoding='utf-8') as f:
    f.write(sql_script)

print(f"Success! Open '{output_sql_file}', copy everything, and paste into Supabase SQL Editor.")
