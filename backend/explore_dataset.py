import pandas as pd

# Load dataset
df = pd.read_csv("../data/oer_resources.csv")

# --------------------
# Step 1: Explore
# --------------------
print("🔹 First 5 rows:")
print(df.head())

print("\n🔹 Column names:")
print(df.columns)

print("\n🔹 Shape of dataset:")
print(df.shape)

print("\n🔹 Dataset info:")
print(df.info())

print("\n🔹 Unique subjects:")
print(df['subjects'].unique())

print("\n🔹 Number of resources per subject:")
print(df['subjects'].value_counts())

# --------------------
# Step 2: Clean + Map
# --------------------
def map_domain(subject):
    subject = str(subject).lower()
    if "math" in subject or "algebra" in subject or "geometry" in subject:
        return "Math"
    elif "physics" in subject or "chemistry" in subject or "biology" in subject or "science" in subject:
        return "Science"
    elif "computer" in subject or "programming" in subject or "software" in subject or "data" in subject or "machine learning" in subject:
        return "Programming"
    elif "history" in subject or "geography" in subject or "politics" in subject or "economics" in subject:
        return "History"
    else:
        return "General"

# Apply mapping
df['domain'] = df['subjects'].apply(map_domain)

# Keep useful columns
resources = df[['domain', 'subjects', 'title', 'url', 'type', 'level_availability']]

# Save cleaned dataset
resources.to_csv("../data/resources_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved at data/resources_cleaned.csv")

# Show domain counts
print("\n🔹 Resources per domain:")
print(resources['domain'].value_counts())
