# ==========================================
# ASSOCIATION RULE MINING FOR CROP DATA
# ==========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# 2. LOAD DATASET
df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")
df.columns = df.columns.str.strip()

# 3. USE ONLY ORIGINAL COLUMNS
df = df[["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH_Value", "Rainfall", "Crop"]]

# 4. CONVERT NUMERIC COLUMNS INTO LOW / MEDIUM / HIGH
for col in ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH_Value", "Rainfall"]:
    df[col] = pd.qcut(df[col], q=3, labels=[f"{col}_Low", f"{col}_Medium", f"{col}_High"])

# Convert Crop into item-style labels
df["Crop"] = "Crop_" + df["Crop"].astype(str)

# 5. ONE-HOT ENCODING
df_encoded = pd.get_dummies(df)

print("Encoded Data Shape:", df_encoded.shape)

# 6. FREQUENT ITEMSET GENERATION
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

print("\n===== FREQUENT ITEMSETS =====")
print(frequent_itemsets.sort_values(by="support", ascending=False).head(20))

# 7. GENERATE ASSOCIATION RULES
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Sort by lift
rules = rules.sort_values(by="lift", ascending=False)

print("\n===== ASSOCIATION RULES =====")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20))

# 8. SAVE RULES TO CSV
rules.to_csv("Association_Rules_Output.csv", index=False)
print("\nRules saved as Association_Rules_Output.csv")

# 9. PERFORMANCE MEASURES
print("\n===== ARM PERFORMANCE =====")
print("Number of Frequent Itemsets:", len(frequent_itemsets))
print("Number of Rules Generated :", len(rules))

# Average metrics
print("Average Support   :", rules["support"].mean())
print("Average Confidence:", rules["confidence"].mean())
print("Average Lift      :", rules["lift"].mean())

# 10. GRAPH - TOP 10 RULES BY LIFT
top_rules = rules.head(10)

plt.figure(figsize=(10,5))
plt.bar(range(len(top_rules)), top_rules["lift"])
plt.xticks(range(len(top_rules)), [f"Rule {i+1}" for i in range(len(top_rules))], rotation=45)
plt.ylabel("Lift")
plt.title("Top 10 Association Rules by Lift")
plt.tight_layout()
plt.show()

# 11. GRAPH - SUPPORT VS CONFIDENCE
plt.figure(figsize=(8,5))
plt.scatter(rules["support"], rules["confidence"])
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")
plt.show()