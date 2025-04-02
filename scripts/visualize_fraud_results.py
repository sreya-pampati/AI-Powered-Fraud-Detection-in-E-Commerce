import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------
# 1. Load CSV results
# ----------------------------
csv_path = os.path.join(os.getcwd(), 'fraud_batch_results.csv')
df = pd.read_csv(csv_path)

# ----------------------------
# Setup static folder path
# ----------------------------
static_dir = os.path.join(os.getcwd(), 'static')
os.makedirs(static_dir, exist_ok=True)

# ----------------------------
# 2. Basic Summary
# ----------------------------
total = len(df)
fraud_count = df['Fraud Detected'].sum()
safe_count = total - fraud_count

print(f"📊 Total Sequences: {total}")
print(f"🚨 Fraudulent: {fraud_count}")
print(f"✅ Safe: {safe_count}")

# ----------------------------
# 3. Pie Chart - Fraud vs Safe
# ----------------------------
plt.figure(figsize=(5, 5))
plt.pie(
    [safe_count, fraud_count],
    labels=['Safe', 'Fraud'],
    autopct='%1.1f%%',
    colors=['#66c2a5', '#fc8d62'],
    startangle=90
)
plt.title('Fraud Distribution')
plt.savefig(os.path.join(static_dir, 'fraud_pie_chart.png'))
plt.close()

# ----------------------------
# 4. Histogram - Fraud Score Distribution
# ----------------------------
plt.figure(figsize=(8, 4))
sns.histplot(df['Max Fraud Score'], bins=20, kde=True, color='purple')
plt.title('Distribution of Max Fraud Scores')
plt.xlabel('Max Fraud Score')
plt.ylabel('Frequency')
plt.axvline(x=0.3, color='red', linestyle='--', label='Fraud Threshold (0.3)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'fraud_score_distribution.png'))
plt.close()

# ----------------------------
# 5. GenAI Explanation Stats Pie
# ----------------------------
genai_summary = df['GenAI Explanation'].value_counts()

custom_colors = {
    "⚠️ Highly suspicious behavior with sudden score spike.": "#a93226",
    "⚠️ Unusual activity pattern detected — possibly high-risk.": "#e74c3c",
    "⚠️ Slightly abnormal behavior, worth investigating.": "#FFB6C1",
    "✅ No suspicious behavior detected.": "#aed6f1"
}

labels = genai_summary.index.tolist()
sizes = genai_summary.values.tolist()
colors = [custom_colors[label] for label in labels]

plt.figure(figsize=(8, 8))
patches, texts, autotexts = plt.pie(
    sizes,
    labels=None,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 8}
)

plt.legend(
    patches,
    labels,
    loc='upper left',
    bbox_to_anchor=(1, 0.9),
    fontsize=11
)
plt.title("GenAI Explanation Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'genai_explanation_stats_pie.png'), bbox_inches='tight')
plt.close()

# ----------------------------
# 6. Done
# ----------------------------
print("\n✅ Plots saved to 'static/' folder:")
print("   🥧 fraud_pie_chart.png")
print("   📈 fraud_score_distribution.png")
print("   🤖 genai_explanation_stats_pie.png")
