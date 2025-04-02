from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Paths
CSV_PATH = os.path.join(os.getcwd(), '..', 'fraud_batch_results.csv')  # One level up
STATIC_PATH = os.path.join(os.getcwd(), 'static')

@app.route('/')
def dashboard():
    fraud_count = 0
    safe_count = 0
    uploaded = False

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        fraud_count = df['Fraud Detected'].sum()
        safe_count = len(df) - fraud_count
        uploaded = True

        # Fraud vs Safe Bar
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Fraud Detected', data=df, palette='Set2')
        plt.xticks([0, 1], ['Safe', 'Fraud'])
        plt.title('Fraud vs Safe Sequences')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_PATH, 'fraud_vs_safe.png'))
        plt.close()

        # Max Fraud Score Distribution
        plt.figure(figsize=(8, 4))
        sns.histplot(df['Max Fraud Score'], bins=20, kde=True, color='purple')
        plt.title('Distribution of Max Fraud Scores')
        plt.xlabel('Max Fraud Score')
        plt.axvline(x=0.3, color='red', linestyle='--', label='Fraud Threshold (0.3)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_PATH, 'fraud_score_distribution.png'))
        plt.close()

        # Pie Chart
        plt.figure(figsize=(5, 5))
        plt.pie([safe_count, fraud_count],
                labels=['Safe', 'Fraud'],
                autopct='%1.1f%%',
                colors=['#66c2a5', '#fc8d62'],
                startangle=90)
        plt.title('Fraud Distribution (Pie Chart)')
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_PATH, 'fraud_pie_chart.png'))
        plt.close()

    return render_template('dashboard.html',
                           fraud_count=fraud_count,
                           safe_count=safe_count,
                           uploaded=uploaded)

if __name__ == '__main__':
    print("🌐 Flask app running at http://127.0.0.1:5000")
    app.run(debug=True)
