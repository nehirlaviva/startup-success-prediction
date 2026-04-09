import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 1. Load and Clean
df = pd.read_csv('yc_companies.csv')
df_train = df[df['status'].isin(['Acquired', 'Public', 'Inactive'])].copy()
df_train['Target'] = df_train['status'].apply(lambda x: 1 if x in ['Acquired', 'Public'] else 0)

df_train['founder_count'] = df_train['active_founders'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
df_train['has_website'] = df_train['website'].notnull().astype(int)
df_train['has_logo'] = df_train['logo'].notnull().astype(int)
df_train['missing_fields_count'] = df_train.isnull().sum(axis=1)
df_train['team_size'] = df_train['team_size'].fillna(df_train['team_size'].median())

# 2. Market Signal Score (using MAX to capture Peak Momentum)
df['combined_text'] = df['tags'].fillna('') + " " + df['short_description'].fillna('') + " " + df['long_description'].fillna('')
df_train['combined_text'] = df_train['tags'].fillna('') + " " + df_train['short_description'].fillna('') + " " + df_train['long_description'].fillna('')

def extract_year(batch):
    if pd.isna(batch): return 0
    batch = str(batch)
    if len(batch) >= 3 and batch[1:].isdigit():
        yr = int(batch[1:3])
        return 2000 + yr if yr < 50 else 1900 + yr
    return 0

df['batch_year'] = df['batch'].apply(extract_year)
df_train['batch_year'] = df_train['batch'].apply(extract_year)

df_nlp = df[df['batch_year'] >= 2005]
years = sorted(df_nlp['batch_year'].unique())

yearly_word_freq = {}
for year in years:
    text_data = df_nlp[df_nlp['batch_year'] == year]['combined_text'].tolist()
    if not text_data: continue
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', lowercase=True, stop_words='english', max_features=800)
    try:
        tfidf_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        yearly_word_freq[year] = dict(zip(feature_names, avg_tfidf))
    except ValueError:
        pass

yearly_word_scores = {}
for current_year in years:
    word_stats = []
    current_freqs = yearly_word_freq.get(current_year, {})
    past_years = [y for y in years if y < current_year and y >= current_year - 3]
    
    for word, tfidf in current_freqs.items():
        aagr = 0
        if len(past_years) > 0:
            growth_rates = []
            for py in past_years:
                past_tfidf = yearly_word_freq.get(py, {}).get(word, 0)
                growth = (tfidf - past_tfidf) / (past_tfidf + 0.0001)
                growth_rates.append(growth)
            aagr = np.mean(growth_rates)
        word_stats.append({'word': word, 'tfidf': tfidf, 'aagr': aagr})
        
    if not word_stats: continue
        
    stats_df = pd.DataFrame(word_stats)
    median_tfidf = stats_df['tfidf'].median()
    median_aagr = stats_df['aagr'].median()
    
    score_dict = {}
    for _, row in stats_df.iterrows():
        w, t, a = row['word'], row['tfidf'], row['aagr']
        if t < median_tfidf and a > median_aagr: score = 3
        elif t >= median_tfidf and a > median_aagr: score = 2
        elif t >= median_tfidf and a <= median_aagr: score = 1
        else: score = 0
        score_dict[w] = score
    yearly_word_scores[current_year] = score_dict

def get_startup_score(row):
    year = row['batch_year']
    text = str(row['combined_text']).lower().split()
    scores = [yearly_word_scores.get(year, {}).get(w) for w in text if yearly_word_scores.get(year, {}).get(w) is not None]
    return np.max(scores) if scores else 0  # <--- np.max is the key!

df_train['Market_Signal_Score'] = df_train.apply(get_startup_score, axis=1)

# 3. Model Training (With SMOTE)
le_ind = LabelEncoder()
df_train['industry_encoded'] = le_ind.fit_transform(df_train['industry'].fillna('Unknown'))

base_features = ['founder_count', 'team_size', 'has_website', 'has_logo', 'missing_fields_count', 'industry_encoded']
aug_features = base_features + ['Market_Signal_Score']

X_base = df_train[base_features]
X_aug = df_train[aug_features]
y = df_train['Target']

X_base_train, X_base_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
X_aug_train, X_aug_test, _, _ = train_test_split(X_aug, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_base_train_smote, y_train_smote = smote.fit_resample(X_base_train, y_train)
X_aug_train_smote, _ = smote.fit_resample(X_aug_train, y_train)

rf_base = RandomForestClassifier(random_state=42).fit(X_base_train_smote, y_train_smote)
rf_aug = RandomForestClassifier(random_state=42).fit(X_aug_train_smote, y_train_smote)

xgb_base = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_base_train_smote, y_train_smote)
xgb_aug = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_aug_train_smote, y_train_smote)

f1_rf_base = f1_score(y_test, rf_base.predict(X_base_test))
f1_rf_aug = f1_score(y_test, rf_aug.predict(X_aug_test))
f1_xgb_base = f1_score(y_test, xgb_base.predict(X_base_test))
f1_xgb_aug = f1_score(y_test, xgb_aug.predict(X_aug_test))

print("--- MODEL BENCHMARKING (F1 SCORE) ---")
print(f"Random Forest - Baseline: {f1_rf_base:.4f} | Augmented: {f1_rf_aug:.4f}")
print(f"XGBoost       - Baseline: {f1_xgb_base:.4f} | Augmented: {f1_xgb_aug:.4f}")

# =========================================================
# --- PHASE 4: MARKET SCORE ANALYSIS (TRUTH TABLE) ---
# =========================================================
print("\nCalculating Market Score Analysis...")

# 1. Round the scores to integers (0, 1, 2, 3) to group them
df_train['Score_Rounded'] = df_train['Market_Signal_Score'].round().astype(int)

# 2. Build the Truth Table
validation_table = df_train.groupby('Score_Rounded').agg(
    Total_Startups=('Target', 'count'),
    Successful_Startups=('Target', 'sum')
)

# 3. Calculate the exact Success Rate percentage
validation_table['Success_Rate_%'] = (validation_table['Successful_Startups'] / validation_table['Total_Startups'] * 100).round(2)

print("\n--- THE TRUTH TABLE (SUCCESS RATE BY SCORE) ---")
print(validation_table)

# 4. Automatically identify the highest success rate
rates = validation_table['Success_Rate_%'].to_dict()
best_score = max(rates, key=rates.get)

print("\n--- THESIS CONCLUSION ---")
for score, rate in rates.items():
    print(f"Score {score}: {rate}% Success Rate")

print(f"\n🏆 The highest success rate belongs to Score {best_score} ({rates[best_score]}%)!")