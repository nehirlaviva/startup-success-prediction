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

# --- PHASE 1: Data Preparation ---
def load_and_clean_data(file_path):
    print("Loading and Cleaning Data...")
    df = pd.read_csv(file_path)
    
    df_train = df[df['status'].isin(['Acquired', 'Public', 'Inactive'])].copy()
    df_train['Target'] = df_train['status'].apply(lambda x: 1 if x in ['Acquired', 'Public'] else 0)
    
    # Feature Engineering
    df_train['founder_count'] = df_train['active_founders'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
    df_train['has_website'] = df_train['website'].notnull().astype(int)
    df_train['has_logo'] = df_train['logo'].notnull().astype(int)
    df_train['missing_fields_count'] = df_train.isnull().sum(axis=1)
    df_train['team_size'] = df_train['team_size'].fillna(df_train['team_size'].median())
    
    return df_train, df

# --- PHASE 2: Market Momentum Scoring ---
def calculate_momentum_scores(df_train, df_full):
    print("Calculating Peak Momentum Scores...")
    
    # Combine ALL text fields for deep analysis
    df_full['combined_text'] = df_full['tags'].fillna('') + " " + df_full['short_description'].fillna('') + " " + df_full['long_description'].fillna('')
    df_train['combined_text'] = df_train['tags'].fillna('') + " " + df_train['short_description'].fillna('') + " " + df_train['long_description'].fillna('')

    def extract_year(batch):
        if pd.isna(batch): return 0
        batch = str(batch)
        if len(batch) >= 3 and batch[1:].isdigit():
            yr = int(batch[1:3])
            return 2000 + yr if yr < 50 else 1900 + yr
        return 0

    df_full['batch_year'] = df_full['batch'].apply(extract_year)
    df_train['batch_year'] = df_train['batch'].apply(extract_year)
    
    df_nlp = df_full[df_full['batch_year'] >= 2005]
    years = sorted(df_nlp['batch_year'].unique())
    
    yearly_word_freq = {}
    for year in years:
        text_data = df_nlp[df_nlp['batch_year'] == year]['combined_text'].tolist()
        if not text_data: continue
        # Optimized to 800 features to filter noise
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
                growth_rates = [(tfidf - yearly_word_freq.get(py, {}).get(word, 0)) / (yearly_word_freq.get(py, {}).get(word, 0) + 0.0001) for py in past_years]
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
        return np.max(scores) if scores else 0 # Peak Momentum

    df_train['Market_Signal_Score'] = df_train.apply(get_startup_score, axis=1)
    return df_train

# --- PHASE 3: Model Training ---
def run_model_experiment(df):
    print("Training Models...")
    le_ind = LabelEncoder()
    df['industry_encoded'] = le_ind.fit_transform(df['industry'].fillna('Unknown'))
    
    base_features = ['founder_count', 'team_size', 'has_website', 'has_logo', 'missing_fields_count', 'industry_encoded']
    aug_features = base_features + ['Market_Signal_Score']
    
    X_base = df[base_features]
    X_aug = df[aug_features]
    y = df['Target']
    
    X_base_train, X_base_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
    X_aug_train, X_aug_test, _, _ = train_test_split(X_aug, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_base_train_smote, y_train_smote = smote.fit_resample(X_base_train, y_train)
    X_aug_train_smote, _ = smote.fit_resample(X_aug_train, y_train)
    
    rf_base = RandomForestClassifier(random_state=42).fit(X_base_train_smote, y_train_smote)
    rf_aug = RandomForestClassifier(random_state=42).fit(X_aug_train_smote, y_train_smote)
    
    xgb_base = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_base_train_smote, y_train_smote)
    xgb_aug = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_aug_train_smote, y_train_smote)
    
    metrics = {
        'Random Forest': {
            'Baseline': f1_score(y_test, rf_base.predict(X_base_test)),
            'Augmented': f1_score(y_test, rf_aug.predict(X_aug_test))
        },
        'XGBoost': {
            'Baseline': f1_score(y_test, xgb_base.predict(X_base_test)),
            'Augmented': f1_score(y_test, xgb_aug.predict(X_aug_test))
        }
    }
    
    print("Models Trained Successfully!")
    return rf_aug, xgb_aug, X_aug_test, y_test, metrics