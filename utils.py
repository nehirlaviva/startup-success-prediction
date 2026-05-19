import pandas as pd
import numpy as np
import spacy
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
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def batch_lemmatize(text_list):
    cleaned_texts = []
    processed_list = [str(text).lower() if pd.notna(text) and text != "" else "" for text in text_list]
    for doc in nlp.pipe(processed_list, batch_size=2000):
        cleaned_texts.append(" ".join([t.lemma_ for t in doc if t.is_alpha and not t.is_stop and t.pos_ in ['NOUN', 'PROPN', 'ADJ']]))
    return cleaned_texts

def load_and_clean_data(file_path):
    print("Loading and Cleaning Data...")
    df = pd.read_csv(file_path)
    
    # Crunchbase specific Target Definition
    valid_statuses = ['acquired', 'closed', 'ipo']
    df_train = df[df['status'].isin(valid_statuses)].copy()
    df_train['Target'] = df_train['status'].apply(lambda x: 1 if x in ['acquired', 'ipo'] else 0)
    
    # Crunchbase Year Extraction
    df_train['founded_year'] = pd.to_datetime(df_train['founded_at'], errors='coerce').dt.year
    df_train = df_train.dropna(subset=['founded_year']).copy()
    df_train['founded_year'] = df_train['founded_year'].astype(int)
    
    df['founded_year'] = pd.to_datetime(df['founded_at'], errors='coerce').dt.year
    df = df.dropna(subset=['founded_year']).copy()
    df['founded_year'] = df['founded_year'].astype(int)

    # drop the "ghost" startups before median imputation
    critical_cols = ['funding_rounds', 'funding_total_usd', 'milestones', 'relationships', 'investment_rounds']
    exist_crit = [c for c in critical_cols if c in df_train.columns]
    df_train['missing_count'] = df_train[exist_crit].isnull().sum(axis=1)
    df_train = df_train[df_train['missing_count'] <= 3].copy()
    df_train.drop(columns=['missing_count'], inplace=True)

    # Crunchbase Numeric Columns (Median Imputation)
    numeric_cols = ['investment_rounds', 'invested_companies', 'funding_rounds', 
                    'funding_total_usd', 'milestones', 'relationships']
    for col in numeric_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna(df_train[col].median())
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
            
    return df_train, df

# --- PHASE 2: Market Momentum Scoring ---
def calculate_momentum_scores(df_train, df_full):
    print("Pre-processing text: Combining, Cleaning, and Lemmatizing (Fast Batch Mode)...")
    
    # Combine
    for dataframe in [df_full, df_train]:
        dataframe['combined_text'] =(
            dataframe['tag_list'].fillna('') + " " + 
            dataframe['overview'].fillna(''))
        
        # Clean
        dataframe['combined_text'] = batch_lemmatize(dataframe['combined_text'].tolist())

    df_full['batch_year'] = df_full['founded_year']
    df_train['batch_year'] = df_train['founded_year']
    
    # TF-IDF 
    df_nlp = df_full[df_full['batch_year'] >= 2005]
    years = sorted(df_nlp['batch_year'].unique())
    
    corporate_jargon = [
        'necessary', 'parameter', 'process', 'optimize', 'provide', 'surface', 
        'solution', 'product', 'service', 'company','dream', 'dress', 'help', 'allow', 
        'classified', 'enable', 'feature', 'include', 'build', 'create', 'offer', 'base',
        'use', 'make', 'new', 'world', 'need', 'way', 'work', 'time', 'high', 'low',
        'app', 'funding', 'transaction', 'london', 'tip', 'appointment', 'kind', 
        'rating', 'hour', 'machine', 'city', 'month', 'year', 'day', 'week', 
        'startup', 'round', 'investment', 'investor', 'market', 'application', 
        'network', 'fund', 'europe', 'america', 'sale', 'price', 'cost', 'revenue', 'growth',
        'creator', 'accessible', 'friend', 'family'
    ]
    
    yearly_word_freq = {}
    for year in years:
        text_data = df_nlp[df_nlp['batch_year'] == year]['combined_text'].tolist()
        if not text_data: continue
        
        vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b[a-zA-Z]{4,}\b', 
            lowercase=True, 
            max_features=1000,
            ngram_range=(1, 2),
            min_df=0.001,
            max_df=0.05,           
            stop_words=corporate_jargon 
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(text_data)
            feature_names = vectorizer.get_feature_names_out()
            avg_tfidf = tfidf_matrix.mean(axis=0).A1
            yearly_word_freq[year] = dict(zip(feature_names, avg_tfidf))
        except ValueError:
            pass
        
    yearly_word_scores = {}
    
    # Sensitivity parameter to prevent division by zero for genuinely novel words that appear in the current year but not in the past
    epsilon = 1e-4 
    
    for current_year in years:
        word_stats = []
        current_freqs = yearly_word_freq.get(current_year, {})
        
        # Define the sequential window ending in current_year (3-year lookback + current year)
        window_years = [y for y in years if current_year - 3 <= y <= current_year]
        
        for word, tfidf in current_freqs.items():
            aagr = 0
            
            if len(window_years) > 1:
                yoy_growth_rates = []
                
                # Calculate step-by-step Year-over-Year (YoY) growth
                for i in range(1, len(window_years)):
                    prev_y = window_years[i-1]
                    curr_y = window_years[i]
                    
                    val_prev = yearly_word_freq.get(prev_y, {}).get(word, 0)
                    val_curr = yearly_word_freq.get(curr_y, {}).get(word, 0)
                    
                    # Standard YoY formula: (Current - Past) / Past
                    # Epsilon prevents ZeroDivisionError for genuinely novel words
                    rate = (val_curr - val_prev) / (val_prev + epsilon)
                    yoy_growth_rates.append(rate)
                
                # AAGR is the arithmetic mean of these consecutive YoY steps
                aagr = np.mean(yoy_growth_rates)
                
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
        text_words = str(row['combined_text']).lower().split()
        
        if not text_words:
            return 0
            
        doc_length = len(text_words)
        weighted_scores = []
        unique_words = set(text_words)
        
        for w in unique_words:
            word_score = yearly_word_scores.get(year, {}).get(w, 0)
            if word_score > 0:
                # Term frequency (TF) within this specific startup's description
                tf = text_words.count(w) / doc_length
                weighted_scores.append(word_score * tf)
                
        if not weighted_scores:
            return 0
            
        # Multiply by 100 to scale the score up to a readable continuous variable
        return np.sum(weighted_scores) * 100

    df_train['Market_Signal_Score'] = df_train.apply(get_startup_score, axis=1)
    return df_train, yearly_word_scores, yearly_word_freq

# --- PHASE 3: Model Training ---
def run_model_experiment(df):
    print("Training Models...")
    le_ind = LabelEncoder()
    df['industry_encoded'] = le_ind.fit_transform(df['category_code'].fillna('Unknown'))
    
    base_features = ['industry_encoded', 'funding_rounds', 'funding_total_usd', 'milestones', 'relationships', 'investment_rounds']
    exist_base = [c for c in base_features if c in df.columns]
    aug_features = exist_base + ['Market_Signal_Score']
    
    X_base = df[exist_base]
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