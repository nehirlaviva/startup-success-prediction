#Phase 1
import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    print("Loading dataset...")
    # 1. Load the CSV file. We use unicode_escape because some startup names have weird characters
    df = pd.read_csv(file_path, encoding='unicode_escape')
    
    # 2. Clean the column names (some of them have invisible spaces like ' market ')
    df.columns = df.columns.str.strip()
    
    print("Cleaning missing values and text...")
    # 3. Drop rows where we are missing the crucial information we need
    df = df.dropna(subset=['founded_year', 'status', 'category_list', 'market'])
    
    # 4. Clean the money column: remove commas, dashes, and extra spaces, then turn it into a real number
    df['funding_total_usd'] = df['funding_total_usd'].astype(str).str.replace(',', '').str.replace('-', '').str.strip()
    df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce').fillna(0)
    
    print("Defining the Success Target...")
    # 5. Define "Success" (Target = 1) based on your thesis Chapter 2.1:
    # Success = The startup was Acquired, went IPO, or raised Series B funding
    def determine_success(row):
        # If it exited successfully
        if row['status'] in ['acquired', 'ipo']:
            return 1
        # If it reached Series B funding (meaning it survived the early death valley)
        if pd.to_numeric(row['round_B'], errors='coerce') > 0:
            return 1
        # Otherwise, it's a 0 (closed, or still operating but hasn't reached major milestones)
        return 0
        
    df['Target'] = df.apply(determine_success, axis=1)
    
    # 6. Keep only the columns we actually need for the next phases to save computer memory
    columns_to_keep = [
        'name', 'category_list', 'market', 'funding_total_usd', 
        'funding_rounds', 'founded_year', 'country_code', 'Target'
    ]
    df_clean = df[columns_to_keep].copy()
    
    # Clean up the category list formatting (e.g., "|Games|Video|" becomes "Games Video")
    df_clean['category_list'] = df_clean['category_list'].str.replace('|', ' ', regex=False).str.strip()
    
    print(f"Data cleaning complete! Final dataset has {len(df_clean)} startups.")
    return df_clean



#Phase 2
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_momentum_scores(df):
    print("Starting Phase 2: Calculating Market Momentum Scores...")
    
    # 1. Make sure years are integers and sorted chronologically
    df['founded_year'] = df['founded_year'].astype(int)
    years = sorted(df['founded_year'].unique())
    
    # We will store the word frequencies (TF-IDF) per year here
    yearly_word_freq = {}
    
    print("Step 1: Calculating TF-IDF (Popularity) for each year...")
    for year in years:
        # Get all category words for startups founded in this specific year
        text_data = df[df['founded_year'] == year]['category_list'].dropna().tolist()
        
        if not text_data:
            yearly_word_freq[year] = {}
            continue
            
        # Use TF-IDF to score words. Lowercase=True ensures 'News' and 'news' are treated the same.
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', lowercase=True)
        try:
            tfidf_matrix = vectorizer.fit_transform(text_data)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get the average popularity score for each word in this year
            avg_tfidf = tfidf_matrix.mean(axis=0).A1
            yearly_word_freq[year] = dict(zip(feature_names, avg_tfidf))
        except ValueError:
            # Skip if there's no usable text data for a weird year
            yearly_word_freq[year] = {}
            
    print("Step 2: Calculating AAGR (Growth) and Mapping to Quadrants...")
    yearly_word_scores = {}
    
    for current_year in years:
        word_stats = []
        current_freqs = yearly_word_freq.get(current_year, {})
        
        # Look back up to 3 years to calculate growth
        past_years = [y for y in years if y < current_year and y >= current_year - 3]
        
        for word, tfidf in current_freqs.items():
            aagr = 0
            if len(past_years) > 0:
                growth_rates = []
                for py in past_years:
                    past_tfidf = yearly_word_freq.get(py, {}).get(word, 0)
                    # We add a tiny number (0.0001) to avoid dividing by zero!
                    growth = (tfidf - past_tfidf) / (past_tfidf + 0.0001)
                    growth_rates.append(growth)
                aagr = np.mean(growth_rates)
            
            word_stats.append({'word': word, 'tfidf': tfidf, 'aagr': aagr})
            
        if not word_stats:
            yearly_word_scores[current_year] = {}
            continue
            
        stats_df = pd.DataFrame(word_stats)
        
        # Find the "Median" line to divide our map into 4 quadrants
        median_tfidf = stats_df['tfidf'].median()
        median_aagr = stats_df['aagr'].median()
        
        # Map words to your thesis quadrants
        score_dict = {}
        for _, row in stats_df.iterrows():
            w = row['word']
            t = row['tfidf']
            a = row['aagr']
            
            if t < median_tfidf and a > median_aagr:
                score = 3  # Weak Signal (High growth, low frequency)
            elif t >= median_tfidf and a > median_aagr:
                score = 2  # Strong Signal (High growth, high frequency)
            elif t >= median_tfidf and a <= median_aagr:
                score = 1  # Stable Area (Low growth, high frequency)
            else:
                score = 0  # Niche Area (Low growth, low frequency)
                
            score_dict[w] = score
            
        yearly_word_scores[current_year] = score_dict

    print("Step 3: Assigning final Market Signal Score to startups...")
    
    def get_startup_score(row):
        year = row['founded_year']
        # Convert categories to lowercase and split into words
        categories = str(row['category_list']).lower().split()
        
        scores = []
        for cat in categories:
            # Look up the score of this word in the year the startup was founded
            s = yearly_word_scores.get(year, {}).get(cat)
            if s is not None:
                scores.append(s)
                
        # Return the average score of all its categories. If none found, return 0.
        return np.mean(scores) if scores else 0

    df['Market_Signal_Score'] = df.apply(get_startup_score, axis=1)
    
    print("Phase 2 complete!")
    return df


#Phase 3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

def run_model_experiment(df):
    print("Starting Phase 3: Model Training and Comparative Benchmarking...")
    
    # 1. Encoding: Turn text columns into numbers
    print("Encoding text features into numbers...")
    le_market = LabelEncoder()
    le_country = LabelEncoder()
    
    df['market_encoded'] = le_market.fit_transform(df['market'].fillna('Unknown'))
    df['country_code_encoded'] = le_country.fit_transform(df['country_code'].fillna('Unknown'))
    
    # 2. Define our two sets of features
    base_features = ['funding_total_usd', 'funding_rounds', 'founded_year', 'market_encoded', 'country_code_encoded']
    aug_features = base_features + ['Market_Signal_Score']
    
    X_base = df[base_features]
    X_aug = df[aug_features]
    y = df['Target']
    
    # 3. Split the data
    print("Splitting data into Train and Test sets...")
    X_base_train, X_base_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
    X_aug_train, X_aug_test, _, _ = train_test_split(X_aug, y, test_size=0.2, random_state=42)
    
    # 4. Apply SMOTE to balance the Training data
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_base_train_smote, y_train_smote = smote.fit_resample(X_base_train, y_train)
    X_aug_train_smote, _ = smote.fit_resample(X_aug_train, y_train)
    
    # 5. Train Random Forest Models
    print("Training Random Forest models...")
    rf_base = RandomForestClassifier(random_state=42)
    rf_base.fit(X_base_train_smote, y_train_smote)
    
    rf_aug = RandomForestClassifier(random_state=42)
    rf_aug.fit(X_aug_train_smote, y_train_smote)
    
    # 6. Train XGBoost Models
    print("Training XGBoost models...")
    xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_base.fit(X_base_train_smote, y_train_smote)
    
    xgb_aug = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_aug.fit(X_aug_train_smote, y_train_smote)
    
    # 7. Make Predictions
    rf_base_preds = rf_base.predict(X_base_test)
    rf_aug_preds = rf_aug.predict(X_aug_test)
    
    xgb_base_preds = xgb_base.predict(X_base_test)
    xgb_aug_preds = xgb_aug.predict(X_aug_test)
    
    # 8. Calculate F1 Scores
    f1_rf_base = f1_score(y_test, rf_base_preds)
    f1_rf_aug = f1_score(y_test, rf_aug_preds)
    
    f1_xgb_base = f1_score(y_test, xgb_base_preds)
    f1_xgb_aug = f1_score(y_test, xgb_aug_preds)
    
    # Print out the Comparative Benchmarking Table
    print("\n" + "="*60)
    print("🏆 COMPARATIVE BENCHMARKING RESULTS (F1-SCORE) 🏆")
    print("-" * 60)
    print(f"{'Model':<15} | {'Baseline (No Timing)':<20} | {'Augmented (With Timing)'}")
    print("-" * 60)
    print(f"{'Random Forest':<15} | {f1_rf_base:<20.4f} | {f1_rf_aug:.4f}")
    print(f"{'XGBoost':<15} | {f1_xgb_base:<20.4f} | {f1_xgb_aug:.4f}")
    print("="*60)

   # We return BOTH augmented models for Phase 4 SHAP comparison
    return rf_aug, xgb_aug, X_aug_test, y_test