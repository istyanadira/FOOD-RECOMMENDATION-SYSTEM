import os
import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, SVD
import pickle

def prepare_all_data():
    print("Mulai mengunduh dan memproses data dari Kaggle...")
    
    # 1. Setup Kaggle Token milik kelompokmu
    KAGGLE_TOKEN = "KGAT_9ed35927e9283a77bc4f4ecacbbf68d7"
    os.makedirs('/root/.kaggle', exist_ok=True)
    with open('/root/.kaggle/access_token', 'w') as f:
        f.write(KAGGLE_TOKEN)
    os.chmod('/root/.kaggle/access_token', 0o600)
    os.environ['KAGGLE_CONFIG_DIR'] = '/root/.kaggle'

    # 2. Download & Unzip Dataset
    import kaggle
    kaggle.api.dataset_download_files('shuyangli94/food-com-recipes-and-user-interactions', path='.', unzip=True)
    
    # 3. Load Data
    df_recipes = pd.read_csv('RAW_recipes.csv')
    df_interactions = pd.read_csv('RAW_interactions.csv')
    
    # 4. Sampling (Diubah ke 3000 agar optimal di server)
    SAMPLE_SIZE = 3000
    np.random.seed(42)
    df_recipes_sample = df_recipes.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    sample_ids = df_recipes_sample['id'].tolist()
    df_interactions_sample = df_interactions[df_interactions['recipe_id'].isin(sample_ids)].reset_index(drop=True)
    
    # 5. Preprocessing Nutrisi & Teks
    def parse_nutrition(nutrition_str):
        try: return ast.literal_eval(nutrition_str)
        except: return [0, 0, 0, 0, 0, 0, 0]

    nutrition_cols = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
    nutrition_data = df_recipes_sample['nutrition'].apply(parse_nutrition)
    df_recipes_sample[nutrition_cols] = pd.DataFrame(nutrition_data.tolist(), index=df_recipes_sample.index)
    
    def list_to_text(val):
        items = ast.literal_eval(val) if isinstance(val, str) else val
        return ' '.join([re.sub(r'[^a-zA-Z0-9 ]', '', str(i).lower()) for i in items])

    df_recipes_sample['tags_text'] = df_recipes_sample['tags'].apply(list_to_text)
    df_recipes_sample['ingredients_text'] = df_recipes_sample['ingredients'].apply(list_to_text)
    df_recipes_sample['content'] = df_recipes_sample['name'].fillna('') + ' ' + df_recipes_sample['tags_text'] + ' ' + df_recipes_sample['ingredients_text']
    
    df_interactions_clean = df_interactions_sample[df_interactions_sample['rating'] > 0].copy()
    user_counts = df_interactions_clean['user_id'].value_counts()
    active_users = user_counts[user_counts >= 3].index
    df_interactions_clean = df_interactions_clean[df_interactions_clean['user_id'].isin(active_users)]
    
    # 6. TF-IDF & Cosine Similarity
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_recipes_sample['content'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    recipe_indices = pd.Series(df_recipes_sample.index, index=df_recipes_sample['name'].str.lower())
    
    # 7. Collaborative SVD Training
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_interactions_clean[['user_id', 'recipe_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd_model = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd_model.fit(trainset)
    
    # 8. Save ringkas ke lokal server
    df_recipes_sample[['id', 'name', 'minutes', 'calories', 'protein']].to_pickle('df_recipes_sample.pkl')
    df_interactions_clean[['user_id', 'recipe_id', 'rating']].to_pickle('df_interactions_clean.pkl')
    recipe_indices.to_pickle('recipe_indices.pkl')
    
    with open('cosine_sim.pkl', 'wb') as f: pickle.dump(cosine_sim, f)
    with open('svd_model.pkl', 'wb') as f: pickle.dump(svd_model, f)
    print("Semua data sukses diproses dan disimpan di server Cloud!")

if __name__ == "__main__":
    prepare_all_data()