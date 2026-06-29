import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Food Recommendation System", layout="wide", page_icon="🍳")

# Memicu pemrosesan data otomatis jika file pickle belum ada di server Cloud
if not os.path.exists('df_recipes_sample.pkl'):
    with st.spinner('Menyiapkan dataset dari Kaggle & training model (Ini hanya memakan waktu ~1-2 menit di awal)...'):
        from setup_data import prepare_all_data
        prepare_all_data()

@st.cache_resource
def load_all_components():
    df_recipes = pd.read_pickle('df_recipes_sample.pkl')
    df_interactions = pd.read_pickle('df_interactions_clean.pkl')
    recipe_indices = pd.read_pickle('recipe_indices.pkl')
    with open('cosine_sim.pkl', 'rb') as f: cosine_sim = pickle.load(f)
    with open('svd_model.pkl', 'rb') as f: svd_model = pickle.load(f)
    return df_recipes, df_interactions, recipe_indices, cosine_sim, svd_model

df_recipes, df_interactions, recipe_indices, cosine_sim, svd_model = load_all_components()

# --- FUNGSI REKOMENDASI ---
def content_based_recommend(recipe_name, top_n=10):
    recipe_name = recipe_name.lower()
    matches = [name for name in recipe_indices.index if recipe_name in str(name)]
    if not matches: return None, None
    matched_name = matches[0]
    idx = recipe_indices[matched_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    result = df_recipes.iloc[[i[0] for i in sim_scores]][['name', 'minutes', 'calories', 'protein']].copy()
    result['similarity_score'] = [round(i[1], 4) for i in sim_scores]
    return result.reset_index(drop=True), matched_name

def collaborative_recommend(user_id, top_n=10):
    rated_recipes = df_interactions[df_interactions['user_id'] == user_id]['recipe_id'].tolist()
    unrated_recipes = [r for r in df_recipes['id'].tolist() if r not in rated_recipes]
    if not unrated_recipes: return None
    preds = [(r, svd_model.predict(user_id, r).est) for r in unrated_recipes]
    preds.sort(key=lambda x: x[1], reverse=True)
    top_recipes = preds[:top_n]
    result = df_recipes[df_recipes['id'].isin([r[0] for r in top_recipes])][['id', 'name', 'minutes', 'calories', 'protein']].copy()
    result['predicted_rating'] = result['id'].map(dict(top_recipes)).round(4)
    return result.sort_values('predicted_rating', ascending=False).reset_index(drop=True)

def hybrid_recommend(user_id, recipe_name, top_n=10, alpha=0.4, beta=0.6):
    recipe_name = recipe_name.lower()
    matches = [name for name in recipe_indices.index if recipe_name in str(name)]
    if not matches: return None, None
    matched_name = matches[0]
    content_score_dict = {i: score for i, score in list(enumerate(cosine_sim[recipe_indices[matched_name]]))}
    rated_recipes = df_interactions[df_interactions['user_id'] == user_id]['recipe_id'].tolist()
    unrated_recipes = [r for r in df_recipes['id'].tolist() if r not in rated_recipes]
    cf_scores = {r: svd_model.predict(user_id, r).est for r in unrated_recipes}
    if cf_scores:
        min_cf, max_cf = min(cf_scores.values()), max(cf_scores.values())
        cf_scores_norm = {k: (v - min_cf) / (max_cf - min_cf + 1e-9) for k, v in cf_scores.items()}
    else: cf_scores_norm = {}
    hybrid_scores = []
    for i, row in df_recipes.iterrows():
        recipe_id = row['id']
        if recipe_id in rated_recipes: continue
        hybrid_score = (alpha * content_score_dict.get(i, 0)) + (beta * cf_scores_norm.get(recipe_id, 0))
        hybrid_scores.append((i, content_score_dict.get(i, 0), cf_scores_norm.get(recipe_id, 0), hybrid_score))
    hybrid_scores.sort(key=lambda x: x[3], reverse=True)
    top_hybrid = hybrid_scores[:top_n]
    result = df_recipes.loc[[h[0] for h in top_hybrid]][['name', 'minutes', 'calories', 'protein']].copy()
    result['cb_score'] = [round(h[1], 4) for h in top_hybrid]
    result['cf_score'] = [round(h[2], 4) for h in top_hybrid]
    result['hybrid_score'] = [round(h[3], 4) for h in top_hybrid]
    return result.reset_index(drop=True), matched_name

# --- INTERFACE UI ---
st.title("🍳 Food Recommendation System")
st.caption("Aplikasi Sistem Rekomendasi Interaktif Kelompok 7 — Tugas Akhir Machine Learning")
st.write("---")

st.sidebar.header("⚙️ Pengaturan Pengguna")
selected_user = st.sidebar.selectbox("Pilih User ID:", options=df_interactions['user_id'].unique())

tab1, tab2, tab3 = st.tabs(["🎯 Content-Based", "👥 Collaborative", "🧬 Hybrid System"])

with tab1:
    st.header("🎯 Rekomendasi Berdasarkan Kemiripan Resep")
    input_recipe = st.text_input("Ketik Nama Resep:", value="chicken soup", key="cb_input")
    top_n_cb = st.slider("Jumlah Rekomendasi:", 5, 20, 10, key="cb_slider")
    if st.button("Cari Resep Mirip", type="primary"):
        res, match = content_based_recommend(input_recipe, top_n_cb)
        if res is not None:
            st.success(f"Referensi ditemukan: **{match.title()}**")
            st.dataframe(res, use_container_width=True)
        else: st.warning("Resep tidak ditemukan.")

with tab2:
    st.header("👥 Rekomendasi Personal User (SVD)")
    st.info(f"Rekomendasi untuk **User ID: {selected_user}**")
    top_n_cf = st.slider("Jumlah Rekomendasi:", 5, 20, 10, key="cf_slider")
    if st.button("Generate Rekomendasi Personal", type="primary"):
        res_cf = collaborative_recommend(selected_user, top_n_cf)
        if res_cf is not None: st.dataframe(res_cf, use_container_width=True)
        else: st.warning("User telah menilai semua resep.")

with tab3:
    st.header("🧬 Sistem Rekomendasi Hybrid")
    col1, col2 = st.columns(2)
    with col1: input_recipe_hyb = st.text_input("Resep Referensi Konten:", value="chicken soup", key="hyb_input")
    with col2: top_n_hyb = st.slider("Jumlah Rekomendasi:", 5, 20, 10, key="hyb_slider")
    alpha = st.slider("Bobot Content-Based (α):", 0.0, 1.0, 0.4, 0.1)
    st.caption(f"Bobot Collaborative (β): **{1.0 - alpha:.1f}**")
    if st.button("Generate Rekomendasi Hybrid", type="primary"):
        res_hyb, match_hyb = hybrid_recommend(selected_user, input_recipe_hyb, top_n_hyb, alpha, 1.0 - alpha)
        if res_hyb is not None:
            st.success(f"Rekomendasi Hybrid | Referensi: **{match_hyb.title()}**")
            st.dataframe(res_hyb, use_container_width=True)
        else: st.warning("Resep tidak ditemukan.")