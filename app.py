import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Optimiseur SBF 120 Pro", layout="wide")

st.title("🏛️ Terminal de Gestion d'Actifs : Optimisation & ESG")

# --- 1. CHARGEMENT DES DONNÉES ESG ---
st.sidebar.header("1. Données Extra-Financières")
uploaded_file = st.sidebar.file_uploader("Chargez votre fichier Excel ESG", type=["xlsx"])

if uploaded_file:
    df_esg = pd.read_excel(uploaded_file)
    tickers = df_esg['Ticker'].tolist()
    
    # Choix des critères ESG prioritaires
    criteres = ["Risque ESG", "Niveau de controverse", "Impact positif", "Impact négatif", "Risque d'exposition", "Score de management", "Risque carbone"]
    selected_criteres = st.sidebar.multiselect("Accentuez votre démarche sur :", criteres, default=criteres)
    
    # Calcul d'un score ESG agrégé basé sur la sélection
    df_esg['Score_Global'] = df_esg[selected_criteres].mean(axis=1)

    # --- 2. EXTRACTION DES DONNÉES BOURSIÈRES (10 ANS) ---
    @st.cache_data
    def download_data(list_tickers):
        data = yf.download(list_tickers, start="2015-01-01", end="2025-01-01")['Adj Close']
        return data

    with st.spinner('Extraction de 10 ans de données boursières...'):
        prices = download_data(tickers)
        returns = prices.resample('ME').last().pct_change().dropna()

    # --- 3. PARAMÈTRES D'OPTIMISATION ---
    st.sidebar.header("2. Objectifs Financiers")
    target_return = st.sidebar.slider("Rendement annuel espéré (%)", 5.0, 30.0, 15.0) / 100

    # Moyennes et Covariance
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12

    def portfolio_stats(weights):
        p_ret = np.sum(mean_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return p_ret, p_vol

    # --- 4. OPTIMISATION : RENDEMENT MAXIMUM SOUS CONTRAINTE ---
    # On minimise la volatilité pour un rendement cible donné
    def objective(weights):
        return portfolio_stats(weights)[1]

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # Somme des poids = 1
        {'type': 'ge', 'fun': lambda x: portfolio_stats(x)[0] - target_return} # Rendement >= Cible
    ]
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    init_guess = [1/len(tickers)] * len(tickers)

    opt_res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if opt_res.success:
        opt_weights = opt_res.x
        p_ret, p_vol = portfolio_stats(opt_weights)
        
        # Calcul de la note ESG moyenne du portefeuille
        p_esg_score = np.dot(opt_weights, df_esg['Score_Global'].values)

        # --- 5. AFFICHAGE DES RÉSULTATS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement Réalisé", f"{p_ret:.2%}")
        col2.metric("Volatilité", f"{p_vol:.2%}")
        col3.metric("Note ESG Moyenne", f"{p_esg_score:.1f}/100")

        # Graphique Frontière Efficiente
        st.subheader("📈 Frontière Efficiente et Point Optimal")
        
        
        
        # Simulation pour le graphique
        sim_vol, sim_ret = [], []
        for _ in range(1000):
            w = np.random.random(len(tickers))
            w /= np.sum(w)
            r, v = portfolio_stats(w)
            sim_vol.append(v)
            sim_ret.append(r)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_vol, y=sim_ret, mode='markers', marker=dict(color='lightgrey'), name="Possibilités"))
        fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode='markers+text', text=["VOTRE CHOIX"], 
                                 marker=dict(color='red', size=15, symbol='star'), name="Optimal"))
        st.plotly_chart(fig, use_container_width=True)

        # Tableau d'allocation
        st.subheader("📋 Allocation suggérée")
        df_weights = pd.DataFrame({'Action': tickers, 'Poids (%)': (opt_weights*100).round(2)})
        st.table(df_weights.sort_values(by='Poids (%)', ascending=False).reset_index(drop=True))

        # --- 6. EXTRACTION EXCEL ---
        st.subheader("📥 Exporter les résultats")
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df_weights)
        st.download_button("Télécharger l'allocation (CSV)", data=csv, file_name='mon_portefeuille.csv', mime='text/csv')

    else:
        st.error("Impossible d'atteindre ce rendement avec ces actions. Essayez de baisser l'objectif.")

else:
    st.info("👋 Bienvenue ! Commencez par charger votre fichier Excel contenant les tickers et les notes ESG.")
