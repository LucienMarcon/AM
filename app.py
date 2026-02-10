import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime

# --- CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="Asset Management Pro - SBF 120", layout="wide")

st.title("🏛️ Terminal de Gestion d'Actifs : Optimisation SBF 120")
st.markdown("""
*Cette application implémente la théorie moderne du portefeuille de Markowitz pour maximiser le **Ratio de Sharpe**.*
""")

# --- 1. SÉLECTION DES TITRES ET PARAMÈTRES ---
st.sidebar.header("⚙️ Paramètres du Portefeuille")

# Liste suggérée de 10 titres majeurs du SBF 120 / CAC 40
default_tickers = ['OR.PA', 'MC.PA', 'AIR.PA', 'TTE.PA', 'SAN.PA', 'KER.PA', 'BNP.PA', 'AI.PA', 'EL.PA', 'RMS.PA']
tickers = st.sidebar.multiselect("Sélectionnez vos 10 actifs :", default_tickers, default=default_tickers)

risk_free_rate = st.sidebar.number_input("Taux sans risque (%)", value=2.0) / 100

# --- 2. EXTRACTION DES DONNÉES FINANCIÈRES ---
@st.cache_data
def download_data(tickers):
    # Période de 10 ans comme demandé
    data = yf.download(tickers, start="2015-01-01", end="2025-01-01")['Adj Close']
    return data

if len(tickers) < 2:
    st.warning("Veuillez sélectionner au moins 2 titres pour optimiser.")
    st.stop()

with st.spinner('Extraction des données Yahoo Finance en cours...'):
    prices = download_data(tickers)
    returns = prices.pct_change().dropna()

# --- 3. FONCTIONS D'OPTIMISATION (Cœur mathématique) ---
def get_portfolio_metrics(weights, returns):
    # Rendements annualisés (252 jours de trading)
    p_return = np.sum(returns.mean() * weights) * 252
    # Volatilité annualisée
    p_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    # Ratio de Sharpe
    sharpe = (p_return - risk_free_rate) / p_std
    return p_return, p_std, sharpe

# Fonction à MINIMISER (Négatif du Sharpe pour maximiser le Sharpe)
def negative_sharpe(weights, returns):
    return -get_portfolio_metrics(weights, returns)[2]

def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    args = (returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Somme des poids = 1
    bounds = tuple((0, 1) for _ in range(num_assets)) # Pas de vente à découvert
    initial_guess = num_assets * [1. / num_assets]
    
    optimized = minimize(negative_sharpe, initial_guess, args=args, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized.x

# --- 4. EXÉCUTION DE L'ANALYSE ---
opt_weights = optimize_portfolio(returns)
p_ret, p_vol, p_sharpe = get_portfolio_metrics(opt_weights, returns)

# --- 5. AFFICHAGE DES RÉSULTATS ---
col1, col2, col3 = st.columns(3)
col1.metric("Rendement Espéré", f"{p_ret:.2%}")
col2.metric("Volatilité (Risque)", f"{p_vol:.2%}")
col3.metric("Ratio de Sharpe Max", f"{p_sharpe:.2f}")

# Tableau des poids optimaux
st.subheader("🎯 Allocation Optimale du Portefeuille")
weights_df = pd.DataFrame({'Actif': tickers, 'Poids (%)': (opt_weights * 100).round(2)})
st.table(weights_df.sort_values(by='Poids (%)', ascending=False).T)

# --- 6. GRAPHIQUE DE LA FRONTIÈRE EFFICIENTE ---
st.subheader("📈 Visualisation de la Frontière Efficiente")

# Simulation de portefeuilles aléatoires pour le nuage de points
num_sim = 1000
sim_results = np.zeros((3, num_sim))
for i in range(num_sim):
    w = np.random.random(len(tickers))
    w /= np.sum(w)
    r, v, s = get_portfolio_metrics(w, returns)
    sim_results[0,i] = v
    sim_results[1,i] = r
    sim_results[2,i] = s

fig = go.Figure()
# Nuage de points
fig.add_trace(go.Scatter(x=sim_results[0,:], y=sim_results[1,:], mode='markers',
                         marker=dict(color=sim_results[2,:], colorscale='Viridis', showscale=True, title="Sharpe"),
                         name="Portefeuilles Simulés"))
# Point Optimal
fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode='markers',
                         marker=dict(color='red', size=15, symbol='star'),
                         name="Portefeuille Optimal (Max Sharpe)"))

fig.update_layout(xaxis_title="Risque (Volatilité)", yaxis_title="Rendement", height=600)
st.plotly_chart(fig, use_container_width=True)

st.info("💡 **Conseil Oral :** Expliquez au jury que ce portefeuille est le point de tangence entre la 'Capital Market Line' et la frontière efficiente.")
