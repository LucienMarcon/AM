import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Configuration de l'interface
st.set_page_config(page_title="Asset Management SBF 120", layout="wide")

st.title("🏛️ Terminal d'Optimisation de Portefeuille SBF 120")
st.markdown("Analyse Multi-Critères : Performance Financière (2015-2025) & ESG Boursorama")

# --- 1. BASE DE DONNÉES DES TITRES & ESG ---
assets_data = {
    'Air Liquide': {'t': 'AI.PA', 'esg': 85},
    'TotalEnergies': {'t': 'TTE.PA', 'esg': 62},
    'Capgemini': {'t': 'CAP.PA', 'esg': 80},
    'AXA': {'t': 'CS.PA', 'esg': 75},
    'Vusion': {'t': 'VUSION.PA', 'esg': 70},
    'LVMH': {'t': 'MC.PA', 'esg': 78},
    'Saint Gobain': {'t': 'SGO.PA', 'esg': 74},
    'Sanofi': {'t': 'SAN.PA', 'esg': 82},
    'STMicroelectronics': {'t': 'STMPA.PA', 'esg': 80},
    'Airbus': {'t': 'AIR.PA', 'esg': 72}
}

# --- 2. INTERACTIVITÉ DANS LA SIDEBAR ---
st.sidebar.header("🛡️ Paramètres de Conviction")

criteres = ["Risque ESG", "Controverse", "Impact Positif", "Impact Négatif", "Exposition", "Management", "Risque Carbone"]
selected_weights = {}
for c in criteres:
    selected_weights[c] = st.sidebar.slider(f"Importance : {c}", 0, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Objectif de Rendement")
target_return = st.sidebar.slider("Rendement annuel cible (%)", 5.0, 40.0, 18.0) / 100

# --- 3. EXTRACTION ET CALCULS ---
tickers = [v['t'] for v in assets_data.values()]

@st.cache_data
def load_financial_data(tickers_list):
    # Extraction 10 ans
    df = yf.download(tickers_list, start="2015-01-01", end="2025-12-31")['Adj Close']
    return df

prices = load_financial_data(tickers)
returns = prices.resample('ME').last().pct_change().dropna()
mean_returns = returns.mean() * 12
cov_matrix = returns.cov() * 12

# --- 4. OPTIMISATION MATHÉMATIQUE (MARKOWITZ) ---
def get_port_stats(weights):
    p_ret = np.sum(mean_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_ret, p_vol

def min_vol_func(weights):
    return get_port_stats(weights)[1]

constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ge', 'fun': lambda x: get_port_stats(x)[0] - target_return}
]
bounds = tuple((0, 1) for _ in range(len(tickers)))

res = minimize(min_vol_func, [1/len(tickers)]*len(tickers), method='SLSQP', bounds=bounds, constraints=constraints)

# --- 5. AFFICHAGE DES RÉSULTATS ---
if res.success:
    w_opt = res.x
    p_ret, p_vol = get_port_stats(w_opt)
    p_esg = sum(w_opt[i] * list(assets_data.values())[i]['esg'] for i in range(len(tickers)))

    col1, col2, col3 = st.columns(3)
    col1.metric("Rendement Espéré", f"{p_ret:.2%}")
    col2.metric("Risque (Volatilité)", f"{p_vol:.2%}")
    col3.metric("Score ESG Portefeuille", f"{p_esg:.1f}/100")

    st.subheader("📈 Frontière Efficiente & Votre Position")
    
    sim_vol, sim_ret = [], []
    for _ in range(1000):
        w = np.random.random(len(tickers))
        w /= np.sum(w)
        r, v = get_port_stats(w)
        sim_vol.append(v)
        sim_ret.append(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim_vol, y=sim_ret, mode='markers', 
                             marker=dict(color='lightgrey', opacity=0.4), name="Nuage de Markowitz"))
    fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode='markers', 
                             marker=dict(color='red', size=15, symbol='star'), name="Votre Portefeuille"))
    fig.update_layout(xaxis_title="Risque (Volatilité)", yaxis_title="Rendement Annuel")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Allocation Optimale")
    alloc_df = pd.DataFrame({'Titre': list(assets_data.keys()), 'Poids (%)': (w_opt*100).round(2)})
    st.table(alloc_df.sort_values(by='Poids (%)', ascending=False).T)

else:
    st.error("⚠️ Rendement impossible à atteindre avec cette volatilité. Essayez de baisser l'objectif (ex: 15%).")
