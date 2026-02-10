import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="SBF 120 Asset Management", layout="wide")

# --- INTERFACE ---
st.title("🏛️ Terminal d'Optimisation de Portefeuille")
st.markdown("Extraction en temps réel des données du **SBF 120** (Période 2015-2025)")

# 1. BASE DE DONNÉES (Tickers SBF 120 et scores ESG fictifs pour l'exercice)
assets = {
    'Air Liquide': {'t': 'AI.PA', 'esg': 85},
    'Airbus': {'t': 'AIR.PA', 'esg': 72},
    'BNP Paribas': {'t': 'BNP.PA', 'esg': 68},
    'Hermès': {'t': 'RMS.PA', 'esg': 80},
    'Kering': {'t': 'KER.PA', 'esg': 78},
    'L\'Oréal': {'t': 'OR.PA', 'esg': 88},
    'LVMH': {'t': 'MC.PA', 'esg': 75},
    'Safran': {'t': 'SAF.PA', 'esg': 70},
    'Sanofi': {'t': 'SAN.PA', 'esg': 74},
    'TotalEnergies': {'t': 'TTE.PA', 'esg': 62}
}

st.sidebar.header("Configuration")
selected_names = st.sidebar.multiselect("Sélectionnez vos 10 actifs", list(assets.keys()), default=list(assets.keys()))
selected_tickers = [assets[name]['t'] for name in selected_names]

# --- 2. EXTRACTION AUTOMATIQUE ---
@st.cache_data
def get_data(tickers):
    # Téléchargement des prix de clôture ajustés
    df = yf.download(tickers, start="2015-01-01", end="2025-01-01")['Adj Close']
    return df

if len(selected_tickers) >= 2:
    with st.spinner('Extraction des cours de bourse...'):
        prices = get_data(selected_tickers)
        # Calcul des rendements mensuels (consigne de l'exercice)
        returns = prices.resample('ME').last().pct_change().dropna()

    # --- 3. MATHÉMATIQUES : OPTIMISATION ---
    mean_ret = returns.mean() * 12 # Annualisé
    cov_mat = returns.cov() * 12    # Annualisé

    def portfolio_stats(weights):
        p_ret = np.sum(mean_ret * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
        return p_ret, p_vol

    # On maximise le Ratio de Sharpe (Rendement / Risque)
    def min_func_sharpe(weights):
        r, v = portfolio_stats(weights)
        return -r / v # On minimise l'opposé pour maximiser

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(selected_tickers)))
    init_guess = [1/len(selected_tickers)] * len(selected_tickers)
    
    opt_res = minimize(min_func_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = opt_res.x

    # --- 4. RÉSULTATS & ESG ---
    p_ret, p_vol = portfolio_stats(opt_weights)
    p_esg = sum(opt_weights[i] * assets[selected_names[i]]['esg'] for i in range(len(selected_names)))

    col1, col2, col3 = st.columns(3)
    col1.metric("Rendement Cible", f"{p_ret:.2%}")
    col2.metric("Volatilité (Risque)", f"{p_vol:.2%}")
    col3.metric("Note ESG Portefeuille", f"{p_esg:.1f}/100")

    # --- 5. FRONTIÈRE EFFICIENTE ---
    st.subheader("Analyse de la Frontière Efficiente")
    
    

    # Simulation de portefeuilles pour la frontière
    sim_vol, sim_ret = [], []
    for _ in range(1000):
        w = np.random.random(len(selected_tickers))
        w /= np.sum(w)
        r, v = portfolio_stats(w)
        sim_vol.append(v)
        sim_ret.append(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim_vol, y=sim_ret, mode='markers', marker=dict(color='lightgrey', size=4), name="Possibilités"))
    fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode='markers+text', text=["Portefeuille Optimal"], 
                             marker=dict(color='red', size=12), name="Efficient"))
    fig.update_layout(xaxis_title="Risque (Écart-type)", yaxis_title="Rendement Espéré")
    st.plotly_chart(fig, use_container_width=True)

    # Affichage des poids
    st.subheader("Allocation détaillée")
    df_weights = pd.DataFrame({'Actif': selected_names, 'Poids (%)': (opt_weights*100).round(2)})
    st.table(df_weights.sort_values(by='Poids (%)', ascending=False).T)

else:
    st.info("Veuillez sélectionner au moins 2 actifs.")
