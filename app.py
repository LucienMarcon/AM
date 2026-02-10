import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Portfolio SBF 120", layout="wide")

st.title("🏛️ Terminal d'Optimisation de Portefeuille SBF 120")
st.markdown("Analyse : Performance (2015-2025) & Critères ESG Boursorama")

# 2. BASE DE DONNÉES DES TITRES (Tes 10 titres réels)
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

# 3. BARRE LATÉRALE - PARAMÈTRES
st.sidebar.header("🛡️ Paramètres de Conviction")
criteres = ["Risque ESG", "Controverse", "Impact Positif", "Impact Négatif", "Exposition", "Management", "Risque Carbone"]
for c in criteres:
    st.sidebar.slider(f"Poids : {c}", 0, 10, 5)

st.sidebar.markdown("---")
target_return_pct = st.sidebar.slider("Rendement annuel cible (%)", 5, 30, 15)
target_return = target_return_pct / 100

# 4. CHARGEMENT DES DONNÉES FINANCIÈRES (10 ANS)
tickers = [v['t'] for v in assets_data.values()]

@st.cache_data
def get_data(tickers_list):
    # Extraction sur la période demandée : 2015-2025
    df = yf.download(tickers_list, start="2015-01-01", end="2025-01-01")['Adj Close']
    return df

try:
    prices = get_data(tickers)
    returns = prices.resample('ME').last().pct_change().dropna()
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12

    # 5. OPTIMISATION (FONCTIONS)
    def get_stats(w):
        p_ret = np.sum(mean_returns * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return p_ret, p_vol

    def min_vol(w):
        return get_stats(w)[1]

    # SYNTAXE UNIVERSELLE POUR LES CONTRAINTES
    # 'ineq' signifie que la fonction doit être >= 0
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: get_stats(x)[0] - target_return} 
    ]
    
    bounds = tuple((0.0, 1.0) for _ in range(len(tickers)))
    init_guess = [1.0 / len(tickers)] * len(tickers)

    # Lancement de l'optimiseur avec la méthode SLSQP
    res = minimize(min_vol, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

    if res.success:
        w_opt = res.x
        p_ret, p_vol = get_stats(w_opt)
        p_esg = sum(w_opt[i] * list(assets_data.values())[i]['esg'] for i in range(len(tickers)))

        # 6. AFFICHAGE DES RÉSULTATS
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement", f"{p_ret:.2%}")
        col2.metric("Volatilité", f"{p_vol:.2%}")
        col3.metric("Note ESG", f"{p_esg:.1f}/100")

        st.subheader("📈 Frontière Efficiente de Markowitz")
        
        
        sim_vol, sim_ret = [], []
        for _ in range(800):
            w = np.random.random(len(tickers))
            w /= np.sum(w)
            r, v = get_stats(w)
            sim_vol.append(v)
            sim_ret.append(r)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_vol, y=sim_ret, mode='markers', marker=dict(color='gray', opacity=0.3), name="Portefeuilles"))
        fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Optimal"))
        fig.update_layout(xaxis_title="Risque (Volatilité)", yaxis_title="Rendement Annuel")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Allocation suggérée")
        alloc_df = pd.DataFrame({'Titre': list(assets_data.keys()), 'Poids (%)': (w_opt*100).round(2)})
        st.table(alloc_df.sort_values(by='Poids (%)', ascending=False).T)

    else:
        st.error(f"⚠️ Impossible d'atteindre {target_return_pct}% de rendement. Baissez l'objectif dans la barre latérale.")

except Exception as e:
    st.error(f"Erreur lors du calcul : {e}")
