import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# 1. CONFIGURATION
st.set_page_config(page_title="Portfolio SBF 120", layout="wide")
st.title("🏛️ Terminal d'Optimisation de Portefeuille SBF 120")
st.markdown("Analyse : Performance (2015-2025) & Critères ESG Boursorama")

# 2. DONNÉES (Tes 10 titres réels)
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

# 3. SIDEBAR
st.sidebar.header("🛡️ Paramètres ESG Boursorama")
criteres = ["Risque ESG", "Controverse", "Impact Positif", "Impact Négatif", "Exposition", "Management", "Risque Carbone"]
for c in criteres:
    st.sidebar.slider(f"Poids : {c}", 0, 10, 5)

st.sidebar.markdown("---")
target_return_pct = st.sidebar.slider("Rendement annuel cible (%)", 5, 30, 15)
target_return = target_return_pct / 100

# 4. EXTRACTION SÉCURISÉE (Correction de l'erreur d'Index)
tickers = [v['t'] for v in assets_data.values()]

@st.cache_data
def get_clean_data(tickers_list):
    # Téléchargement et nettoyage immédiat
    data = yf.download(tickers_list, start="2015-01-01", end="2025-01-01")['Adj Close']
    # Si yfinance renvoie un format bizarre, on s'assure que c'est un DataFrame propre
    df = pd.DataFrame(data)
    return df

try:
    prices = get_clean_data(tickers)
    # Calcul des rendements mensuels comme demandé
    returns = prices.resample('ME').last().pct_change().dropna()
    
    # Statistiques annualisées
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12

    # 5. OPTIMISATION
    def get_stats(w):
        p_ret = np.sum(mean_returns * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return p_ret, p_vol

    def min_vol(w):
        return get_stats(w)[1]

    # Contraintes compatibles SciPy
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: get_stats(x)[0] - target_return} 
    ]
    
    bounds = tuple((0.0, 1.0) for _ in range(len(tickers)))
    init_guess = [1.0 / len(tickers)] * len(tickers)

    res = minimize(min_vol, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

    if res.success:
        w_opt = res.x
        p_ret, p_vol = get_stats(w_opt)
        p_esg = sum(w_opt[i] * list(assets_data.values())[i]['esg'] for i in range(len(tickers)))

        # 6. AFFICHAGE
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement Annuel", f"{p_ret:.2%}")
        col2.metric("Volatilité", f"{p_vol:.2%}")
        col3.metric("Note ESG Moyenne", f"{p_esg:.1f}/100")

        st.subheader("📈 Frontière Efficiente (Markowitz)")
        
        

        # Simulation du nuage de points
        sim_vol, sim_ret = [], []
        for _ in range(500):
            w = np.random.random(len(tickers))
            w /= np.sum(w)
            r_s, v_s = get_stats(w)
            sim_vol.append(v_s)
            sim_ret.append(r_s)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_vol, y=sim_ret, mode='markers', marker=dict(color='gray', opacity=0.3), name="Univers de choix"))
        fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Votre Portefeuille"))
        fig.update_layout(xaxis_title="Risque (Volatilité)", yaxis_title="Rendement Espéré")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Allocation Finale des Titres")
        alloc_df = pd.DataFrame({'Titre': list(assets_data.keys()), 'Poids (%)': (w_opt*100).round(2)})
        st.table(alloc_df.sort_values(by='Poids (%)', ascending=False).T)

    else:
        st.error(f"⚠️ Rendement de {target_return_pct}% impossible à atteindre. Baissez l'objectif.")

except Exception as e:
    st.error(f"Détail technique : {e}")
