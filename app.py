import streamlit as st
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import feedparser
import plotly.graph_objects as go
import numpy as np  # Ajouté pour les calculs statistiques

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Market AI Tool", layout="wide")

# 2. DICTIONNAIRE IA
FINANCIAL_SENTIMENT_DICT = {
    "perd": -0.8, "perte": -0.8, "hack": -0.9, "faillite": -1.0, 
    "lancement": 0.7, "adoption": 0.7, "record": 0.8, "croissance": 0.6
}

def analyser_sentiment(titre):
    blob = TextBlob(titre, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    score = blob.sentiment[0]
    for mot, poids in FINANCIAL_SENTIMENT_DICT.items():
        if mot in titre.lower(): score += poids
    return max(min(score, 1.0), -1.0)

# 3. TITRE ET BANDEAU DÉFILANT
st.title("🌍 Analyseur de Marché Universel")

tickers_bandeau = ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "NVDA", "^FCHI"]
bandeau_html = "<marquee style='color: #00ff00; background: #1e1e1e; padding: 10px; font-family: monospace;'>"
for t in tickers_bandeau:
    try:
        p = yf.Ticker(t).fast_info['last_price']
        bandeau_html += f" &nbsp;&nbsp;&nbsp; 🔹 {t}: {p:.2f} &nbsp;&nbsp;&nbsp; |"
    except: continue
bandeau_html += "</marquee>"
st.markdown(bandeau_html, unsafe_allow_html=True)

# 4. BARRE LATÉRALE (CHOIX)
st.sidebar.header("🚀 Configuration")
mode = st.sidebar.selectbox("Catégorie", ["Cryptos", "Bourse", "Recherche Manuelle"])

if mode == "Cryptos":
    liste = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"}
    choix = st.sidebar.selectbox("Choisir", list(liste.keys()))
    ticker = liste[choix]
elif mode == "Bourse":
    liste = {"Nvidia": "NVDA", "Apple": "AAPL", "Tesla": "TSLA", "LVMH": "MC.PA"}
    choix = st.sidebar.selectbox("Choisir", list(liste.keys()))
    ticker = liste[choix]
else:
    ticker = st.sidebar.text_input("Symbole (ex: GOOGL)", "BTC-USD").upper()

st.sidebar.subheader("📅 Période du Graphique")
periode_choisie = st.sidebar.selectbox(
    "Durée du recul :",
    ["1 Jour", "1 Semaine", "1 Mois", "1 An", "Depuis le début (Max)"]
)

mapping_periodes = {
    "1 Jour": {"period": "1d", "interval": "15m"},      
    "1 Semaine": {"period": "7d", "interval": "1h"},    
    "1 Mois": {"period": "1mo", "interval": "1d"},      
    "1 An": {"period": "1y", "interval": "1wk"},        
    "Depuis le début (Max)": {"period": "max", "interval": "1mo"} 
}

params_temps = mapping_periodes[periode_choisie]

# --- NOUVEAU : CALCULATEUR DE POSITION (MONEY MANAGEMENT) ---
st.sidebar.divider()
st.sidebar.subheader("💰 Gestion du Capital")
capital = st.sidebar.number_input("Votre capital total (USD) :", min_value=10, value=1000, step=50)
profil = st.sidebar.selectbox("Profil de Risque :", ["Prudent (5%)", "Équilibré (10%)", "Agressif (25%)"])

# Correspondance du pourcentage d'allocation selon le profil
mapping_profils = {"Prudent (5%)": 0.05, "Équilibré (10%)": 0.10, "Agressif (25%)": 0.25}
allocation_pct = mapping_profils[profil]

lancer = st.sidebar.button("🔍 ANALYSER")

# 5. ANALYSE ET AFFICHAGE
if lancer:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Graphique ({periode_choisie}) : {ticker}")
        data = yf.download(ticker, period=params_temps["period"], interval=params_temps["interval"])
        
        if not data.empty:
            # --- LE GRAPHIQUE EN BOUGIES JAPONAISES ---
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'].squeeze(),
                high=data['High'].squeeze(),
                low=data['Low'].squeeze(),
                close=data['Close'].squeeze(),
                increasing_line_color='#26a69a', 
                decreasing_line_color='#ef5350'  
            )])
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- EXTRACTION DES PRIX ---
            derniere_ligne = data['Close'].iloc[-1]
            prix_actuel = float(derniere_ligne.values[0]) if hasattr(derniere_ligne, 'values') else float(derniere_ligne)
                
            ligne_prec = data['Close'].iloc[0] 
            prix_prec = float(ligne_prec.values[0]) if hasattr(ligne_prec, 'values') else float(ligne_prec)

            variation = ((prix_actuel - prix_prec) / prix_prec) * 100
            
            # --- CALCULATEUR DE VOLATILITÉ ET RISQUE ---
            prix_plats = data['Close'].squeeze()
            rendements = prix_plats.pct_change().dropna()

            std_brut = rendements.std()
            if hasattr(std_brut, 'iloc'):
                volatilit_score = float(std_brut.iloc[0] * 100)
            elif hasattr(std_brut, 'values') and len(std_brut.values) > 0:
                volatilit_score = float(std_brut.values[0] * 100)
            else:
                volatilit_score = float(std_brut * 100)

            # --- CALCULATEUR DE RSI ---
            delta = prix_plats.diff()
            hausse = delta.clip(lower=0)
            baisse = -1 * delta.clip(upper=0)
            
            fenetre = 14 if len(prix_plats) > 14 else len(prix_plats)
            if fenetre > 2:
                ma_hausse = hausse.rolling(window=fenetre).mean()
                ma_baisse = baisse.rolling(window=fenetre).mean()
                rs = ma_hausse / ma_baisse
                rsi_series = 100 - (100 / (1 + rs))
                
                derniere_val_rsi = rsi_series.iloc[-1]
                rsi = float(derniere_val_rsi.values[0]) if hasattr(derniere_val_rsi, 'values') else float(derniere_val_rsi)
            else:
                rsi = 50.0

            # --- AFFICHAGE EN 3 COLONNES DES MÉTRIQUES ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Prix Actuel", f"{prix_actuel:.2f} USD", f"{variation:.2f}% (période)")
            
            with m2:
                if volatilit_score < 1.5:
                    st.metric("Niveau de Risque", "Faible 🟢", f"Volatilité: {volatilit_score:.2f}%")
                elif 1.5 <= volatilit_score <= 3.5:
                    st.metric("Niveau de Risque", "Modéré 🟡", f"Volatilité: {volatilit_score:.2f}%")
                else:
                    st.metric("Niveau de Risque", "Élevé 🔴", f"Volatilité: {volatilit_score:.2f}%")

            with m3:
                if rsi >= 70:
                    st.metric("Indicateur RSI", f"{rsi:.1f} 🔥", "Suracheté (Cher)")
                elif rsi <= 30:
                    st.metric("Indicateur RSI", f"{rsi:.1f} 🛒", "Survendu (Soldé)")
                else:
                    st.metric("Indicateur RSI", f"{rsi:.1f} ⚖️", "Zone Neutre")
                    
        else:
            st.error("Impossible de charger les données financières pour cette période.")
        
    with col2:
        st.subheader("Sentiment des News")
        url = f"https://news.google.com/rss/search?q={ticker}&hl=fr&gl=FR&ceid=FR:fr"
        flux = feedparser.parse(url)
        scores = []
        for article in flux.entries[:5]:
            s = analyser_sentiment(article.title)
            scores.append(s)
            st.markdown(f"**Score {s:.2f}** | [{article.title[:60]}...]({article.link})")
        
        if scores:
            moyenne = sum(scores) / len(scores)
            st.write(f"### Score Global : {moyenne:.2f}")
            if moyenne > 0.05: st.success("SENTIMENT POSITIF")
            elif moyenne < -0.05: st.error("SENTIMENT NÉGATIF")
            else: st.warning("SENTIMENT NEUTRE")
            
            # Affichage dynamique des résultats du Money Management dans la barre latérale
            montant_conseille = capital * allocation_pct
            nombre_actions = montant_conseille / prix_actuel
            
            st.sidebar.divider()
            st.sidebar.write("📊 **Conseil de Position**")
            st.sidebar.info(f"Montant Max : **{montant_conseille:.2f} USD**")
            st.sidebar.caption(f"Soit environ **{nombre_actions:.2f}** unité(s) de {ticker}")

            # Jauge de psychologie
            st.sidebar.divider()
            st.sidebar.write("🌡️ **Psychologie**")
            jauge = int((moyenne + 1) * 50)
            st.sidebar.progress(jauge)
        else:
            moyenne = 0

    # --- ÉLÉMENTS SUR TOUTE LA LARGEUR ---
    st.divider()
    
    # --- BLOC CONSEILLER IA ---
    st.subheader("💡 Avis de l'Assistant IA")
    if moyenne > 0.25:
        st.info("✅ **Signal Favorable** : Le sentiment global est très positif. Si le graphique montre une tendance stable ou montante, cela pourrait être un bon moment pour envisager un investissement.")
    elif moyenne < -0.25:
        st.warning("❌ **Prudence** : Le sentiment est négatif. Il y a beaucoup de mauvaises nouvelles. Il vaut mieux attendre que la situation se calme avant d'acheter.")
    else:
        st.write("⚖️ **Position Neutre** : Les news sont partagées ou peu d'informations circulent. Pas de signal fort pour le moment.")
    st.caption("⚠️ Rappel : Ceci est une analyse basée sur le sentiment des news, ce n'est pas un conseil financier officiel.")

    # --- TABLEAU DE RÉFÉRENCE ---
    st.subheader("📊 Comparaison Marché")
    ref_ticker = "^GSPC" if mode == "Bourse" else "BTC-USD"
    ref_data = yf.download(ref_ticker, period="1mo", interval="1d")
    
    if not ref_data.empty and not data.empty:
        val_debut = ref_data['Close'].iloc[0]
        val_fin = ref_data['Close'].iloc[-1]
        val_debut = float(val_debut.values[0]) if hasattr(val_debut, 'values') else float(val_debut)
        val_fin = float(val_fin.values[0]) if hasattr(val_fin, 'values') else float(val_fin)
        
        ref_var = ((val_fin - val_debut) / val_debut) * 100
        
        df_comp = pd.DataFrame({
            "Actif": [ticker, f"Référence ({ref_ticker})"],
            "Variation 30j": [f"{variation:.2f}%", f"{ref_var:.2f}%"]
        })
        st.table(df_comp)

    # --- ANALYSE DES MOTS-CLÉS ---
    st.subheader("🏷️ Thématiques détectées")
    texte_complet = " ".join([art.title for art in flux.entries]).lower()
    
    dictionnaire_mots = {
        "🔥": ["record", "hausse", "sommet", "croissance", "gain"],
        "⚠️": ["baisse", "chute", "danger", "perte", "risque"],
        "💰": ["banque", "investissement", "etf", "bourse", "dividende"],
        "🌐": ["adoption", "crypto", "blockchain", "web3", "technologie"],
        "⚖️": ["procès", "sec", "régulation", "loi", "interdiction"]
    }

    col_tags1, col_tags2 = st.columns(2)
    count = 0

    for icone, mots in dictionnaire_mots.items():
        for m in mots:
            if m in texte_complet:
                target_col = col_tags1 if count % 2 == 0 else col_tags2
                target_col.info(f"{icone} {m.upper()}")
                count += 1
    
    if count == 0:
        st.write("Aucune thématique majeure détectée dans les titres.")