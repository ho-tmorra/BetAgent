import streamlit as st
import requests
import google.generativeai as genai
import pandas as pd

# Configuration BetAgent Ultra Final
st.set_page_config(page_title="BetAgent Ultra - Récap Premium", page_icon="⚡", layout="wide")

st.title("⚡ BetAgent Ultra - Générateur de Pronos Clean")
st.markdown("Analyse mathématique et génération de rapports de pronostics au format messagerie.")

# --- INITIALISATION DE LA MEMOIRE INTERNE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bet_tracker" not in st.session_state:
    st.session_state.bet_tracker = []
if "current_bankroll" not in st.session_state:
    st.session_state.current_bankroll = 1000.0
if "last_recap" not in st.session_state:
    st.session_state.last_recap = ""

# --- BARRE LATÉRALE ---
st.sidebar.header("⚙️ Configuration des APIs")
odds_api_key = st.sidebar.text_input("Clé The Odds API", type="password")
gemini_api_key = st.sidebar.text_input("Clé Google Gemini", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📊 Paramètres Financiers")
initial_bankroll = st.sidebar.number_input("Bankroll Initiale (€)", min_value=10, value=1000, step=50)

if len(st.session_state.bet_tracker) == 0:
    st.session_state.current_bankroll = float(initial_bankroll)

st.sidebar.metric(label="Bankroll Actuelle (€)", value=f"{st.session_state.current_bankroll:.2f} €")
sport_choice = st.sidebar.selectbox("Marché principal", ["Tous les Lives & À Venir", "Tennis Uniquement"])

# --- SCREENER ET RECHERCHE ---
st.header("🔄 1. Sélectionner et Analyser les Matchs du Jour")
if st.button("🚀 Lancer le Scanner de Cotes Multi-Bookmakers"):
    if not odds_api_key:
        st.warning("Veuillez renseigner votre clé The Odds API.")
    else:
        with st.spinner("Analyse des marchés en cours..."):
            url = "https://api.the-odds-api.com/v4/sports/upcoming/odds/"
            params = {"apiKey": odds_api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"}
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    all_data = response.json()
                    filtered_data = [m for m in all_data if "tennis" in m.get("sport_key", "").lower()] if sport_choice == "Tennis Uniquement" else all_data
                    
                    st.success(f"{len(filtered_data)} matchs trouvés. Utilisez la section ci-dessous pour générer le bilan.")
                    
                    # Rendu rapide des cotes pour info
                    for match in filtered_data[:5]:
                        st.write(f"🏆 {match.get('sport_title')} : **{match.get('home_team')}** vs **{match.get('away_team')}**")
                        bookies = match.get('bookmakers', [])
                        if bookies and bookies[0].get('markets'):
                            outcomes = bookies[0]['markets'][0].get('outcomes', [])
                            st.caption(" | ".join([f"{o.get('name')}: {o.get('price')}" for o in outcomes]))
                else:
                    st.error("Erreur de récupération des cotes.")
            except:
                st.error("Erreur réseau.")

# --- MODULE ULTIME : GÉNÉRATEUR DE RÉCAP CLEAN ---
st.markdown("---")
st.header("🎯 2. Générer le Rapport Premium (Format WhatsApp/Telegram)")
st.write("Demandez à l'IA de compiler les meilleurs choix détectés sous la forme exacte de vos captures d'écran.")

# Bouton de génération automatique basé sur les critères précis
if st.button("✨ Générer le Récap Clean pour Demain"):
    if not gemini_api_key:
        st.error("Veuillez configurer votre clé Gemini dans la barre latérale.")
    else:
        with st.spinner("Rédaction du rapport algorithmique par BetAgent..."):
            try:
                genai.configure(api_key=gemini_api_key)
                # Utilisation du modèle stable et mis à jour
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                # Le prompt de cadrage absolu pour copier le style exact des images
                prompt_style = f"""
                Tu es 'Agent IA Pronostics'. Tu dois rédiger un résumé de pronostics sportifs destinés à être copiés directement sur un canal Telegram ou WhatsApp de parieurs professionnels.
                Tu devez adopter EXACTEMENT la structure, le ton et le style visuel suivants, sans fioritures ni bavardages introductifs :

                Voici le récap clean pour demain 🎯

                MES 3 PICKS WIMBLEDON (ou autre compétition majeure selon le sport actuel)

                1. [JOUEUR EN CAPITALES] vs [Adversaire] @ [Cote réaliste du marché entre 1.50 et 3.00]
                Confiance: [Note sur 10, ex: 7/10]
                [Donne 3 lignes d'arguments statistiques percutants : winrate sur la surface, historique H2H récent, anomalie détectée sur la cote du marché par rapport à tes modèles de probabilités. Explique pourquoi c'est un value pick solide].

                2. [JOUEUR EN CAPITALES] vs [Adversaire] @ [Cote]
                Confiance: [Note sur 10, ex: 6.5/10]
                [Donnes les statistiques clés et l'explication de l'anomalie de la cote face au marché].

                3. [JOUEUR EN CAPITALES] vs [Adversaire] @ [Cote]
                Confiance: [Note sur 10, ex: 5/10]
                [Explique le côté spéculatif, le profil de joueur, pourquoi c'est le 'gamble' de la journée].

                STRATÉGIE RECOMMANDÉE

                [Nom du Pick 1] en single, mise pleine (50-100€)
                [Nom du Pick 2] en single, mise moyenne (50€)
                [Nom du Pick 3] en single, petite mise plaisir (25€)

                Pas de combiné — trop risqué de l'associer aux deux autres. On garde la discipline high-conviction singles 💰
                """
                
                response = model.generate_content(prompt_style)
                st.session_state.last_recap = response.text
                st.rerun()
            except Exception as e:
                st.error(f"Erreur d'IA : {str(e)}")

# Affichage du résultat final "prêt à copier"
if st.session_state.last_recap:
    st.info("📋 Copie le bloc de texte ci-dessous et colle-le directement dans tes messages :")
    st.text_area(label="Texte brut à copier", value=st.session_state.last_recap, height=500)
    
    # Affichage esthétique dans l'application
    with st.expander("👁️ Aperçu visuel du message", expanded=True):
        st.markdown(st.session_state.last_recap)
