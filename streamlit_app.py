import streamlit as st
import requests
import google.generativeai as genai

# Configuration de la page de l'application
st.set_page_config(page_title="Copilote Paris", page_icon="🎾", layout="wide")

st.title("🎾 Mon Conseiller de Paris Sportifs")
st.markdown("Analyse de cotes en direct et interaction stratégique.")

# Barre latérale pour la configuration des clés
st.sidebar.header("⚙️ Configuration des Clés API")
odds_api_key = st.sidebar.text_input("Clé The Odds API", type="password")
gemini_api_key = st.sidebar.text_input("Clé Google Gemini", type="password")

sport_choice = st.sidebar.selectbox("Choisir un sport", ["Tennis (ATP/WTA)", "Football", "Basketball"])

# Initialisation de la mémoire du chat pour l'aspect conseiller interactif
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def fetch_live_odds(api_key, sport):
    # Clé par défaut pour le Tennis ATP ou Ligue 1 pour le foot
    sport_key = "tennis_atp" if "Tennis" in sport else "soccer_france_ligue1"
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    try:
        response = requests.get(url, params=params)
        return response.json() if response.status_code == 200 else None
    except:
        return None

# Interface de récupération en direct
if st.button("🔄 Scanner les cotes en direct"):
    if not odds_api_key:
        st.warning("Veuillez renseigner votre clé The Odds API.")
    else:
        with st.spinner("Analyse du flux live..."):
            data = fetch_live_odds(odds_api_key, sport_choice)
            if data:
                st.success(f"Analyse terminée. {len(data)} matchs trouvés.")
                for match in data[:5]: # Affiche les 5 premiers matchs trouvés
                    st.subheader(f"🏆 {match.get('home_team')} vs {match.get('away_team')}")
                    st.write("Match disponible sur les marchés enregistrés.")
                    st.markdown("---")
            else:
                st.info("Aucun match en direct ou à venir trouvé sur ce marché actuellement.")

# Section Conseiller Interactif (Le Chat)
st.markdown("---")
st.header("💬 Discuter avec votre Conseiller de Paris")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ex: 'Tu as vu la blessure de Alcaraz au dos pendant son match ?'")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    if not gemini_api_key:
        st.error("Veuillez configurer votre clé Gemini dans la barre latérale pour activer le conseiller.")
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Message de contexte pour spécialiser l'IA
            prompt = f"Tu es un expert en paris sportifs et statistiques de tennis. Tu analyses les données contextuelles de manière froide et mathématique. Le parieur te dit : {user_input}. Donne ton avis sur l'impact sur les cotes et si cela crée une opportunité de value ou un piège."
            
            with st.chat_message("assistant"):
                response = model.generate_content(prompt)
                st.write(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Erreur avec le moteur Gemini : {str(e)}")
