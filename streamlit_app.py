import streamlit as st
import requests
import google.generativeai as genai

# Configuration complète de la page BetAgent
st.set_page_config(page_title="BetAgent - Multisport Live", page_icon="🎾", layout="wide")

st.title("🎾 BetAgent - Conseiller de Paris Multisport")
st.markdown("Scanner universel de cotes en direct (In-Play) et Intelligence Artificielle.")

# Barre latérale : Gestion des clés API et options
st.sidebar.header("⚙️ Configuration des Clés API")
odds_api_key = st.sidebar.text_input("Clé The Odds API", type="password")
gemini_api_key = st.sidebar.text_input("Clé Google Gemini", type="password")

# On passe le sélecteur en mode global par défaut pour l'utilisateur
sport_choice = st.sidebar.selectbox("Filtre d'affichage de l'interface", ["Tous les Lives & À Venir", "Tennis Uniquement"])

# Initialisation de l'historique du conseiller
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def fetch_all_live_and_upcoming_odds(api_key):
    """
    Appelle le flux universel d'API pour récupérer TOUS les sports en direct.
    """
    # 'upcoming' est la clé magique pour récupérer le live de TOUS les sports ouverts
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds/"
    
    params = {
        "apiKey": api_key,
        "regions": "eu",       # Bookmakers européens (Winamax, Betclic, Unibet...)
        "markets": "h2h",       # Marché principal : Victoire 1 ou 2 (ou Nul)
        "oddsFormat": "decimal"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur d'API : Code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur réseau : {str(e)}")
        return None

# Action du Scanner principal
if st.button("🔄 Lancer le Scanner Universel Live"):
    if not odds_api_key:
        st.warning("Veuillez renseigner votre clé The Odds API dans la barre latérale.")
    else:
        with st.spinner("Recherche active de toutes les cotes en cours sur le marché international..."):
            all_data = fetch_all_live_and_upcoming_odds(odds_api_key)
            
            if all_data:
                # Filtrage optionnel à la volée pour le confort visuel de l'utilisateur
                if sport_choice == "Tennis Uniquement":
                    filtered_data = [m for m in all_data if "tennis" in m.get("sport_key", "").lower()]
                else:
                    filtered_data = all_data

                match_count = len(filtered_data)
                
                if match_count == 0:
                    st.info("Aucun match en direct disponible pour le filtre sélectionné. L'API est connectée mais le marché est calme.")
                else:
                    st.success(f"🔥 Connexion Réussie ! {match_count} événements majeurs détectés actuellement.")
                    
                    # On affiche les matchs sous forme de cartes propres
                    for match in filtered_data[:12]:  # Affiche jusqu'à 12 gros événements en simultané
                        home = match.get('home_team')
                        away = match.get('away_team')
                        sport_name = match.get('sport_title')
                        
                        # Création d'un bloc d'affichage visuel
                        with st.container():
                            st.markdown(f"### 🏆 {sport_name} : **{home}** vs **{away}**")
                            
                            # Extraction des cotes si disponibles sur le premier bookmaker européen trouvé
                            bookmakers = match.get('bookmakers', [])
                            if bookmakers:
                                first_bookie = bookmakers[0]
                                st.caption(f"Exemple de cotes via {first_bookie.get('title')} :")
                                markets = first_bookie.get('markets', [])
                                if markets:
                                    outcomes = markets[0].get('outcomes', [])
                                    cols = st.columns(len(outcomes))
                                    for idx, outcome in enumerate(outcomes):
                                        with cols[idx]:
                                            st.metric(label=outcome.get('name'), value=str(outcome.get('price')))
                            else:
                                st.write("⚠️ Match en cours - Cotes en cours de réajustement chez les bookmakers.")
                            st.markdown("---")
            else:
                st.error("Impossible de récupérer les données. Vérifiez la validité de votre clé API ou vos quotas d'appels.")

# Section Conseiller Interactif (Le Chat)
st.markdown("---")
st.header("💬 Discuter stratégiquement avec BetAgent")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ex: 'Wimbledon : Alcaraz concède un break d'entrée au 3ème set, sa cote passe à 3.10. Value ?'")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    if not gemini_api_key:
        st.error("Veuillez configurer votre clé Gemini dans la barre latérale pour activer les conseils de l'IA.")
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""
            Tu es BetAgent, un conseiller expert en paris sportifs algorithmiques. Tu es froid, analytique et tu ne jures que par les probabilités et le concept de 'Value Betting'. 
            Le parieur te donne cet élément contextuel en direct : '{user_input}'.
            Analyse l'impact de cette information sur la psychologie du match, la viabilité de la cote actuelle, et donne une recommandation claire (Mise prudente, Value confirmée, ou Piège à éviter).
            """
            
            with st.chat_message("assistant"):
                response = model.generate_content(prompt)
                st.write(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Erreur d'exécution de l'IA : {str(e)}")
