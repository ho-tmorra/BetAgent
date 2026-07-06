import streamlit as st
import requests
import google.generativeai as genai

# Configuration complète de la page BetAgent Pro
st.set_page_config(page_title="BetAgent Pro - Algorithmique", page_icon="📈", layout="wide")

st.title("📈 BetAgent Pro - Scanner de Value & Money Management")
st.markdown("Analyse mathématique des cotes en direct et optimisation des mises via le Critère de Kelly.")

# Barre latérale : Clés API et Gestion de Bankroll
st.sidebar.header("⚙️ Configuration Globale")
odds_api_key = st.sidebar.text_input("Clé The Odds API", type="password")
gemini_api_key = st.sidebar.text_input("Clé Google Gemini", type="password")

st.sidebar.markdown("---")
st.sidebar.header("💰 Gestion de Bankroll")
bankroll = st.sidebar.number_input("Ta Bankroll totale (€)", min_value=10, value=1000, step=50)
risk_profile = st.sidebar.slider("Agressivité (Fraction de Kelly)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, 
                                 help="0.5 signifie qu'on mise la moitié de ce que recommande la formule de Kelly brute pour maximiser la sécurité.")

sport_choice = st.sidebar.selectbox("Filtre d'affichage", ["Tous les Lives & À Venir", "Tennis Uniquement"])

# Initialisation de l'historique du conseiller
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def fetch_all_live_and_upcoming_odds(api_key):
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds/"
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

# Action du Scanner principal
if st.button("🔄 Lancer le Scanner & Calculateur de Value"):
    if not odds_api_key:
        st.warning("Veuillez renseigner votre clé The Odds API dans la barre latérale.")
    else:
        with st.spinner("Analyse des cotes du marché international en cours..."):
            all_data = fetch_all_live_and_upcoming_odds(odds_api_key)
            
            if all_data:
                if sport_choice == "Tennis Uniquement":
                    filtered_data = [m for m in all_data if "tennis" in m.get("sport_key", "").lower()]
                else:
                    filtered_data = all_data

                match_count = len(filtered_data)
                
                if match_count == 0:
                    st.info("Aucun match disponible actuellement pour ce filtre.")
                else:
                    st.success(f"🔥 Connexion Réussie ! {match_count} événements analysés.")
                    
                    for match in filtered_data[:10]:
                        home = match.get('home_team')
                        away = match.get('away_team')
                        sport_name = match.get('sport_title')
                        
                        with st.container():
                            st.markdown(f"### 🏆 {sport_name} : **{home}** vs **{away}**")
                            
                            bookmakers = match.get('bookmakers', [])
                            if bookmakers:
                                first_bookie = bookmakers[0]
                                markets = first_bookie.get('markets', [])
                                if markets:
                                    outcomes = markets[0].get('outcomes', [])
                                    
                                    # Affichage des cotes brutes
                                    cols = st.columns(len(outcomes))
                                    for idx, outcome in enumerate(outcomes):
                                        with cols[idx]:
                                            cote = outcome.get('price')
                                            prob_implied = (1 / cote) * 100
                                            st.metric(
                                                label=f"{outcome.get('name')} (Cote)", 
                                                value=f"{cote:.2f}", 
                                                delta=f"Probabilité bookie: {prob_implied:.1f}%",
                                                delta_color="off"
                                            )
                                            
                                    # Zone d'analyse algorithmique rapide pour chaque match
                                    st.markdown("**📊 Simulateur de Value-Bet Rapide :**")
                                    sim_cols = st.columns(len(outcomes))
                                    for idx, outcome in enumerate(outcomes):
                                        with sim_cols[idx]:
                                            cote = outcome.get('price')
                                            # On crée un petit bouton curseur fictif pour simuler si l'utilisateur estime une autre probabilité
                                            prob_estimee = st.slider(f"Ta probabilité estimée pour {outcome.get('name')} (%)", min_value=1, max_value=99, value=int(100/len(outcomes)), key=f"slider_{match.get('id')}_{idx}")
                                            
                                            prob_estimee_decimal = prob_estimee / 100
                                            value = (cote * prob_estimee_decimal) - 1
                                            
                                            if value > 0:
                                                # Calcul de Kelly
                                                kelly_raw = (value / (cote - 1)) if (cote - 1) != 0 else 0
                                                kelly_adjusted = max(0.0, kelly_raw * risk_profile)
                                                mise_euros = bankroll * kelly_adjusted
                                                
                                                st.success(f"✅ **Value Détectée : +{value*100:.1f}%**\n\n👉 Mise Kelly conseillée : **{kelly_adjusted*100:.2f}%** de ta bankroll, soit **{mise_euros:.2f} €**")
                                            else:
                                                st.info("❌ Aucune value sur cette sélection.")
                            else:
                                st.write("⚠️ Cotes en cours de réajustement.")
                            st.markdown("---")
            else:
                st.error("Erreur de récupération des données.")

# Section Conseiller Interactif (Le Chat)
st.markdown("---")
st.header("💬 Consultation Avancée BetAgent")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ex: 'À Wimbledon, Djokovic montre des signes de frustration, sa cote en live est à 1.90. Calcule la value s'il a 60% de chances de gagner.'")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    if not gemini_api_key:
        st.error("Veuillez configurer votre clé Gemini dans la barre latérale.")
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""
            Tu es BetAgent Pro. Tu es un expert en probabilités appliquées aux paris sportifs. Tu as à ta disposition les formules du Critère de Kelly et du calcul de Value Bet.
            Le parieur te donne cette situation : '{user_input}'.
            
            Tu dois analyser la situation de manière froide :
            1) Estimer mathématiquement la pertinence de la cote citée.
            2) Appliquer la formule de la Value : (Cote * Probabilité_Estimée) - 1.
            3) Donner le pourcentage précis recommandé par la formule de Kelly si une Value est confirmée, et le traduire en euros en prenant comme exemple une bankroll théorique de {bankroll} € (ajustée avec un coefficient de prudence de {risk_profile}).
            Soyez concis, précis et purement mathématique.
            """
            
            with st.chat_message("assistant"):
                response = model.generate_content(prompt)
                st.write(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Erreur d'exécution de l'IA : {str(e)}")
