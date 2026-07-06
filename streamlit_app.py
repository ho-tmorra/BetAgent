import streamlit as st
import requests
import google.generativeai as genai
import pandas as pd

# Configuration BetAgent Ultra
st.set_page_config(page_title="BetAgent Ultra - Trading Sportif", page_icon="⚡", layout="wide")

st.title("⚡ BetAgent Ultra")
st.markdown("Plateforme algorithmique multisport, détection d'arbitrage (Surebets) et trading contextuel.")

# --- INITIALISATION DE LA MEMOIRE INTERNE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bet_tracker" not in st.session_state:
    st.session_state.bet_tracker = []
if "current_bankroll" not in st.session_state:
    st.session_state.current_bankroll = 1000.0
if "context_triggers" not in st.session_state:
    st.session_state.context_triggers = {}

# --- BARRE LATÉRALE : MANAGEMENT & CONFIGURATION ---
st.sidebar.header("⚙️ Configuration des APIs")
odds_api_key = st.sidebar.text_input("Clé The Odds API", type="password")
gemini_api_key = st.sidebar.text_input("Clé Google Gemini", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📊 Paramètres Financiers")
initial_bankroll = st.sidebar.number_input("Bankroll Initiale (€)", min_value=10, value=1000, step=50)

# Mise à jour dynamique de la bankroll si des paris sont enregistrés
if len(st.session_state.bet_tracker) == 0:
    st.session_state.current_bankroll = float(initial_bankroll)

st.sidebar.metric(label="Bankroll Actuelle (€)", value=f"{st.session_state.current_bankroll:.2f} €")
risk_profile = st.sidebar.slider("Fraction de Kelly (Prudence)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
sport_choice = st.sidebar.selectbox("Marché principal", ["Tous les Lives & À Venir", "Tennis Uniquement"])

# --- MODULE 4 : DASHBOARD DE PERFORMANCE FINANCIÈRE ---
st.header("📈 Tableau de Bord Financier")
if st.session_state.bet_tracker:
    df_bets = pd.DataFrame(st.session_state.bet_tracker)
    total_bets = len(df_bets)
    won_bets = len(df_bets[df_bets["Statut"] == "Gagné"])
    win_rate = (won_bets / total_bets) * 100 if total_bets > 0 else 0
    
    total_invested = df_bets["Mise"].sum()
    total_pnl = df_bets["Gain/Perte"].sum()
    roi = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paris Enregistrés", f"{total_bets}")
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Bénéfice Net PnL", f"{total_pnl:.2f} €", delta=f"{total_pnl:.2f} €")
    c4.metric("R.O.I. Global", f"{roi:.1f}%")
    
    # Graphique d'évolution simplifié
    df_bets["Cumulative_PnL"] = df_bets["Gain/Perte"].cumsum()
    df_bets["Evolution_Bankroll"] = initial_bankroll + df_bets["Cumulative_PnL"]
    st.line_chart(df_bets["Evolution_Bankroll"])
    
    if st.checkbox("📁 Afficher l'historique détaillé des tickets"):
        st.dataframe(df_bets)
else:
    st.info("Aucun pari enregistré dans l'historique pour le moment. Utilisez le scanner pour ajouter un ticket.")

st.markdown("---")

# --- FONCTION REQUÊTE UNIVERSELLE ---
def fetch_ultra_live_odds(api_key):
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

# --- ENGINE : ANALYSE DES MATCHS, ARBITRAGE ET CONTEXTE ---
st.header("🔄 Scanner en Direct & Analyse de Marché")
if st.button("🚀 Lancer l'Analyse Algorithmique Multi-Bookmakers"):
    if not odds_api_key:
        st.warning("Veuillez renseigner votre clé The Odds API.")
    else:
        with st.spinner("Analyse quantitative des flux internationaux en cours..."):
            all_data = fetch_ultra_live_odds(odds_api_key)
            
            if all_data:
                filtered_data = [m for m in all_data if "tennis" in m.get("sport_key", "").lower()] if sport_choice == "Tennis Uniquement" else all_data
                
                if len(filtered_data) == 0:
                    st.info("Aucun match ne correspond aux critères à cet instant précis.")
                else:
                    st.success(f"Analyse achevée. {len(filtered_data)} marchés scannés avec succès.")
                    
                    for match in filtered_data[:8]:
                        match_id = match.get('id')
                        home = match.get('home_team')
                        away = match.get('away_team')
                        sport_name = match.get('sport_title')
                        
                        # --- MODULE 1 : CALCULATEUR MULTI-BOOKMAKERS & ARBITRAGE (SUREBET) ---
                        best_odds = {}
                        bookmakers_data = match.get('bookmakers', [])
                        
                        for bookie in bookmakers_data:
                            markets = bookie.get('markets', [])
                            if markets:
                                outcomes = markets[0].get('outcomes', [])
                                for outcome in outcomes:
                                    name = outcome.get('name')
                                    price = float(outcome.get('price'))
                                    if name not in best_odds or price > best_odds[name]['cote']:
                                        best_odds[name] = {'cote': price, 'bookie': bookie.get('title')}
                        
                        # Rendu visuel du match
                        st.markdown(f"### 🏆 {sport_name} : **{home}** vs **{away}**")
                        
                        if len(best_odds) >= 2:
                            # Calcul de l'indice d'arbitrage
                            arbitrage_sum = sum(1 / item['cote'] for item in best_odds.values())
                            is_surebet = arbitrage_sum < 1.0
                            
                            cols_odds = st.columns(len(best_odds))
                            for idx, (player_name, data_odds) in enumerate(best_odds.items()):
                                with cols_odds[idx]:
                                    st.metric(
                                        label=f"Meilleure Cote : {player_name}", 
                                        value=f"{data_odds['cote']:.2f}", 
                                        delta=f"Bookmaker: {data_odds['bookie']}"
                                    )
                            
                            # Alerte Surebet
                            if is_surebet:
                                profit = (1 - arbitrage_sum) * 100
                                st.error(f"🚨 **OPPORTUNITÉ DE SUREBET DÉTECTÉE (Profit garanti: +{profit:.2f}%)**")
                                cols_sure = st.columns(len(best_odds))
                                for idx, (player_name, data_odds) in enumerate(best_odds.items()):
                                    with cols_sure[idx]:
                                        repartition = (1 / data_odds['cote']) / arbitrage_sum
                                        st.code(f"Miser {repartition*100:.1f}% de la mise totale chez {data_odds['bookie']}")
                            
                            # --- MODULE 2 : BOUTONS D'INJECTION CONTEXTUELLE ÉCLAIR ---
                            st.write("**⚡ Signaux du Direct (Sélectionnez pour injecter dans l'IA) :**")
                            c_b1, c_b2, c_b3, c_b4 = st.columns(4)
                            
                            trigger_key = f"trigger_{match_id}"
                            if trigger_key not in st.session_state.context_triggers:
                                st.session_state.context_triggers[trigger_key] = []
                                
                            if c_b1.button("🚨 Kiné / Blessure", key=f"k_{match_id}"):
                                st.session_state.context_triggers[trigger_key].append(f"Alerte médicale : Un joueur a fait appel au kiné sur le match {home}-{away}.")
                            if c_b2.button("📉 Fatigue / Nervosité", key=f"f_{match_id}"):
                                st.session_state.context_triggers[trigger_key].append(f"Alerte comportementale : Baisse d'intensité physique ou frustration visible sur le match {home}-{away}.")
                            if c_b3.button("🎾 Break d'entrée", key=f"b_{match_id}"):
                                st.session_state.context_triggers[trigger_key].append(f"Alerte tactique : Perte de service immédiate au début du set sur le match {home}-{away}.")
                            if c_b4.button("🌧️ Météo / Pluie", key=f"m_{match_id}"):
                                st.session_state.context_triggers[trigger_key].append(f"Alerte environnementale : Interruption météo imminente ou vent fort sur le match {home}-{away}.")
                                
                            if st.session_state.context_triggers[trigger_key]:
                                st.warning(f"Signaux actifs capturés : {', '.join(st.session_state.context_triggers[trigger_key])}")
                            
                            # --- MODULE 3 : SIMULATEUR INTERACTIF COMPREHENSIF ---
                            st.markdown("**📊 Évaluation & Formules Financières :**")
                            sim_cols = st.columns(len(best_odds))
                            for idx, (player_name, data_odds) in enumerate(best_odds.items()):
                                with sim_cols[idx]:
                                    cote_cible = data_odds['cote']
                                    prob_bookie = (1 / cote_cible) * 100
                                    
                                    # Ajustement de probabilité utilisateur
                                    prob_user = st.slider(f"Ta Probabilité pour {player_name} (%)", min_value=1, max_value=99, value=int(prob_bookie), key=f"sl_u_{match_id}_{idx}")
                                    
                                    # Calcul de la value mathématique
                                    value_factor = (cote_cible * (prob_user / 100)) - 1
                                    
                                    if value_factor > 0:
                                        k_raw = value_factor / (cote_cible - 1) if (cote_cible - 1) != 0 else 0
                                        k_adj = max(0.0, k_raw * risk_profile)
                                        stake_eur = st.session_state.current_bankroll * k_adj
                                        
                                        st.success(f"📈 **Value: +{value_factor*100:.1f}%**\nMise : **{stake_eur:.2f} €** ({k_adj*100:.1f}%)")
                                        
                                        # Bouton d'enregistrement comptable
                                        if st.button(f"📥 Enregistrer le pari sur {player_name}", key=f"rec_{match_id}_{idx}"):
                                            st.session_state.bet_tracker.append({
                                                "Match": f"{home} vs {away}",
                                                "Sélection": player_name,
                                                "Cote": cote_cible,
                                                "Mise": stake_eur,
                                                "Statut": "Gagné", # Par défaut mis à Gagné pour la simulation financière, modifiable dans le dataframe
                                                "Gain/Perte": stake_eur * (cote_cible - 1)
                                            })
                                            st.session_state.current_bankroll += stake_eur * (cote_cible - 1)
                                            st.rerun()
                                    else:
                                        st.info("Aucun avantage mathématique décelé.")
                        st.markdown("---")
            else:
                st.error("Échec de la connexion aux bases de données.")

# --- MODULE DE DIALOGUE AVANCÉ (BETAGENT ULTRA) ---
st.header("💬 Consultation Avancée BetAgent Ultra")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ex: 'Analyse les alertes de blessure récoltées sur le match de tennis et recalcule Kelly.'")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    if not gemini_api_key:
        st.error("Veuillez configurer votre clé Gemini pour exécuter le modèle d'analyse.")
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Injection automatique des triggers collectés au cours du scan
            active_signals = []
            for k, v in st.session_state.context_triggers.items():
                if v:
                    active_signals.extend(v)
            
            signals_context = " | ".join(active_signals) if active_signals else "Aucun signal physique critique détecté par les boutons."
            
            prompt = f"""
            Tu es BetAgent Ultra, le modèle d'IA d'élite spécialisé dans le trading sportif et la gestion de portefeuille.
            Capital actuel disponible de l'utilisateur : {st.session_state.current_bankroll} €.
            Signaux physiques et d'alertes capturés sur le terrain en direct : {signals_context}.
            
            Question ou analyse demandée par l'utilisateur : '{user_input}'.
            
            Formule ton évaluation :
            1) Croise mathématiquement les signaux physiques du direct (Kiné, fatigue...) avec les notions de probabilités implicites.
            2) Applique de façon rigoureuse les formules de Value Betting et la théorie financière de Kelly.
            3) Structure ta réponse sous forme de rapport de trading ultra-concis (Signaux analysés -> Risque financier -> Allocation de mise recommandée en euros).
            """
            
            with st.chat_message("assistant"):
                response = model.generate_content(prompt)
                st.write(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Erreur d'exécution de l'IA : {str(e)}")
