import streamlit as st
import requests
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import json
import re


def _extract_h2h_outcomes(match):
    outcomes = []
    for bookmaker in match.get("bookmakers", []):
        markets = bookmaker.get("markets", [])
        for market in markets:
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = outcome.get("price")
                    if name and isinstance(price, (int, float)) and price > 1.01:
                        outcomes.append({"bookmaker": bookmaker.get("title", "?"), "name": name, "price": float(price)})
    return outcomes


def _compute_value_candidates(matches, max_candidates=200):
    candidates = []
    for match in matches:
        sport = match.get("sport_title", "?")
        home = match.get("home_team", "?")
        away = match.get("away_team", "?")
        start = match.get("commence_time", "?")
        outcomes = _extract_h2h_outcomes(match)
        if len(outcomes) < 2:
            continue

        rows = pd.DataFrame(outcomes)
        if rows.empty:
            continue

        rows["implied_prob_raw"] = 1.0 / rows["price"]
        grouped = rows.groupby("name", as_index=False).agg(
            best_price=("price", "max"),
            avg_price=("price", "mean"),
            median_raw_prob=("implied_prob_raw", "median"),
            n_books=("bookmaker", "nunique")
        )
        if grouped.empty:
            continue

        sum_raw = grouped["median_raw_prob"].sum()
        if sum_raw <= 0:
            continue

        grouped["market_prob"] = grouped["median_raw_prob"] / sum_raw
        grouped["fair_odds"] = 1.0 / grouped["market_prob"]
        grouped["edge_pct"] = ((grouped["best_price"] * grouped["market_prob"]) - 1.0) * 100.0

        grouped = grouped.sort_values(by="edge_pct", ascending=False)
        for _, row in grouped.iterrows():
            candidates.append(
                {
                    "sport": sport,
                    "home": home,
                    "away": away,
                    "start": start,
                    "selection": row["name"],
                    "best_price": float(row["best_price"]),
                    "avg_price": float(row["avg_price"]),
                    "market_prob": float(row["market_prob"]),
                    "fair_odds": float(row["fair_odds"]),
                    "edge_pct": float(row["edge_pct"]),
                    "n_books": int(row["n_books"]),
                }
            )

    candidates = sorted(candidates, key=lambda x: x["edge_pct"], reverse=True)
    return candidates[:max_candidates]


def _kelly_fraction(price, win_prob):
    b = price - 1.0
    q = 1.0 - win_prob
    if b <= 0:
        return 0.0
    f = (b * win_prob - q) / b
    return max(0.0, f)


def _compute_tracker_metrics(bets, initial_bankroll):
    if not bets:
        return {
            "total_bets": 0,
            "settled_bets": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "staked": 0.0,
            "pnl": 0.0,
            "roi_pct": 0.0,
            "hit_rate_pct": 0.0,
            "current_bankroll": float(initial_bankroll),
        }

    df = pd.DataFrame(bets)
    if "status" not in df.columns:
        df["status"] = "pending"
    if "stake" not in df.columns:
        df["stake"] = 0.0
    if "odds" not in df.columns:
        df["odds"] = 1.0

    df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(1.0)

    def _row_pnl(row):
        if row["status"] == "won":
            return row["stake"] * (row["odds"] - 1.0)
        if row["status"] == "lost":
            return -row["stake"]
        return 0.0

    df["pnl"] = df.apply(_row_pnl, axis=1)

    settled = df[df["status"].isin(["won", "lost"])].copy()
    total_staked_settled = float(settled["stake"].sum()) if not settled.empty else 0.0
    total_pnl = float(df["pnl"].sum())
    wins = int((df["status"] == "won").sum())
    losses = int((df["status"] == "lost").sum())
    settled_count = wins + losses
    hit_rate = (wins / settled_count * 100.0) if settled_count > 0 else 0.0
    roi = (total_pnl / total_staked_settled * 100.0) if total_staked_settled > 0 else 0.0
    current_bankroll = float(initial_bankroll) + total_pnl

    return {
        "total_bets": int(len(df)),
        "settled_bets": settled_count,
        "wins": wins,
        "losses": losses,
        "pending": int((df["status"] == "pending").sum()),
        "staked": total_staked_settled,
        "pnl": total_pnl,
        "roi_pct": roi,
        "hit_rate_pct": hit_rate,
        "current_bankroll": current_bankroll,
    }


def _build_tracker_dataframe(bets):
    if not bets:
        return pd.DataFrame(columns=["date", "sport", "event", "selection", "odds", "stake", "status", "close_odds"])

    df = pd.DataFrame(bets).copy()
    for col, default_value in [("date", str(datetime.now().date())), ("sport", "?"), ("event", "?"), ("selection", "?"), ("status", "pending")]:
        if col not in df.columns:
            df[col] = default_value
    if "odds" not in df.columns:
        df["odds"] = 1.0
    if "stake" not in df.columns:
        df["stake"] = 0.0
    if "close_odds" not in df.columns:
        df["close_odds"] = None

    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(1.0)
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
    df["close_odds"] = pd.to_numeric(df["close_odds"], errors="coerce")

    def _pnl(row):
        if row["status"] == "won":
            return row["stake"] * (row["odds"] - 1.0)
        if row["status"] == "lost":
            return -row["stake"]
        return 0.0

    def _clv_pct(row):
        if pd.notnull(row["close_odds"]) and row["close_odds"] > 1.01 and row["odds"] > 1.01:
            return ((row["close_odds"] - row["odds"]) / row["odds"]) * 100.0
        return None

    df["PnL (€)"] = df.apply(_pnl, axis=1)
    df["CLV %"] = df.apply(_clv_pct, axis=1)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _extract_bets_from_screenshot(uploaded_file, gemini_api_key):
    if not gemini_api_key:
        raise ValueError("Clé Gemini requise pour analyser une image.")

    mime_type = uploaded_file.type or "image/jpeg"
    image_bytes = uploaded_file.getvalue()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
    Tu analyses un screenshot d'application de paris (ex: Winamax).
    Extrait uniquement les paris visibles et retourne STRICTEMENT un JSON valide (sans markdown) au format:
    {
      "bets": [
        {
          "date": "YYYY-MM-DD",
          "sport": "string",
          "event": "string",
          "selection": "string",
          "odds": 1.85,
          "stake": 25.0,
                    "potential_return": 46.25,
          "status": "pending|won|lost"
        }
      ]
    }

    Règles:
    - Si une donnée est absente, infère prudemment ou mets une valeur par défaut réaliste.
    - status = pending par défaut si non explicite.
    - odds et stake doivent être numériques (accepte format FR avec virgule dans ta lecture).
    - Si 'mise' absente mais 'gain potentiel' visible, renseigne potential_return.
    - Si status visible en FR ('gagné', 'perdu', 'en cours'), convertis-le en won/lost/pending.
    - date au format YYYY-MM-DD (utilise la date du jour si non visible).
    - Ne renvoie aucun texte hors JSON.
    """

    response = model.generate_content(
        [
            {"mime_type": mime_type, "data": image_bytes},
            prompt,
        ]
    )

    raw = (response.text or "").strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            parsed = json.loads(raw[start_idx:end_idx + 1])
        else:
            raise ValueError("Réponse IA invalide: JSON non lisible.")

    if not isinstance(parsed, dict) or "bets" not in parsed or not isinstance(parsed["bets"], list):
        raise ValueError("Réponse IA invalide: format JSON bets non conforme.")

    def _safe_float(value, default=0.0):
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        text = text.replace("€", "").replace("eur", "").replace(" ", "")
        text = text.replace("\u202f", "")
        text = re.sub(r"[^0-9,.-]", "", text)
        if text.count(",") == 1 and text.count(".") == 0:
            text = text.replace(",", ".")
        elif text.count(",") > 1 and text.count(".") == 0:
            text = text.replace(",", "")
        if text.count(".") > 1 and text.count(",") == 0:
            text = text.replace(".", "")
        try:
            return float(text)
        except Exception:
            return float(default)

    def _normalize_status(status_value):
        text = str(status_value or "pending").strip().lower()
        mapping = {
            "won": "won",
            "win": "won",
            "gagne": "won",
            "gagné": "won",
            "gagnee": "won",
            "gagnée": "won",
            "lost": "lost",
            "lose": "lost",
            "perdu": "lost",
            "perdue": "lost",
            "pending": "pending",
            "open": "pending",
            "en cours": "pending",
            "encours": "pending",
        }
        return mapping.get(text, "pending")

    def _normalize_date(date_value, default_date):
        text = str(date_value or "").strip()
        if not text:
            return default_date
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]:
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except Exception:
                continue
        return default_date

    def _pick_first(item, keys, default=None):
        for key in keys:
            if key in item and item.get(key) not in [None, ""]:
                return item.get(key)
        return default

    cleaned = []
    today_str = str(datetime.now().date())
    for item in parsed["bets"]:
        if not isinstance(item, dict):
            continue
        odds_raw = _pick_first(item, ["odds", "cote", "côte", "price"], 1.01)
        stake_raw = _pick_first(item, ["stake", "mise", "mise_totale", "montant"], None)
        potential_return_raw = _pick_first(item, ["potential_return", "gain_potentiel", "retour_potentiel"], None)

        odds_value = max(_safe_float(odds_raw, 1.01), 1.01)
        stake_value = _safe_float(stake_raw, 0.0)
        potential_return_value = _safe_float(potential_return_raw, 0.0)
        if stake_value <= 0 and potential_return_value > 0 and odds_value > 1.01:
            stake_value = potential_return_value / odds_value

        event_value = _pick_first(item, ["event", "match", "rencontre"], "?")
        selection_value = _pick_first(item, ["selection", "pick", "pronostic", "sélection"], "?")
        sport_value = _pick_first(item, ["sport", "discipline"], "?")

        cleaned.append(
            {
                "date": _normalize_date(item.get("date"), today_str),
                "sport": str(sport_value),
                "event": str(event_value),
                "selection": str(selection_value),
                "odds": round(float(odds_value), 3),
                "stake": round(float(max(stake_value, 0.0)), 2),
                "status": _normalize_status(item.get("status")),
                "close_odds": None,
            }
        )
    return cleaned


def _extract_close_updates_from_screenshot(uploaded_file, gemini_api_key):
    if not gemini_api_key:
        raise ValueError("Clé Gemini requise pour analyser une image.")

    mime_type = uploaded_file.type or "image/jpeg"
    image_bytes = uploaded_file.getvalue()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
    Tu analyses un screenshot de paris clôturés (historique/résultats, ex: Winamax).
    Retourne STRICTEMENT un JSON valide (sans markdown) au format:
    {
      "updates": [
        {
          "date": "YYYY-MM-DD",
          "event": "string",
          "selection": "string",
          "close_odds": 1.75,
          "status": "won|lost|pending"
        }
      ]
    }

    Règles:
    - Utilise close_odds = cote finale visible sur le screenshot (si visible).
    - Convertis les statuts FR (gagné/perdu/en cours) vers won/lost/pending.
    - Si un champ n'est pas visible, fournis la meilleure estimation prudente.
    - Ne renvoie aucun texte hors JSON.
    """

    response = model.generate_content(
        [
            {"mime_type": mime_type, "data": image_bytes},
            prompt,
        ]
    )

    raw = (response.text or "").strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            parsed = json.loads(raw[start_idx:end_idx + 1])
        else:
            raise ValueError("Réponse IA invalide: JSON non lisible.")

    if not isinstance(parsed, dict) or "updates" not in parsed or not isinstance(parsed["updates"], list):
        raise ValueError("Réponse IA invalide: format JSON updates non conforme.")

    def _safe_float(value, default=0.0):
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        text = text.replace("€", "").replace("eur", "").replace(" ", "").replace("\u202f", "")
        text = re.sub(r"[^0-9,.-]", "", text)
        if text.count(",") == 1 and text.count(".") == 0:
            text = text.replace(",", ".")
        try:
            return float(text)
        except Exception:
            return float(default)

    def _normalize_status(status_value):
        text = str(status_value or "pending").strip().lower()
        mapping = {
            "won": "won",
            "win": "won",
            "gagne": "won",
            "gagné": "won",
            "gagnee": "won",
            "gagnée": "won",
            "lost": "lost",
            "lose": "lost",
            "perdu": "lost",
            "perdue": "lost",
            "pending": "pending",
            "open": "pending",
            "en cours": "pending",
            "encours": "pending",
        }
        return mapping.get(text, "pending")

    cleaned = []
    for item in parsed["updates"]:
        if not isinstance(item, dict):
            continue
        close_odds = max(_safe_float(item.get("close_odds"), 0.0), 0.0)
        cleaned.append(
            {
                "date": str(item.get("date") or ""),
                "event": str(item.get("event") or "?"),
                "selection": str(item.get("selection") or "?"),
                "close_odds": round(close_odds, 3) if close_odds > 0 else None,
                "status": _normalize_status(item.get("status")),
            }
        )
    return cleaned


def _norm_text(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _find_best_bet_index_for_update(bets, update_item):
    target_event = _norm_text(update_item.get("event"))
    target_selection = _norm_text(update_item.get("selection"))

    if not target_event and not target_selection:
        return None

    best_idx = None
    best_score = -1

    for idx, bet in enumerate(bets):
        bet_event = _norm_text(bet.get("event"))
        bet_selection = _norm_text(bet.get("selection"))

        selection_match = target_selection and bet_selection and (target_selection in bet_selection or bet_selection in target_selection)
        event_match = target_event and bet_event and (target_event in bet_event or bet_event in target_event)

        score = 0
        if selection_match:
            score += 2
        if event_match:
            score += 2
        if bet.get("status") == "pending":
            score += 1

        if score > best_score and score >= 2:
            best_score = score
            best_idx = idx

    return best_idx


def _merge_unique_bets(existing_bets, new_bets):
    def _fingerprint(bet):
        return (
            str(bet.get("date", "")),
            str(bet.get("event", "")).strip().lower(),
            str(bet.get("selection", "")).strip().lower(),
            round(float(bet.get("odds", 0.0)), 3),
            round(float(bet.get("stake", 0.0)), 2),
        )

    existing_keys = {_fingerprint(b) for b in existing_bets}
    merged = []
    skipped = 0
    for bet in new_bets:
        fp = _fingerprint(bet)
        if fp in existing_keys:
            skipped += 1
            continue
        existing_keys.add(fp)
        merged.append(bet)
    return merged, skipped

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
if "scanned_matches" not in st.session_state:
    st.session_state.scanned_matches = []
if "value_candidates" not in st.session_state:
    st.session_state.value_candidates = []
if "value_top_picks" not in st.session_state:
    st.session_state.value_top_picks = []
if "ocr_preview_bets" not in st.session_state:
    st.session_state.ocr_preview_bets = []
if "ocr_close_updates_preview" not in st.session_state:
    st.session_state.ocr_close_updates_preview = []

# --- BARRE LATÉRALE ---
st.sidebar.header("⚙️ Configuration des APIs")
odds_api_key = st.sidebar.text_input("Clé The Odds API", type="password")
gemini_api_key = st.sidebar.text_input("Clé Google Gemini", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📊 Paramètres Financiers")
initial_bankroll = st.sidebar.number_input("Bankroll Initiale (€)", min_value=10, value=1000, step=50)

tracker_metrics = _compute_tracker_metrics(st.session_state.bet_tracker, float(initial_bankroll))
st.session_state.current_bankroll = float(tracker_metrics["current_bankroll"])

st.sidebar.metric(label="Bankroll Actuelle (€)", value=f"{st.session_state.current_bankroll:.2f} €")
sport_choice = st.sidebar.selectbox("Marché principal", ["Tous les Lives & À Venir", "Tennis Uniquement"])

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Modèle Value & Risque")
min_edge_pct = st.sidebar.slider("Edge minimum (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
min_books = st.sidebar.slider("Bookmakers minimum", min_value=1, max_value=8, value=2, step=1)
top_n_picks = st.sidebar.slider("Nombre de picks", min_value=1, max_value=5, value=3, step=1)
kelly_fraction_scale = st.sidebar.slider("Kelly fractionné", min_value=0.1, max_value=1.0, value=0.33, step=0.01)
max_stake_pct = st.sidebar.slider("Mise max (% bankroll)", min_value=1, max_value=15, value=5, step=1)

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
                    
                    # Mémorisation des matchs RÉELS pour le générateur de récap
                    st.session_state.scanned_matches = filtered_data
                    st.session_state.value_candidates = _compute_value_candidates(filtered_data)
                    
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

# --- MODULE QUANT : VALUE + STAKING ---
st.markdown("---")
st.header("📈 2. Moteur Value Picks (anti-biais marché)")

if not st.session_state.scanned_matches:
    st.caption("Lancez d'abord le scanner pour alimenter le moteur quantitatif.")
else:
    candidates_df = pd.DataFrame(st.session_state.value_candidates)
    if candidates_df.empty:
        st.warning("Aucun candidat exploitable détecté (cotes insuffisantes).")
    else:
        filtered_candidates = candidates_df[
            (candidates_df["edge_pct"] >= float(min_edge_pct))
            & (candidates_df["n_books"] >= int(min_books))
            & (candidates_df["best_price"] >= 1.5)
            & (candidates_df["best_price"] <= 3.5)
        ].copy()

        if filtered_candidates.empty:
            st.warning("Aucun pick ne passe vos filtres actuels. Réduisez l'edge min ou le nombre de bookmakers minimum.")
        else:
            filtered_candidates["event_key"] = filtered_candidates["sport"] + "|" + filtered_candidates["home"] + "|" + filtered_candidates["away"]
            top_candidates = (
                filtered_candidates.sort_values("edge_pct", ascending=False)
                .drop_duplicates(subset=["event_key"], keep="first")
                .head(int(top_n_picks))
                .copy()
            )

            bankroll = float(st.session_state.current_bankroll)
            max_stake_eur = bankroll * (float(max_stake_pct) / 100.0)
            stakes = []
            for _, row in top_candidates.iterrows():
                kelly_full = _kelly_fraction(float(row["best_price"]), float(row["market_prob"]))
                kelly_scaled = kelly_full * float(kelly_fraction_scale)
                stake_eur = min(bankroll * kelly_scaled, max_stake_eur)
                stakes.append(round(max(0.0, stake_eur), 2))
            top_candidates["stake_eur"] = stakes
            top_candidates["confidence_score"] = (5 + (top_candidates["edge_pct"] * 0.6)).clip(lower=5, upper=9.5).round(1)

            st.session_state.value_top_picks = top_candidates.to_dict("records")

            st.subheader("Top picks quantitatifs")
            st.dataframe(
                top_candidates[
                    [
                        "sport",
                        "home",
                        "away",
                        "selection",
                        "best_price",
                        "market_prob",
                        "fair_odds",
                        "edge_pct",
                        "n_books",
                        "stake_eur",
                    ]
                ].rename(
                    columns={
                        "sport": "Sport",
                        "home": "Home",
                        "away": "Away",
                        "selection": "Pick",
                        "best_price": "Best Odds",
                        "market_prob": "Market Prob",
                        "fair_odds": "Fair Odds",
                        "edge_pct": "Edge %",
                        "n_books": "Books",
                        "stake_eur": "Stake (€)",
                    }
                ),
                use_container_width=True,
            )

# --- TRACKER DE PERFORMANCE ---
st.markdown("---")
st.header("🧾 3. Tracker de Paris & Performance")

with st.expander("📸 Import depuis screenshot (Winamax / mobile)", expanded=False):
    uploaded_file = st.file_uploader(
        "Ajoute un screenshot de ton app de paris",
        type=["png", "jpg", "jpeg", "webp"],
        key="screenshot_uploader",
    )
    c_scan, c_add = st.columns(2)
    with c_scan:
        if st.button("Analyser le screenshot", use_container_width=True):
            if uploaded_file is None:
                st.warning("Ajoute d'abord une image.")
            elif not gemini_api_key:
                st.warning("Configure d'abord la clé Gemini en barre latérale.")
            else:
                with st.spinner("Lecture OCR/IA du screenshot..."):
                    try:
                        extracted = _extract_bets_from_screenshot(uploaded_file, gemini_api_key)
                        st.session_state.ocr_preview_bets = extracted
                        st.success(f"{len(extracted)} pari(s) détecté(s). Vérifie puis importe.")
                    except Exception as e:
                        st.error(f"Analyse image impossible: {str(e)}")
    with c_add:
        if st.button("Importer dans le tracker", use_container_width=True):
            preview = st.session_state.get("ocr_preview_bets", [])
            if not preview:
                st.warning("Aucun pari extrait à importer. Lance d'abord l'analyse.")
            else:
                to_add, skipped = _merge_unique_bets(st.session_state.bet_tracker, preview)
                st.session_state.bet_tracker.extend(to_add)
                st.session_state.ocr_preview_bets = []
                st.success(f"{len(to_add)} pari(s) importé(s) dans le tracker. Doublons ignorés: {skipped}.")
                st.rerun()

    if st.session_state.get("ocr_preview_bets"):
        st.dataframe(pd.DataFrame(st.session_state.ocr_preview_bets), use_container_width=True)

with st.expander("📷 Mise à jour résultats / close odds depuis screenshot", expanded=False):
    settled_file = st.file_uploader(
        "Ajoute un screenshot de paris clôturés (historique)",
        type=["png", "jpg", "jpeg", "webp"],
        key="screenshot_settled_uploader",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Analyser clôture", use_container_width=True):
            if settled_file is None:
                st.warning("Ajoute d'abord une image de paris clôturés.")
            elif not gemini_api_key:
                st.warning("Configure d'abord la clé Gemini en barre latérale.")
            else:
                with st.spinner("Extraction des résultats et close odds..."):
                    try:
                        updates = _extract_close_updates_from_screenshot(settled_file, gemini_api_key)
                        st.session_state.ocr_close_updates_preview = updates
                        st.success(f"{len(updates)} mise(s) à jour détectée(s).")
                    except Exception as e:
                        st.error(f"Analyse impossible: {str(e)}")

    with c2:
        if st.button("Appliquer au tracker", use_container_width=True):
            updates = st.session_state.get("ocr_close_updates_preview", [])
            if not updates:
                st.warning("Aucune mise à jour à appliquer. Lance d'abord l'analyse.")
            elif not st.session_state.bet_tracker:
                st.warning("Tracker vide. Importe d'abord des paris.")
            else:
                updated = 0
                unmatched = 0
                for item in updates:
                    idx = _find_best_bet_index_for_update(st.session_state.bet_tracker, item)
                    if idx is None:
                        unmatched += 1
                        continue

                    close_odds = item.get("close_odds")
                    if close_odds is not None and close_odds > 1.01:
                        st.session_state.bet_tracker[idx]["close_odds"] = float(close_odds)

                    new_status = item.get("status")
                    if new_status in ["won", "lost", "pending"]:
                        st.session_state.bet_tracker[idx]["status"] = new_status

                    updated += 1

                st.session_state.ocr_close_updates_preview = []
                st.success(f"{updated} pari(s) mis à jour. Non appariés: {unmatched}.")
                st.rerun()

    if st.session_state.get("ocr_close_updates_preview"):
        st.dataframe(pd.DataFrame(st.session_state.ocr_close_updates_preview), use_container_width=True)

with st.expander("➕ Ajouter un pari au tracker", expanded=False):
    with st.form("add_bet_form"):
        c1, c2 = st.columns(2)
        with c1:
            bet_sport = st.text_input("Sport", value="Tennis")
            bet_event = st.text_input("Match / Event", value="")
            bet_selection = st.text_input("Sélection", value="")
            bet_odds = st.number_input("Cote", min_value=1.01, value=1.80, step=0.01, format="%.2f")
        with c2:
            suggested_stake = 0.0
            if st.session_state.get("value_top_picks"):
                suggested_stake = float(st.session_state.value_top_picks[0].get("stake_eur", 0.0))
            bet_stake = st.number_input("Mise (€)", min_value=0.0, value=float(round(suggested_stake, 2)), step=1.0, format="%.2f")
            bet_status = st.selectbox("Statut", options=["pending", "won", "lost"], index=0)
            bet_date = st.date_input("Date", value=datetime.now().date())

        submitted = st.form_submit_button("Ajouter le pari")
        if submitted:
            if not bet_event.strip() or not bet_selection.strip():
                st.warning("Renseignez au minimum le match et la sélection.")
            else:
                st.session_state.bet_tracker.append(
                    {
                        "date": str(bet_date),
                        "sport": bet_sport.strip() if bet_sport else "?",
                        "event": bet_event.strip(),
                        "selection": bet_selection.strip(),
                        "odds": float(bet_odds),
                        "stake": float(bet_stake),
                        "status": bet_status,
                    }
                )
                st.success("Pari ajouté au tracker.")
                st.rerun()

if st.session_state.bet_tracker:
    bets_df = _build_tracker_dataframe(st.session_state.bet_tracker)
    bets_df["PnL (€)"] = bets_df["PnL (€)"].round(2)
    bets_df["CLV %"] = pd.to_numeric(bets_df["CLV %"], errors="coerce").round(2)

    st.subheader("Gestion rapide des statuts")
    selected_idx = st.selectbox(
        "Sélectionnez un pari",
        options=list(range(len(st.session_state.bet_tracker))),
        format_func=lambda i: f"#{i+1} | {st.session_state.bet_tracker[i].get('event', '?')} | {st.session_state.bet_tracker[i].get('selection', '?')} | {st.session_state.bet_tracker[i].get('status', 'pending')}",
    )
    c_upd_1, c_upd_2, c_upd_3 = st.columns([2, 1, 1])
    with c_upd_1:
        new_status = st.selectbox("Nouveau statut", ["pending", "won", "lost"], key="tracker_status_update")
    with c_upd_2:
        if st.button("Mettre à jour", use_container_width=True):
            st.session_state.bet_tracker[selected_idx]["status"] = new_status
            st.success("Statut mis à jour.")
            st.rerun()
    with c_upd_3:
        if st.button("Supprimer", use_container_width=True):
            st.session_state.bet_tracker.pop(selected_idx)
            st.success("Pari supprimé.")
            st.rerun()

    tracker_metrics = _compute_tracker_metrics(st.session_state.bet_tracker, float(initial_bankroll))
    st.session_state.current_bankroll = float(tracker_metrics["current_bankroll"])

    clv_values = bets_df["CLV %"].dropna()
    avg_clv = float(clv_values.mean()) if not clv_values.empty else 0.0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Paris", tracker_metrics["total_bets"])
    m2.metric("Hit Rate", f"{tracker_metrics['hit_rate_pct']:.1f}%")
    m3.metric("ROI", f"{tracker_metrics['roi_pct']:.1f}%")
    m4.metric("PnL", f"{tracker_metrics['pnl']:.2f} €")
    m5.metric("Bankroll", f"{tracker_metrics['current_bankroll']:.2f} €")
    m6.metric("CLV moyen", f"{avg_clv:.2f}%")

    st.subheader("Évolution bankroll")
    settled = bets_df[bets_df["status"].isin(["won", "lost"])].copy()
    if not settled.empty:
        settled = settled.sort_values("date")
        settled["bankroll"] = float(initial_bankroll) + settled["PnL (€)"].cumsum()
        chart_df = settled[["date", "bankroll"]].groupby("date", as_index=True)["bankroll"].last().to_frame()
        st.line_chart(chart_df)
    else:
        st.caption("Le graphe apparaîtra dès qu'au moins un pari sera settled (won/lost).")

    st.dataframe(
        bets_df[["date", "sport", "event", "selection", "odds", "close_odds", "stake", "status", "PnL (€)", "CLV %"]],
        use_container_width=True,
    )

    csv_data = bets_df.copy()
    csv_data["date"] = csv_data["date"].dt.strftime("%Y-%m-%d")
    st.download_button(
        "⬇️ Export CSV du tracker",
        data=csv_data.to_csv(index=False).encode("utf-8"),
        file_name=f"bet_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    if st.button("🧹 Réinitialiser le tracker", type="secondary"):
        st.session_state.bet_tracker = []
        st.session_state.current_bankroll = float(initial_bankroll)
        st.success("Tracker réinitialisé.")
        st.rerun()
else:
    st.caption("Aucun pari suivi pour l'instant. Ajoutez votre premier pari ci-dessus.")

# --- MODULE ULTIME : GÉNÉRATEUR DE RÉCAP CLEAN ---
st.markdown("---")
st.header("🎯 4. Générer le Rapport Premium (Format WhatsApp/Telegram)")
st.write("Demandez à l'IA de compiler les meilleurs choix détectés sous la forme exacte de vos captures d'écran.")

# Bouton de génération automatique basé sur les critères précis
if st.button("✨ Générer le Récap Clean pour Demain"):
    if not gemini_api_key:
        st.error("Veuillez configurer votre clé Gemini dans la barre latérale.")
    elif not st.session_state.scanned_matches:
        st.error("⚠️ Aucun match en mémoire. Lancez d'abord le Scanner de Cotes (Section 1) pour récupérer les matchs réels du jour. Sans cela, l'IA inventerait des matchs fictifs.")
    elif not st.session_state.get("value_top_picks"):
        st.error("⚠️ Aucun pick quantitatif disponible. Ajustez les filtres du module Value (edge / bookmakers) puis regénérez.")
    else:
        with st.spinner("Rédaction du rapport algorithmique par BetAgent..."):
            try:
                genai.configure(api_key=gemini_api_key)
                # Utilisation du modèle stable et mis à jour
                model = genai.GenerativeModel("gemini-2.5-flash")

                top_picks = st.session_state.get("value_top_picks", [])
                picks_summary = []
                for pick in top_picks:
                    picks_summary.append(
                        f"- [{pick['sport']}] {pick['home']} vs {pick['away']} | Pick: {pick['selection']} | Cote: {pick['best_price']:.2f} | Prob marché: {pick['market_prob']:.3f} | Fair odds: {pick['fair_odds']:.2f} | Edge: {pick['edge_pct']:.2f}% | Mise suggérée: {pick['stake_eur']:.2f}€ | Books: {pick['n_books']}"
                    )
                picks_block = "\n".join(picks_summary)

                # Le prompt de cadrage absolu : données réelles + style exact des images
                prompt_style = f"""
                Tu es 'Agent IA Pronostics'. Tu dois rédiger un résumé de pronostics sportifs destinés à être copiés directement sur un canal Telegram ou WhatsApp de parieurs professionnels.

                RÈGLES ABSOLUES ET NON NÉGOCIABLES :
                - Tu dois choisir tes picks EXCLUSIVEMENT parmi la shortlist quantitative ci-dessous.
                - INTERDICTION FORMELLE d'inventer un match, un joueur, une équipe, une cote, un edge ou une mise.
                - Utilise UNIQUEMENT les cotes/mises/edges réels fournis ci-dessous.
                - Tes connaissances sur les joueurs peuvent être obsolètes (blessures, forfaits) : ne mentionne JAMAIS un joueur absent de la liste, même s'il te semble être une star du tournoi.
                - Si la shortlist contient moins de 3 picks exploitables, propose seulement le nombre disponible.

                SHORTLIST QUANTITATIVE VALIDÉE (seule source autorisée) :
                {picks_block}

                Tu dois adopter EXACTEMENT la structure, le ton et le style visuel suivants, sans fioritures ni bavardages introductifs :

                Voici le récap clean pour demain 🎯

                MES PICKS [NOM DE LA COMPÉTITION RÉELLE des matchs choisis]

                1. [JOUEUR/ÉQUIPE EN CAPITALES issu de la shortlist] vs [Adversaire issu de la shortlist] @ [Cote réelle]
                Confiance: [Note sur 10, ex: 7/10]
                [Donne 3 lignes d'arguments percutants en t'appuyant sur les métriques de la shortlist (prob marché, fair odds, edge) + contexte sportif.]

                2. [JOUEUR/ÉQUIPE EN CAPITALES issu de la shortlist] vs [Adversaire issu de la shortlist] @ [Cote réelle]
                Confiance: [Note sur 10, ex: 6.5/10]
                [Donnes les statistiques clés et l'explication de la value face au marché].

                3. [JOUEUR/ÉQUIPE EN CAPITALES issu de la shortlist] vs [Adversaire issu de la shortlist] @ [Cote réelle]
                Confiance: [Note sur 10, ex: 5/10]
                [Explique le côté spéculatif, le profil de joueur, pourquoi c'est le 'gamble' de la journée].

                STRATÉGIE RECOMMANDÉE

                - Reprends EXACTEMENT les mises suggérées de la shortlist quantitative.
                - Ne propose que des singles.

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
