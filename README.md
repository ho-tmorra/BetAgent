# BetAgent

## Dépannage Streamlit Cloud

Si vous voyez l'erreur:

`TypeError: Failed to fetch dynamically imported module: .../static/js/DataFrame....js`

cela vient généralement d'un décalage entre le build frontend Streamlit et le cache navigateur (ou d'un redeploy avec versions non figées).

### Correctifs appliqués dans ce repo

- Versions Python figées dans [requirements.txt](requirements.txt) pour éviter les builds non déterministes.
- Dépendances directes explicites (`pandas`, `pyarrow`) pour la couche DataFrame.

### Actions à faire après push

1. Redéployer l'app Streamlit Cloud (reboot/redeploy complet).
2. Ouvrir l'URL en navigation privée ou faire un hard refresh (`Ctrl+F5`).
3. Si besoin, vider le cache du site `betagent.streamlit.app` puis recharger.
