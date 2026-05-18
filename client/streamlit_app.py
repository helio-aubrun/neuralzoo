"""
Interface Streamlit — Client de l'API CIFAR-10 Animal Classifier
=================================================================
Accessibilité :
  - Alternatives textuelles pour toutes les images (aria-label / st.image alt)
  - Contrastes respectant WCAG AA (ratio ≥ 4.5:1)
  - Navigation clavier (widgets Streamlit natifs)
  - Labels explicites sur tous les champs
  - Retours visuels ET textuels pour chaque action
"""

import io
import requests
import streamlit as st
from PIL import Image

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Classificateur d'animaux CIFAR-10",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/votre-repo",
        "Report a bug": "https://github.com/votre-repo/issues",
        "About": "Classificateur d'animaux basé sur un CNN entraîné sur CIFAR-10.",
    },
)

# ── CSS accessible : contrastes WCAG AA, focus visible, responsive ────────────
st.markdown("""
<style>
/* ── Palette principale ── */
:root {
    --primary:      #1a56db;   /* bleu accessible sur blanc, ratio > 4.5:1 */
    --primary-dark: #1e40af;
    --success:      #166534;   /* vert foncé sur blanc > 7:1 */
    --success-bg:   #dcfce7;
    --warning:      #92400e;
    --warning-bg:   #fef3c7;
    --error:        #991b1b;
    --error-bg:     #fee2e2;
    --text:         #111827;
    --text-light:   #374151;
    --border:       #d1d5db;
    --bg-card:      #f9fafb;
}

/* ── Reset focus pour accessibilité clavier ── */
*:focus-visible {
    outline: 3px solid var(--primary) !important;
    outline-offset: 2px !important;
    border-radius: 4px !important;
}

/* ── Titres ── */
h1 { color: var(--text) !important; font-size: 1.9rem !important; }
h2 { color: var(--text) !important; font-size: 1.4rem !important; }
h3 { color: var(--text-light) !important; }

/* ── Carte résultat ── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 5px solid var(--primary);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

/* ── Badge confiance ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.9rem;
}
.badge-high   { background: var(--success-bg); color: var(--success); }
.badge-medium { background: var(--warning-bg); color: var(--warning); }
.badge-low    { background: var(--error-bg);   color: var(--error);   }

/* ── Barre de progression top-3 ── */
.bar-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.bar-label { width: 110px; font-size: 0.9rem; color: var(--text-light); text-align: right; }
.bar-track {
    flex: 1; background: #e5e7eb; border-radius: 6px;
    height: 18px; overflow: hidden;
}
.bar-fill {
    height: 100%; border-radius: 6px;
    background: var(--primary);
    transition: width 0.4s ease;
}
.bar-pct { font-size: 0.85rem; color: var(--text-light); min-width: 45px; }

/* ── Status API ── */
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%; margin-right: 6px;
}
.dot-ok      { background: #16a34a; }
.dot-error   { background: #dc2626; }
.dot-unknown { background: #9ca3af; }

/* ── Skip-link accessibilité ── */
.skip-link {
    position: absolute; top: -40px; left: 0;
    background: var(--primary); color: white;
    padding: 8px; font-weight: bold; border-radius: 4px;
    z-index: 9999;
}
.skip-link:focus { top: 0; }
</style>

<!-- Lien d'évitement (accessibilité clavier) -->
<a class="skip-link" href="#main-content">Aller au contenu principal</a>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
ANIMAL_EMOJIS = {
    "Oiseau":      "🐦",
    "Chat":        "🐱",
    "Cerf":        "🦌",
    "Chien":       "🐶",
    "Grenouille":  "🐸",
    "Cheval":      "🐴",
}
DEFAULT_API = "http://localhost:8000"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Configuration et informations
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_url = st.text_input(
        label="URL de l'API FastAPI",
        value=DEFAULT_API,
        help="Adresse de l'API (ex. http://localhost:8000)",
        placeholder="http://localhost:8000",
    )

    st.markdown("---")

    # ── Vérification du statut de l'API ──────────────────────────────────────
    st.markdown("### 📡 Statut de l'API")
    if st.button("🔄 Vérifier la connexion", use_container_width=True):
        try:
            resp = requests.get(f"{api_url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                dot_class = "dot-ok" if data["status"] == "ok" else "dot-error"
                st.markdown(
                    f'<span class="status-dot {dot_class}"></span>'
                    f'**{data["status"].upper()}** — modèle `{data["model_type"]}`',
                    unsafe_allow_html=True,
                )
                st.success(f"✅ API accessible — v{data['version']}")
                with st.expander("Classes supportées"):
                    for cls in data["classes"]:
                        st.write(f"{ANIMAL_EMOJIS.get(cls, '🐾')} {cls}")
            else:
                st.error(f"❌ Erreur HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de joindre l'API.")
            st.info("Vérifiez que l'API est lancée avec :\n```\nuvicorn app.main:app --reload\n```")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

    st.markdown("---")
    st.markdown("""
### ℹ️ À propos
**Modèle :** CNN entraîné sur CIFAR-10

**Classes :**
🐦 Oiseau · 🐱 Chat · 🦌 Cerf
🐶 Chien · 🐸 Grenouille · 🐴 Cheval

**Taille d'entrée :** 32 × 32 px

**Documentation API :**
🔗 [Swagger /docs]({api_url}/docs)
    """.format(api_url=api_url))

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.8rem; color:#6b7280;">Interface conforme WCAG 2.1 AA</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONTENU PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="main-content">', unsafe_allow_html=True)

st.markdown("# 🐾 Classificateur d'animaux CIFAR-10")
st.markdown(
    "Chargez une photo d'animal et le modèle CNN vous indiquera de quelle espèce il s'agit, "
    "avec un score de confiance."
)

# ── Onglets ───────────────────────────────────────────────────────────────────
tab_upload, tab_url, tab_doc = st.tabs([
    "📁 Uploader une image",
    "🔗 URL d'image",
    "📖 Documentation",
])


# ────────────────────────────────────────────────────────────────────────────
# Fonction principale : appel API + affichage des résultats
# ────────────────────────────────────────────────────────────────────────────
def call_api_and_display(image_bytes: bytes, filename: str, mime: str):
    """Envoie l'image à l'API et affiche les résultats de manière accessible."""

    col_img, col_res = st.columns([1, 1], gap="large")

    # ── Aperçu de l'image ────────────────────────────────────────────────────
    with col_img:
        st.markdown("#### 🖼️ Image envoyée")
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(
                pil_img,
                caption=f"Fichier : {filename} ({pil_img.width}×{pil_img.height} px)",
                use_container_width=True,
            )
            # Alternative textuelle explicite (accessibilité)
            st.markdown(
                f'<p class="sr-only" aria-label="Image uploadée : {filename}, '
                f'dimensions {pil_img.width} par {pil_img.height} pixels"></p>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.error("Impossible d'afficher l'image.")
            return

    # ── Appel API ─────────────────────────────────────────────────────────────
    with col_res:
        st.markdown("#### 🔮 Résultat de la prédiction")
        with st.spinner("Analyse en cours…"):
            try:
                resp = requests.post(
                    f"{api_url}/predict",
                    files={"file": (filename, image_bytes, mime)},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Impossible de joindre l'API.**\n\n"
                    "Vérifiez que le serveur FastAPI est lancé sur `" + api_url + "`."
                )
                return
            except Exception as e:
                st.error(f"❌ Erreur réseau : {e}")
                return

        # ── Traitement de la réponse ──────────────────────────────────────
        if resp.status_code == 200:
            data = resp.json()
            predicted = data["predicted_class"]
            confidence = data["confidence"]
            emoji = ANIMAL_EMOJIS.get(predicted, "🐾")

            # Niveau de confiance → couleur badge
            if confidence >= 0.70:
                badge_cls, level_text = "badge-high",   "Élevée"
            elif confidence >= 0.40:
                badge_cls, level_text = "badge-medium", "Moyenne"
            else:
                badge_cls, level_text = "badge-low",    "Faible"

            pct = confidence * 100

            # ── Carte résultat ────────────────────────────────────────────
            st.markdown(f"""
<div class="result-card" role="region" aria-label="Résultat de classification">
  <p style="font-size:2.5rem; margin:0; text-align:center;" aria-hidden="true">{emoji}</p>
  <p style="font-size:1.6rem; font-weight:700; text-align:center; color:#111827; margin:4px 0;">
    {predicted}
  </p>
  <p style="text-align:center; margin:4px 0;">
    Confiance : <span class="badge {badge_cls}" aria-label="Score de confiance : {pct:.1f}%, niveau {level_text}">
      {pct:.1f}% — {level_text}
    </span>
  </p>
  <p style="text-align:center; font-size:0.82rem; color:#6b7280; margin-top:8px;">
    ⏱ Inférence : {data['inference_time_ms']} ms
  </p>
</div>
""", unsafe_allow_html=True)

            # ── Top-3 prédictions ─────────────────────────────────────────
            st.markdown("#### 📊 Top 3 des prédictions")
            st.markdown(
                '<div role="list" aria-label="Tableau des 3 meilleures prédictions">',
                unsafe_allow_html=True,
            )
            for i, pred in enumerate(data["top_predictions"]):
                cls_name = pred["class_name"]
                cls_pct  = pred["confidence"] * 100
                bar_w    = max(2, int(cls_pct))
                rank_emoji = ["🥇", "🥈", "🥉"][i]
                st.markdown(f"""
<div class="bar-row" role="listitem"
     aria-label="{rank_emoji} {cls_name} : {cls_pct:.1f} pourcent">
  <span class="bar-label">{rank_emoji} {ANIMAL_EMOJIS.get(cls_name,'🐾')} {cls_name}</span>
  <div class="bar-track" role="progressbar"
       aria-valuenow="{cls_pct:.1f}" aria-valuemin="0" aria-valuemax="100"
       aria-label="Score {cls_name}">
    <div class="bar-fill" style="width:{bar_w}%"></div>
  </div>
  <span class="bar-pct">{cls_pct:.1f}%</span>
</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        elif resp.status_code == 400:
            err = resp.json().get("detail", "Requête invalide.")
            st.error(f"❌ **Image rejetée :** {err}")
        elif resp.status_code == 503:
            st.warning("⏳ Le modèle se charge encore. Veuillez patienter et réessayer.")
        else:
            st.error(f"❌ Erreur API {resp.status_code} : {resp.text}")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 1 — Upload de fichier
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### 📁 Uploader une image depuis votre appareil")
    st.markdown(
        "Formats acceptés : **JPEG, PNG, WebP, BMP** — Taille maximale : **10 Mo**",
    )

    uploaded = st.file_uploader(
        label="Sélectionnez une image d'animal",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Choisissez une image contenant un animal parmi : "
             "oiseau, chat, cerf, chien, grenouille, cheval.",
        accept_multiple_files=False,
    )

    if uploaded is not None:
        image_bytes = uploaded.read()
        # Correction du type MIME pour les JPEG
        mime = uploaded.type if uploaded.type else "image/jpeg"
        if st.button(
            "🔍 Lancer la classification",
            key="btn_upload",
            use_container_width=True,
            type="primary",
        ):
            call_api_and_display(image_bytes, uploaded.name, mime)
    else:
        st.info(
            "💡 **Conseil :** Pour de meilleurs résultats, utilisez une image "
            "avec un seul animal bien visible, sur fond simple."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 2 — URL d'image
# ─────────────────────────────────────────────────────────────────────────────
with tab_url:
    st.markdown("### 🔗 Classifier une image depuis une URL")

    img_url = st.text_input(
        label="URL de l'image",
        placeholder="https://example.com/mon_animal.jpg",
        help="Entrez l'URL directe d'une image JPEG, PNG ou WebP.",
    )

    if img_url:
        if st.button(
            "🔍 Lancer la classification",
            key="btn_url",
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("Téléchargement de l'image…"):
                try:
                    r = requests.get(img_url, timeout=10)
                    r.raise_for_status()
                    content_type = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
                    filename = img_url.split("/")[-1] or "image_url.jpg"
                    call_api_and_display(r.content, filename, content_type)
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ Impossible de télécharger l'image : {e}")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 3 — Documentation
# ─────────────────────────────────────────────────────────────────────────────
with tab_doc:
    st.markdown("### 📖 Documentation de l'API")

    st.markdown(f"""
L'API est documentée automatiquement via **Swagger UI** (FastAPI).

| Interface | URL |
|-----------|-----|
| Swagger UI (interactif) | [{api_url}/docs]({api_url}/docs) |
| ReDoc (lisible) | [{api_url}/redoc]({api_url}/redoc) |
| Schéma OpenAPI JSON | [{api_url}/openapi.json]({api_url}/openapi.json) |
""")

    st.markdown("---")
    st.markdown("#### 🔌 Endpoints disponibles")

    with st.expander("GET /health — Vérifier l'état de l'API", expanded=False):
        st.code("""
# Requête
GET http://localhost:8000/health

# Réponse (200 OK)
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "CNN",
  "classes": ["Oiseau", "Chat", "Cerf", "Chien", "Grenouille", "Cheval"],
  "version": "1.0.0"
}
""", language="json")

    with st.expander("POST /predict — Classifier une image", expanded=True):
        st.code("""
# Requête (multipart/form-data)
POST http://localhost:8000/predict
Content-Type: multipart/form-data

file: <fichier image>

# Réponse (200 OK)
{
  "predicted_class":  "Chat",
  "class_index":      1,
  "confidence":       0.8731,
  "top_predictions": [
    {"class_name": "Chat",    "class_index": 1, "confidence": 0.8731},
    {"class_name": "Chien",   "class_index": 3, "confidence": 0.0921},
    {"class_name": "Oiseau",  "class_index": 0, "confidence": 0.0187}
  ],
  "inference_time_ms": 12.45,
  "image_size": "640x480",
  "filename": "mon_chat.jpg"
}
""", language="json")

    st.markdown("---")
    st.markdown("#### 🐍 Exemple d'appel Python")
    st.code("""
import requests

with open("mon_chat.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("mon_chat.jpg", f, "image/jpeg")},
    )

result = response.json()
print(f"Classe prédite : {result['predicted_class']}")
print(f"Confiance      : {result['confidence']*100:.1f}%")
""", language="python")

    st.markdown("#### 🖥️ Exemple cURL")
    st.code("""
curl -X POST "http://localhost:8000/predict" \\
     -H "accept: application/json" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@mon_chat.jpg;type=image/jpeg"
""", language="bash")

st.markdown("</div>", unsafe_allow_html=True)
