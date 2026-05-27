import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="EDA Fraude Bancaire",
    page_icon="📊",
    layout="wide"
)

# ── Chargement des données ─────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df.drop_duplicates()

df = load_data()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("EDA Dashboard")
    st.markdown("---")
    st.metric("Lignes", f"{len(df):,}")
    st.metric("Colonnes", df.shape[1])

# ── Page principale ────────────────────────────────────────
st.title("EDA — Fraude Bancaire")
st.markdown("---")

# Métriques
col1, col2, col3 = st.columns(3)
col1.metric("Total transactions", f"{len(df):,}")
col2.metric("Fraudes", f"{df['Class'].sum():,}")
col3.metric("Taux de fraude",
            f"{df['Class'].mean()*100:.2f}%")

st.markdown("---")

# Aperçu des données
st.subheader("Aperçu des données")
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# Premier graphique
st.subheader("Distribution des classes")
fig, ax = plt.subplots(figsize=(6, 3))
counts = df["Class"].value_counts()
ax.bar(["Légitimes (0)", "Fraudes (1)"],
       counts.values,
       color=["steelblue", "red"],
       alpha=0.8)
for i, v in enumerate(counts.values):
    ax.text(i, v + 500,
            f"{v:,} ({v/len(df)*100:.2f}%)",
            ha="center", fontsize=9)
ax.set_ylabel("Nombre de transactions")
plt.tight_layout()
st.pyplot(fig)