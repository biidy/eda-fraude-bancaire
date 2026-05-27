import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Configuration de la page ─────────────────────────────
st.set_page_config(
    page_title="EDA — Fraude Bancaire",
    page_icon="📊",
    layout="wide"
)

# ─── Chargement des données ────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists("data/creditcard.csv"):
        df = pd.read_csv("data/creditcard.csv")
        source = "réel"
    else:
        np.random.seed(42)
        n = 50000
        df = pd.DataFrame({
            "Time"   : np.random.uniform(0, 172792, n),
            "Amount" : np.abs(np.random.exponential(88, n)),
            "Class"  : np.random.choice([0,1], n, p=[0.998, 0.002]),
            **{f"V{i}": np.random.normal(0, 1, n) for i in range(1, 29)}
        })
        source = "démo"
    df_clean = df.drop_duplicates()
    return df_clean, source

df, source = load_data()

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("EDA Dashboard")

    # Badge source données
    if source == "réel":
        st.success("Dataset réel chargé")
    else:
        st.warning("Mode démo (CSV absent)")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Vue d'ensemble", "Univarié", "Multivarié", "Analyse fraude"]
    )

    st.markdown("---")
    st.subheader("Filtres")

    classe = st.selectbox(
        "Classe",
        ["Toutes", "Légitimes (0)", "Fraudes (1)"]
    )

    variable = st.selectbox(
        "Variable",
        ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
    )

    if classe == "Légitimes (0)":
        df_filtered = df[df["Class"] == 0]
    elif classe == "Fraudes (1)":
        df_filtered = df[df["Class"] == 1]
    else:
        df_filtered = df

    st.markdown("---")
    st.caption(f"Dataset : creditcard.csv")
    st.caption(f"{len(df):,} lignes · {df.shape[1]} colonnes")

# ─── PAGE 1 : Vue d'ensemble ───────────────────────────────
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble — Dataset fraude")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{len(df):,}")
    col2.metric("Fraudes", f"{df['Class'].sum():,}",
                delta=f"{df['Class'].mean()*100:.2f}%",
                delta_color="inverse")
    col3.metric("Valeurs manquantes", df.isnull().sum().sum())
    col4.metric("Doublons supprimés", 1081)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Répartition des classes")
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = df["Class"].value_counts()
        ax.bar(["Légitimes", "Fraudes"], counts.values,
               color=["steelblue", "red"], alpha=0.8)
        for i, v in enumerate(counts.values):
            ax.text(i, v + 500, f"{v:,}\n({v/len(df)*100:.2f}%)",
                    ha="center", fontsize=9)
        ax.set_ylabel("Nombre de transactions")
        plt.tight_layout()
        st.pyplot(fig)

    with col_b:
        st.subheader("Statistiques descriptives")
        st.dataframe(
            df[["Amount", "Time"]].describe().round(2),
            use_container_width=True
        )

# ─── PAGE 2 : Univarié ─────────────────────────────────────
elif page == "Univarié":
    st.title(f"Analyse univariée — {variable}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df_filtered, x=variable, hue="Class",
                     bins=50, kde=True, ax=ax,
                     palette={0: "steelblue", 1: "red"})
        ax.set_title(f"Distribution de {variable}")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Boxplot")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df_filtered, x="Class", y=variable,
                    palette={0: "steelblue", 1: "red"}, ax=ax)
        ax.set_xticklabels(["Légitime", "Fraude"])
        ax.set_title(f"Boxplot — {variable}")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Statistiques détaillées")
    stats = df_filtered.groupby("Class")[variable].describe().round(3)
    stats.index = ["Légitimes", "Fraudes"]
    st.dataframe(stats, use_container_width=True)

# ─── PAGE 3 : Multivarié ───────────────────────────────────
elif page == "Multivarié":
    st.title("Analyse multivariée")

    st.subheader("Matrice de corrélation")
    num_cols = df.select_dtypes(include=["float64","int64"]).columns
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.eye(len(corr), dtype=bool)
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                mask=mask, square=True,
                linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.8})
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Scatter plot interactif")
    col1, col2 = st.columns(2)
    var_x = col1.selectbox("Variable X", list(num_cols), index=0)
    var_y = col2.selectbox("Variable Y", list(num_cols), index=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    sample = df_filtered.sample(min(5000, len(df_filtered)),
                                random_state=42)
    sns.scatterplot(data=sample, x=var_x, y=var_y,
                    hue="Class", alpha=0.4,
                    palette={0: "steelblue", 1: "red"}, ax=ax)
    ax.set_title(f"{var_x} vs {var_y}")
    plt.tight_layout()
    st.pyplot(fig)

# ─── PAGE 4 : Analyse fraude ───────────────────────────────
elif page == "Analyse fraude":
    st.title("Analyse de la fraude")

    col1, col2, col3 = st.columns(3)
    col1.metric("Taux de fraude", "0.17%")
    col2.metric("Montant moyen (fraude)",
                f"{df[df['Class']==1]['Amount'].mean():.2f}€")
    col3.metric("Variable la plus prédictive", "V14")

    st.markdown("---")

    st.subheader("Variables les plus corrélées avec la fraude")
    corr_class = (df.select_dtypes(include=["float64","int64"])
                  .corr()["Class"]
                  .drop("Class")
                  .sort_values(key=abs, ascending=False)
                  .head(15))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["red" if v < 0 else "steelblue" for v in corr_class]
    corr_class.plot(kind="barh", color=colors, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Corrélation avec Class (fraude)")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Montants : fraudes vs légitimes")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x="Amount", hue="Class",
                 bins=80, kde=True, ax=ax,
                 palette={0: "steelblue", 1: "red"})
    ax.set_xlim(0, 500)
    ax.set_title("Distribution des montants (zoom 0–500€)")
    plt.tight_layout()
    st.pyplot(fig)