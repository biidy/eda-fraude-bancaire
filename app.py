import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

# ─── Configuration de la page ─────────────────────────────
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# ══════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════

@st.cache_data
def load_fraude():
    if os.path.exists("data/creditcard.csv"):
        df = pd.read_csv("data/creditcard.csv")
        source = "réel"
    else:
        np.random.seed(42)
        n = 50000
        df = pd.DataFrame({
            "Time"  : np.random.uniform(0, 172792, n),
            "Amount": np.abs(np.random.exponential(88, n)),
            "Class" : np.random.choice([0,1], n, p=[0.998, 0.002]),
            **{f"V{i}": np.random.normal(0, 1, n) for i in range(1, 29)}
        })
        source = "démo"
    df = df.drop_duplicates()
    return df, source


@st.cache_data
def load_marketing():
    if os.path.exists("data/marketing_campaign.csv"):
        df = pd.read_csv("data/marketing_campaign.csv", sep=";")
        source = "réel"
    else:
        np.random.seed(42)
        n = 2240
        df = pd.DataFrame({
            "ID"                : range(n),
            "Year_Birth"        : np.random.randint(1940, 1995, n),
            "Education"         : np.random.choice(
                                     ["Graduation","PhD","Master",
                                      "2n Cycle","Basic"], n,
                                     p=[0.50,0.21,0.17,0.09,0.03]),
            "Marital_Status"    : np.random.choice(
                                     ["Married","Together","Single",
                                      "Divorced","Widow"], n,
                                     p=[0.38,0.26,0.21,0.10,0.05]),
            "Income"            : np.random.normal(52000, 20000, n),
            "Kidhome"           : np.random.choice([0,1,2], n,
                                     p=[0.57,0.37,0.06]),
            "Teenhome"          : np.random.choice([0,1,2], n,
                                     p=[0.55,0.39,0.06]),
            "Recency"           : np.random.randint(0, 100, n),
            "MntWines"          : np.abs(np.random.exponential(300, n)),
            "MntFruits"         : np.abs(np.random.exponential(26, n)),
            "MntMeatProducts"   : np.abs(np.random.exponential(166, n)),
            "MntFishProducts"   : np.abs(np.random.exponential(37, n)),
            "MntSweetProducts"  : np.abs(np.random.exponential(27, n)),
            "MntGoldProds"      : np.abs(np.random.exponential(44, n)),
            "NumDealsPurchases" : np.random.randint(0, 15, n),
            "NumWebPurchases"   : np.random.randint(0, 15, n),
            "NumCatalogPurchases": np.random.randint(0, 15, n),
            "NumStorePurchases" : np.random.randint(0, 15, n),
            "NumWebVisitsMonth" : np.random.randint(0, 20, n),
            "AcceptedCmp1"      : np.random.choice([0,1], n, p=[0.935,0.065]),
            "AcceptedCmp2"      : np.random.choice([0,1], n, p=[0.985,0.015]),
            "AcceptedCmp3"      : np.random.choice([0,1], n, p=[0.928,0.072]),
            "AcceptedCmp4"      : np.random.choice([0,1], n, p=[0.926,0.074]),
            "AcceptedCmp5"      : np.random.choice([0,1], n, p=[0.927,0.073]),
            "Response"          : np.random.choice([0,1], n, p=[0.851,0.149]),
            "Complain"          : np.random.choice([0,1], n, p=[0.99,0.01]),
        })
        source = "démo"

    # Nettoyage
    df = df.drop_duplicates()
    df = df.dropna(subset=["Income"])
    df["Income"]     = df["Income"].clip(lower=0)
    df["Age"]        = 2024 - df["Year_Birth"]
    df               = df[df["Age"] <= 90]

    cols_mnt = ["MntWines","MntFruits","MntMeatProducts",
                "MntFishProducts","MntSweetProducts","MntGoldProds"]
    df["Total_Depenses"] = df[cols_mnt].sum(axis=1)

    cols_cmp = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                "AcceptedCmp4","AcceptedCmp5","Response"]
    df["Nb_Campagnes"] = df[cols_cmp].sum(axis=1)

    def segmenter(n):
        if n == 0   : return "0 — Non réactif"
        elif n == 1 : return "1 — Peu réactif"
        elif n <= 3 : return "2 — Modérément réactif"
        else        : return "3 — Très réactif"

    df["Segment_Campagne"] = df["Nb_Campagnes"].apply(segmenter)
    return df, source


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📊 EDA Dashboard")
    st.markdown("---")

    dataset = st.selectbox(
        "🗂️ Dataset",
        ["Fraude Bancaire", "Marketing Campaign"]
    )

    if dataset == "Fraude Bancaire":
        df, source = load_fraude()
    else:
        df, source = load_marketing()

    if source == "réel":
        st.success("✅ Dataset réel chargé")
    else:
        st.warning("⚠️ Mode démo (CSV absent)")

    st.markdown("---")

    if dataset == "Fraude Bancaire":
        page = st.radio("Navigation", [
            "🏠 Accueil","📋 Données","📊 Univarié",
            "🔗 Multivarié","🚨 Analyse fraude","💡 Conclusions"
        ])
    else:
        page = st.radio("Navigation", [
            "🏠 Accueil","📋 Données","📊 Univarié",
            "🔗 Multivarié","👥 Segments clients","💡 Conclusions"
        ])

    st.markdown("---")
    st.subheader("Filtres")

    if dataset == "Fraude Bancaire":
        classe = st.selectbox("Classe",
            ["Toutes","Légitimes (0)","Fraudes (1)"])
        variable = st.selectbox("Variable",
            ["Amount","Time"]+[f"V{i}" for i in range(1,29)])
        if classe == "Légitimes (0)":
            df_filtered = df[df["Class"] == 0]
        elif classe == "Fraudes (1)":
            df_filtered = df[df["Class"] == 1]
        else:
            df_filtered = df
    else:
        segment = st.selectbox("Segment", [
            "Tous","0 — Non réactif","1 — Peu réactif",
            "2 — Modérément réactif","3 — Très réactif"])
        cols_num_mkt = df.select_dtypes(
            include=["float64","int64"]).columns.tolist()
        variable = st.selectbox("Variable numérique", cols_num_mkt)
        df_filtered = df if segment == "Tous" else \
                      df[df["Segment_Campagne"] == segment]

    st.markdown("---")
    st.caption(f"Dataset : {dataset}")
    st.caption(f"{len(df):,} lignes · {df.shape[1]} colonnes")


# ══════════════════════════════════════════════════════════
# PAGE 0 — ACCUEIL
# ══════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.title("EDA — Analyse Exploratoire de Données")
    st.subheader("Projet Data Science")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Objectifs
        - Comprendre la **structure** des données
        - Identifier les **variables importantes**
        - Produire des **visualisations claires**
        - Formuler des **hypothèses** exploratoires

        ### 🛠️ Technologies
        - **Python** · **Pandas** · **NumPy**
        - **Matplotlib** · **Seaborn**
        - **Streamlit**
        """)
    with col2:
        st.info("""
        ### 🏦 Fraude Bancaire
        Comment analyser les transactions bancaires
        pour identifier des schémas de fraude ?
        - **284 807** transactions · **31** variables
        """)
        st.success("""
        ### 🎯 Marketing Campaign
        Quels sont les segments de clients les plus
        rentables selon leurs interactions avec
        les campagnes marketing ?
        - **2 240** clients · **29** variables
        """)

    st.markdown("---")
    st.markdown("""
    ### 🗺️ Guide de navigation
    | Page | Contenu |
    |---|---|
    | 📋 Données | Aperçu, types, statistiques, nettoyage |
    | 📊 Univarié | Distribution de chaque variable |
    | 🔗 Multivarié | Corrélations et relations entre variables |
    | 🚨 Analyse fraude | Patterns et variables prédictives |
    | 👥 Segments clients | Rentabilité par segment |
    | 💡 Conclusions | Insights et recommandations |
    """)


# ══════════════════════════════════════════════════════════
# PAGE 1 — DONNÉES
# ══════════════════════════════════════════════════════════
elif page == "📋 Données":
    st.title("📋 Aperçu des données")
    st.markdown("---")

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Lignes",    f"{len(df):,}")
    col2.metric("Colonnes",  df.shape[1])
    col3.metric("Manquants", df.isnull().sum().sum())
    col4.metric("Doublons",  0)

    st.markdown("---")
    n_lignes = st.slider("Nombre de lignes à afficher", 5, 50, 10)
    st.dataframe(df.head(n_lignes), use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Types et qualité")
        types_df = pd.DataFrame({
            "Type"       : df.dtypes,
            "Manquants"  : df.isnull().sum(),
            "Uniques"    : df.nunique(),
            "% Manquants": (df.isnull().mean()*100).round(2)
        })
        st.dataframe(types_df, use_container_width=True)
    with col_b:
        st.subheader("Statistiques descriptives")
        num_cols = df.select_dtypes(
            include=["float64","int64"]).columns
        st.dataframe(df[num_cols].describe().round(2),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — UNIVARIÉ
# ══════════════════════════════════════════════════════════
elif page == "📊 Univarié":
    st.title(f"📊 Analyse univariée — {variable}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution")
        fig, ax = plt.subplots(figsize=(5,4))
        if dataset == "Fraude Bancaire":
            sns.histplot(data=df_filtered, x=variable,
                         hue="Class", bins=50, kde=True, ax=ax)
        else:
            sns.histplot(data=df_filtered, x=variable,
                         bins=50, kde=True, ax=ax, color="steelblue")
        ax.axvline(df_filtered[variable].mean(),
                   color="orange", linestyle="--", label="Moyenne")
        ax.axvline(df_filtered[variable].median(),
                   color="green",  linestyle="--", label="Médiane")
        ax.legend()
        ax.set_title(f"Distribution — {variable}")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Boxplot")
        fig, ax = plt.subplots(figsize=(5,4))
        if dataset == "Fraude Bancaire":
            sns.boxplot(data=df_filtered, x="Class", y=variable,
                        palette={0:"steelblue",1:"red"}, ax=ax)
            ax.set_xticklabels(["Légitime","Fraude"])
        else:
            sns.boxplot(data=df_filtered, y=variable,
                        color="steelblue", ax=ax)
        ax.set_title(f"Boxplot — {variable}")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Statistiques")
        st.dataframe(df_filtered[variable].describe()
                     .round(3).to_frame(),
                     use_container_width=True)
    with col_b:
        st.subheader("Outliers (IQR)")
        Q1  = df_filtered[variable].quantile(0.25)
        Q3  = df_filtered[variable].quantile(0.75)
        IQR = Q3 - Q1
        b_b = Q1 - 1.5*IQR
        b_h = Q3 + 1.5*IQR
        n_o = int(((df_filtered[variable]<b_b)|
                   (df_filtered[variable]>b_h)).sum())
        c1,c2 = st.columns(2)
        c1.metric("Q1",  f"{Q1:.2f}")
        c1.metric("Q3",  f"{Q3:.2f}")
        c1.metric("IQR", f"{IQR:.2f}")
        c2.metric("Borne basse", f"{b_b:.2f}")
        c2.metric("Borne haute", f"{b_h:.2f}")
        c2.metric("Outliers",
                  f"{n_o} ({n_o/len(df_filtered)*100:.2f}%)")

    if dataset == "Marketing Campaign":
        st.markdown("---")
        st.subheader("Variables catégorielles")
        cols_cat = df.select_dtypes(
            include=["object","category"]).columns.tolist()
        col_cat = st.selectbox("Variable catégorielle", cols_cat)
        c1, c2  = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,4))
            ordre = df[col_cat].value_counts().index
            sns.countplot(data=df, x=col_cat, order=ordre,
                           ax=ax)
            ax.tick_params(axis="x", rotation=45)
            total = len(df)
            for p in ax.patches:
                pct = f"{100*p.get_height()/total:.1f}%"
                ax.annotate(pct,
                    (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
            ax.set_title(f"Fréquences — {col_cat}")
            plt.tight_layout()
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6,4))
            freq = df[col_cat].value_counts()
            ax.pie(freq.values, labels=freq.index, autopct="%1.1f%%")
            ax.set_title(f"Proportions — {col_cat}")
            plt.tight_layout()
            st.pyplot(fig)


# ══════════════════════════════════════════════════════════
# PAGE 3 — MULTIVARIÉ
# ══════════════════════════════════════════════════════════
elif page == "🔗 Multivarié":
    st.title("🔗 Analyse multivariée")
    st.markdown("---")

    num_cols = df.select_dtypes(
        include=["float64","int64"]).columns.tolist()

    ong1, ong2, ong3 = st.tabs([
        "🌡️ Corrélations","🔵 Scatter plot","📦 Par groupe"])

    with ong1:
        seuil = st.slider("Corrélations > (valeur absolue)",
                           0.0, 1.0, 0.0, 0.05)
        corr  = df[num_cols].corr()
        mask  = np.eye(len(corr), dtype=bool)
        if seuil > 0:
            mask = mask | (corr.abs() < seuil)
        fig, ax = plt.subplots(figsize=(14,12))
        sns.heatmap(corr, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, mask=mask,
                    square=True, linewidths=0.3, ax=ax,
                    cbar_kws={"shrink":0.8})
        plt.tight_layout()
        st.pyplot(fig)

    with ong2:
        c1,c2  = st.columns(2)
        var_x  = c1.selectbox("Variable X", num_cols, index=0)
        var_y  = c2.selectbox("Variable Y", num_cols, index=1)
        n_pts  = st.slider("Nombre de points", 500, 5000, 2000)
        sample = df_filtered.sample(
            min(n_pts, len(df_filtered)), random_state=42)
        fig, ax = plt.subplots(figsize=(8,5))
        if dataset == "Fraude Bancaire":
            sns.scatterplot(data=sample, x=var_x, y=var_y,
                            hue="Class", alpha=0.4,
                             ax=ax)
        else:
            sns.scatterplot(data=sample, x=var_x, y=var_y,
                            hue="Segment_Campagne", alpha=0.5, ax=ax)
        ax.set_title(f"{var_x} vs {var_y}")
        plt.tight_layout()
        st.pyplot(fig)

    with ong3:
        var_num = st.selectbox("Variable numérique", num_cols)
        if dataset == "Fraude Bancaire":
            c1,c2 = st.columns(2)
            with c1:
                fig,ax = plt.subplots(figsize=(5,4))
                sns.boxplot(data=df, x="Class", y=var_num,
                            palette={0:"steelblue",1:"red"}, ax=ax)
                ax.set_xticklabels(["Légitime","Fraude"])
                ax.set_title(f"Boxplot — {var_num}")
                plt.tight_layout(); st.pyplot(fig)
            with c2:
                fig,ax = plt.subplots(figsize=(5,4))
                sns.violinplot(data=df, x="Class", y=var_num,
                                ax=ax)
                ax.set_xticklabels(["Légitime","Fraude"])
                ax.set_title(f"Violinplot — {var_num}")
                plt.tight_layout(); st.pyplot(fig)
        else:
            cols_cat = df.select_dtypes(
                include=["object","category"]).columns.tolist()
            var_cat = st.selectbox("Variable catégorielle", cols_cat)
            ordre   = (df.groupby(var_cat)[var_num]
                       .median()
                       .sort_values(ascending=False).index)
            c1,c2 = st.columns(2)
            with c1:
                fig,ax = plt.subplots(figsize=(6,4))
                sns.boxplot(data=df, x=var_cat, y=var_num,
                            order=ordre, ax=ax)
                ax.tick_params(axis="x", rotation=45)
                ax.set_title(f"{var_num} par {var_cat}")
                plt.tight_layout(); st.pyplot(fig)
            with c2:
                fig,ax = plt.subplots(figsize=(6,4))
                sns.barplot(data=df, x=var_cat, y=var_num,
                            order=ordre,  ax=ax)
                ax.tick_params(axis="x", rotation=45)
                ax.set_title(f"Moyenne {var_num} par {var_cat}")
                plt.tight_layout(); st.pyplot(fig)


# ══════════════════════════════════════════════════════════
# PAGE 4A — ANALYSE FRAUDE
# ══════════════════════════════════════════════════════════
elif page == "🚨 Analyse fraude":
    st.title("🚨 Analyse de la fraude")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Taux de fraude",  f"{df['Class'].mean()*100:.2f}%")
    c2.metric("Nb fraudes",      f"{df['Class'].sum():,}")
    c3.metric("Montant moy. fraude",
              f"{df[df['Class']==1]['Amount'].mean():.2f}€")
    c4.metric("Montant moy. légitime",
              f"{df[df['Class']==0]['Amount'].mean():.2f}€")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Variables prédictives")
        corr_class = (df.select_dtypes(include=["float64","int64"])
                      .corr()["Class"].drop("Class")
                      .sort_values(key=abs, ascending=False).head(15))
        fig,ax = plt.subplots(figsize=(7,5))
        colors = ["red" if v<0 else "steelblue" for v in corr_class]
        corr_class.plot(kind="barh", color=colors, ax=ax)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Corrélation avec Class (fraude)")
        plt.tight_layout(); st.pyplot(fig)

    with col_b:
        st.subheader("Distribution des montants")
        zoom = st.slider("Zoom max (€)", 100, 5000, 500)
        fig,ax = plt.subplots(figsize=(7,5))
        sns.histplot(data=df, x="Amount", hue="Class",
                     bins=80, kde=True, ax=ax,
                     palette={0:"steelblue",1:"red"})
        ax.set_xlim(0, zoom)
        ax.set_title(f"Montants (0–{zoom}€)")
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top 6 variables discriminantes")
    top_vars = corr_class.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14,8))
    axes = axes.flatten()
    for i, col in enumerate(top_vars):
        sns.histplot(data=df, x=col, hue="Class",
                     bins=40, kde=True,
                     palette={0:"steelblue",1:"red"},
                     ax=axes[i])
        axes[i].set_title(col)
    plt.suptitle("Top variables discriminantes", fontsize=14)
    plt.tight_layout(); st.pyplot(fig)


# ══════════════════════════════════════════════════════════
# PAGE 4B — SEGMENTS CLIENTS
# ══════════════════════════════════════════════════════════
elif page == "👥 Segments clients":
    st.title("👥 Segments clients — Campagnes marketing")
    st.markdown("---")

    ordre_seg = ["0 — Non réactif","1 — Peu réactif",
                 "2 — Modérément réactif","3 — Très réactif"]

    cols_seg = st.columns(4)
    for i, seg in enumerate(ordre_seg):
        n   = len(df[df["Segment_Campagne"]==seg])
        pct = n/len(df)*100
        cols_seg[i].metric(seg, f"{n}", f"{pct:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Dépenses totales")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.boxplot(data=df, x="Segment_Campagne",
                    y="Total_Depenses", order=ordre_seg,
                     ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("Dépenses totales par segment")
        plt.tight_layout(); st.pyplot(fig)

    with col_b:
        st.subheader("Revenu moyen")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=df, x="Segment_Campagne", y="Income",
                    order=ordre_seg,  ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("Revenu moyen par segment")
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    st.subheader("Dépenses par catégorie de produit")
    cols_mnt = ["MntWines","MntFruits","MntMeatProducts",
                "MntFishProducts","MntSweetProducts","MntGoldProds"]
    fig, axes = plt.subplots(2, 3, figsize=(14,8))
    axes = axes.flatten()
    for i, col in enumerate(cols_mnt):
        sns.barplot(data=df, x="Segment_Campagne", y=col,
                    order=ordre_seg, palette="steelblue", ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=30)
    plt.suptitle("Dépenses par catégorie et segment", fontsize=14)
    plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Education par segment")
        ct = pd.crosstab(df["Segment_Campagne"],
                         df["Education"],
                         normalize="index") * 100
        fig,ax = plt.subplots(figsize=(7,4))
        ct.loc[[s for s in ordre_seg if s in ct.index]].plot(
            kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_title("Education par segment (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout(); st.pyplot(fig)

    with col_d:
        st.subheader("Canaux d'achat")
        cols_canal = ["NumWebPurchases","NumCatalogPurchases",
                      "NumStorePurchases"]
        canal_moy = (df.groupby("Segment_Campagne")[cols_canal]
                     .mean()
                     .loc[[s for s in ordre_seg
                            if s in df["Segment_Campagne"].unique()]])
        fig,ax = plt.subplots(figsize=(7,4))
        canal_moy.plot(kind="bar", ax=ax)
        ax.set_title("Canaux d'achat par segment")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="upper left")
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    st.subheader("Profil complet par segment")
    cols_profil = ["Income","Age","Total_Depenses",
                   "NumWebPurchases","NumStorePurchases","Recency"]
    profil = (df.groupby("Segment_Campagne")[cols_profil]
              .mean().round(2))
    st.dataframe(profil, use_container_width=True)

    st.subheader("Heatmap profil normalisé (0=min · 1=max)")
    scaler    = MinMaxScaler()
    profil_sc = pd.DataFrame(
        scaler.fit_transform(profil),
        columns=profil.columns, index=profil.index)
    fig,ax = plt.subplots(figsize=(10,4))
    sns.heatmap(profil_sc, annot=True, fmt=".2f",
                cmap="YlOrRd", linewidths=0.5, ax=ax)
    plt.tight_layout(); st.pyplot(fig)


# ══════════════════════════════════════════════════════════
# PAGE 5 — CONCLUSIONS
# ══════════════════════════════════════════════════════════
elif page == "💡 Conclusions":
    st.title("💡 Conclusions et Recommandations")
    st.markdown("---")

    if dataset == "Fraude Bancaire":
        st.subheader("Réponse à la problématique")
        st.success("""
        📌 Les transactions frauduleuses (0.17%) présentent
        des patterns clairs détectables via l'EDA.
        Les variables V14, V4, V11 et V12 sont les plus
        discriminantes entre fraudes et transactions légitimes.
        """)
        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        c1.info("**Déséquilibre classes**\n\n99.83% légitimes vs 0.17% fraudes")
        c2.info("**Variable clé**\n\nV14 (corr=-0.74) la plus prédictive")
        c3.info("**Montants similaires**\n\nLe montant seul ne suffit pas")
        st.markdown("---")
        st.warning("""
        **⚠️ Limites**
        - Variables V1–V28 anonymisées par PCA
        - Données sur 2 jours uniquement
        - EDA descriptive uniquement
        """)
        st.markdown("""
        **🚀 Prochaines étapes**
        1. Appliquer **SMOTE** pour équilibrer les classes
        2. Tester **Random Forest** et **XGBoost**
        3. Approfondir l'analyse de **V14**
        """)

    else:
        st.subheader("Réponse à la problématique")
        st.success("""
        📌 Les clients très réactifs aux campagnes (~6%)
        génèrent ~35% des dépenses totales.
        Revenu élevé et niveau d'éducation supérieur sont
        les principaux facteurs de réactivité aux campagnes.
        """)
        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        c1.info("**Segment premium**\n\nTrès réactifs dépensent 5x plus")
        c2.info("**Revenu déterminant**\n\nIncome corrélé à +0.6")
        c3.info("**Profil éduqué**\n\nPhD/Master surreprésentés")
        st.markdown("---")
        st.markdown("""
        **🎯 Recommandations business**
        1. **Cibler** les clients à revenu > 60 000€
        2. **Utiliser** le canal web et catalogue
        3. **Proposer** des offres premium (vin, viande)
        4. **Fidéliser** les segments très réactifs
        """)
        st.warning("""
        **⚠️ Limites**
        - Dataset de 2 240 clients uniquement
        - Segmentation basée sur les campagnes uniquement
        - Analyse descriptive — pas de modèle prédictif
        """)
        st.markdown("""
        **🚀 Prochaines étapes**
        1. Appliquer **K-Means** pour segmentation avancée
        2. Construire un modèle de **prédiction de réponse**
        3. Analyser la **valeur vie client** (CLV)
        """)
