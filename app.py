import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from KNN import KNN
from LogRegress import LogRegress
from RandForest import RandForest
from PlayoffSimulation import PlayoffSimulation

st.set_page_config(page_title="CFB Simulator", layout="wide")

st.title("🏆 College Football Playoff Simulator")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("full_data.csv")

df = load_data()

# -----------------------------
# USER INPUT
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    start_year = st.number_input("Start Year", min_value=2024, value=2027)

with col2:
    end_year = st.number_input("End Year", min_value=2024, value=2027)

with col3:
    sim_size = st.selectbox("Simulations", [10, 50, 100, 500, 1000])

chaos = st.slider("Chaos Level", 0.01, 0.15, 0.075, 0.005)
st.caption("🎲 Lower = predictable • Higher = chaos & upsets")

# -----------------------------
# DEFAULT MODELS
# -----------------------------
model_options = ["Logistic Regression", "Random Forest", "KNN"]

# -----------------------------
# MODEL EVALUATION
# -----------------------------
if st.button("📊 Evaluate Models (Optional)"):

    train_df = df[df["Year"].between(2013, 2019)]
    val_df = df[df["Year"] == 2020]

    drop_cols = ["Team", "Conference", "Year", "Target", "Win %"]

    x_train = train_df.drop(columns=drop_cols)
    y_train = train_df["Target"]

    x_val = val_df.drop(columns=drop_cols)
    y_val = val_df["Target"]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    models = {
        "Logistic Regression": LogRegress(),
        "Random Forest": RandForest(),
        "KNN": KNN()
    }

    results = {}
    for name, m in models.items():
        m.train(x_train, y_train)
        results[name] = m.evaluate(x_val, y_val)

    acc_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"]).sort_values(by="Accuracy")

    st.session_state["accuracy_df"] = acc_df

# -----------------------------
# SHOW MODEL ACCURACY
# -----------------------------
if "accuracy_df" in st.session_state:

    st.subheader("📊 Model Accuracy")

    acc_df = st.session_state["accuracy_df"]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(acc_df["Model"], acc_df["Accuracy"])

    for i, v in enumerate(acc_df["Accuracy"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')

    ax.set_xlim(0,1)
    st.pyplot(fig)

    model_options = acc_df["Model"].tolist()

# -----------------------------
# MODEL SELECTION
# -----------------------------
model_choice = st.selectbox("Choose Model", model_options)

# -----------------------------
# RUN SIMULATION
# -----------------------------
if st.button("🚀 Run Simulation"):

    train_df = df[df["Year"].between(2013, 2019)]
    test_df = df[df["Year"].between(2021, 2023)].copy()

    drop_cols = ["Team", "Conference", "Year", "Target", "Win %"]

    x_train = train_df.drop(columns=drop_cols)
    y_train = train_df["Target"]

    x_test = test_df.drop(columns=drop_cols)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if model_choice == "Logistic Regression":
        model = LogRegress()
    elif model_choice == "Random Forest":
        model = RandForest()
    else:
        model = KNN()

    model.train(x_train, y_train)

    probs = model.model.predict_proba(x_test)[:,1]
    test_df["Probability"] = probs

    sim_df = test_df[test_df["Year"].isin([2021, 2022, 2023])]
    sim = PlayoffSimulation(sim_df)

    all_results = {}
    all_brackets = {}

    for year in range(start_year, end_year + 1):

        results = {}
        last_bracket = None

        for _ in range(sim_size):
            teams = sim.build_playoff_teams(year, chaos)
            champ = sim.simulate_playoff(teams)
            results[champ] = results.get(champ, 0) + 1
            last_bracket = teams

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        all_results[year] = sorted_results
        all_brackets[year] = last_bracket

    st.session_state["all_results"] = all_results
    st.session_state["all_brackets"] = all_brackets
    st.session_state["years"] = list(all_results.keys())
    st.session_state["idx"] = 0

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
# -----------------------------
# DISPLAY RESULTS
# -----------------------------
if "all_results" in st.session_state:

    idx = st.session_state["idx"]
    year = st.session_state["years"][idx]
    results = st.session_state["all_results"][year]
    bracket = st.session_state["all_brackets"][year]

    df_results = pd.DataFrame(results, columns=["Team", "Wins"])
    df_results["Win %"] = df_results["Wins"] / sim_size

    st.markdown("---")
    st.subheader(f"📅 {year} Season")

    tab1, tab2 = st.tabs(["📊 Analytics", "🏆 Bracket"])

    # -----------------------------
    # TAB 1: ANALYTICS (UNCHANGED)
    # -----------------------------
    with tab1:

        df_results = df_results.sort_values(by="Win %", ascending=True)

        col1, col2 = st.columns(2)

        # Graph 1
        fig1, ax1 = plt.subplots(figsize=(5,4))
        max_val = df_results["Win %"].max()
        colors = ["#1f77b4" if v != max_val else "#FFD700" for v in df_results["Win %"]]

        ax1.barh(df_results["Team"], df_results["Win %"], color=colors)

        for i, v in enumerate(df_results["Win %"]):
            ax1.text(v + 0.01, i, f"{v:.1%}", va='center')

        ax1.set_xlim(0,1)
        ax1.set_title("Championship Probabilities")

        # Graph 2
        fig2, ax2 = plt.subplots(figsize=(5,4))
        counts, bins, _ = ax2.hist(df_results["Win %"], bins=8)

        for i in range(len(counts)):
            ax2.text(bins[i], counts[i], str(int(counts[i])))

        ax2.set_title("Parity / Chaos")

        with col1:
            st.pyplot(fig1)
        with col2:
            st.pyplot(fig2)

        # Graph 3 + 4
        col3, col4 = st.columns(2)

        seed_df = bracket[["Seed", "Team"]].merge(df_results, on="Team")

        fig3, ax3 = plt.subplots(figsize=(5,4))
        ax3.plot(seed_df["Seed"], seed_df["Win %"], marker='o')

        for _, row in seed_df.iterrows():
            ax3.text(row["Seed"], row["Win %"], f"{row['Win %']:.2f}")

        ax3.set_title("Seed vs Probability")

        conf_counts = bracket["Conference"].value_counts()

        fig4, ax4 = plt.subplots(figsize=(5,4))
        ax4.barh(conf_counts.index, conf_counts.values)

        for i, v in enumerate(conf_counts.values):
            ax4.text(v + 0.1, i, str(v))

        ax4.set_title("Conference Representation")

        with col3:
            st.pyplot(fig3)
        with col4:
            st.pyplot(fig4)

        st.dataframe(df_results)

    # -----------------------------
    # TAB 2: BRACKET (UPGRADED)
    # -----------------------------
    with tab2:

        st.subheader("🏆 Playoff Bracket")

        import time

        # Extract teams by seed
        seeds = bracket.sort_values("Seed").reset_index(drop=True)

        # First round matchups
        matchups = [
            (4,11), (5,10), (6,9), (7,8)
        ]

        # Store winners
        round1_winners = []

        st.write("### 🎯 First Round")

        for a, b in matchups:
            teamA = seeds.iloc[a]
            teamB = seeds.iloc[b]

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"{teamA['Seed']} {teamA['Team']}")
            with col2:
                st.write(f"{teamB['Seed']} {teamB['Team']}")

            # Simulate winner (visual only)
            winner = teamA if teamA["Sim_prob"] > teamB["Sim_prob"] else teamB
            round1_winners.append(winner)

            st.success(f"➡ {winner['Team']} advances")

            time.sleep(0.2)

        # Quarterfinals
        st.write("### 🏆 Quarterfinals")

        top4 = seeds.iloc[:4]

        qf_winners = []

        for i in range(4):
            teamA = top4.iloc[i]
            teamB = round1_winners[3 - i]

            st.write(f"{teamA['Team']} vs {teamB['Team']}")

            winner = teamA if teamA["Sim_prob"] > teamB["Sim_prob"] else teamB
            qf_winners.append(winner)

            st.success(f"➡ {winner['Team']} advances")

            time.sleep(0.2)

        # Semifinals
        st.write("### 🔥 Semifinals")

        sf_winners = []

        for i in range(2):
            teamA = qf_winners[i]
            teamB = qf_winners[3 - i]

            st.write(f"{teamA['Team']} vs {teamB['Team']}")

            winner = teamA if teamA["Sim_prob"] > teamB["Sim_prob"] else teamB
            sf_winners.append(winner)

            st.success(f"➡ {winner['Team']} advances")

            time.sleep(0.2)

        # Championship
        st.write("### 🏆 Championship")

        teamA = sf_winners[0]
        teamB = sf_winners[1]

        st.write(f"{teamA['Team']} vs {teamB['Team']}")

        champ = teamA if teamA["Sim_prob"] > teamB["Sim_prob"] else teamB

        st.success(f"🏆 Champion: {champ['Team']}")

    # -----------------------------
    # NAVIGATION
    # -----------------------------
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        if st.button("⬅ Previous"):
            if idx > 0:
                st.session_state["idx"] -= 1

    with col3:
        if st.button("Next ➡"):
            if idx < len(st.session_state["years"]) - 1:
                st.session_state["idx"] += 1