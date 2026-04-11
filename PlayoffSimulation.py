import numpy as np
import pandas as pd
import random

class PlayoffSimulation:
    def __init__(self, sim_df):
        self.sim_df = sim_df

    def build_playoff_teams(self, year, chaos=0.075):
        # Randomly choose a base year
        year_choice = random.choice([2021, 2022, 2023])
        df = self.sim_df[self.sim_df["Year"] == year_choice].copy()
        df["Year"] = year

        # Add randomness
        df['Sim_prob'] = np.random.normal(df['Probability'], chaos)
        df['Sim_prob'] = df['Sim_prob'].clip(0.01, 0.97)

        df = df.sort_values(by='Sim_prob', ascending=False)

        # Conference champs
        conference_champs = df.drop_duplicates(subset='Conference', keep='first')
        top5 = conference_champs.head(5)

        # Fill remaining spots
        remaining = df[~df['Team'].isin(top5['Team'])]
        fillers = remaining.head(12 - len(top5))

        playoff_teams = pd.concat([top5, fillers])

        playoff_teams = playoff_teams.sort_values(by='Sim_prob', ascending=False).reset_index(drop=True)
        playoff_teams['Seed'] = playoff_teams.index + 1

        return playoff_teams

    def simulate_game(self, team_A, team_B):
        probA = team_A['Sim_prob']
        probB = team_B['Sim_prob']

        probA_win = probA / (probA + probB)
        return team_A if random.random() < probA_win else team_B

    def first_round(self, teams):
        return [
            (teams.iloc[4], teams.iloc[11]),
            (teams.iloc[5], teams.iloc[10]),
            (teams.iloc[6], teams.iloc[9]),
            (teams.iloc[7], teams.iloc[8]),
        ]

    def play_round(self, matchups):
        return [self.simulate_game(a, b) for a, b in matchups]

    def quarterfinals(self, teams, winners):
        return [
            (teams.iloc[0], winners[3]),
            (teams.iloc[1], winners[2]),
            (teams.iloc[2], winners[1]),
            (teams.iloc[3], winners[0]),
        ]

    def semifinals(self, winners):
        return [
            (winners[0], winners[3]),
            (winners[1], winners[2]),
        ]

    def championship(self, winners):
        return (winners[0], winners[1])

    def simulate_playoff(self, teams):
        r1 = self.play_round(self.first_round(teams))
        qf = self.play_round(self.quarterfinals(teams, r1))
        sf = self.play_round(self.semifinals(qf))
        final = self.championship(sf)

        champ = self.simulate_game(final[0], final[1])
        return champ['Team']