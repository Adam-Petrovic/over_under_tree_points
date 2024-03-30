import customtkinter as ctk
from tkinter import messagebox
from train import predict_game

class NBA_Betting_Predictor_App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("NBA Betting Predictor")
        self.geometry("400x300")

        # Mapping of team names to abbreviations
        self.team_abbreviations = {
            "Atlanta Hawks": "ATL",
            "Boston Celtics": "BOS",
            "Brooklyn Nets": "BKN",
            "Charlotte Hornets": "CHA",
            "Chicago Bulls": "CHI",
            "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL",
            "Denver Nuggets": "DEN",
            "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW",
            "Houston Rockets": "HOU",
            "Indiana Pacers": "IND",
            "Los Angeles Clippers": "LAC",
            "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL",
            "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOL",
            "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL",
            "Philadelphia 76ers": "PHI",
            "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR",
            "Sacramento Kings": "SAC",
            "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR",
            "Utah Jazz": "UTA",
            "Washington Wizards": "WAS"
        }

        # Home team entry
        self.home_team_entry = ctk.CTkEntry(self, placeholder_text="Enter Home Team")
        self.home_team_entry.pack(pady=10)

        # Away team entry
        self.away_team_entry = ctk.CTkEntry(self, placeholder_text="Enter Away Team")
        self.away_team_entry.pack(pady=10)

        # Over/Under entry
        self.over_under_entry = ctk.CTkEntry(self, placeholder_text="Over/Under Points")
        self.over_under_entry.pack(pady=10)

        # Prediction button
        self.predict_button = ctk.CTkButton(self, text="Predict", command=self.predict_outcome)
        self.predict_button.pack(pady=20)

        # Result display
        self.result_label = ctk.CTkLabel(self, text="")
        self.result_label.pack(pady=10)

    def predict_outcome(self):
        # Get user inputs
        home_team_name = self.home_team_entry.get()
        away_team_name = self.away_team_entry.get()
        over_under = self.over_under_entry.get()

        # Convert team names to abbreviations
        home_team = self.team_abbreviations.get(home_team_name)
        away_team = self.team_abbreviations.get(away_team_name)

        # Validate inputs
        if not home_team or not away_team:
            messagebox.showerror("Error", "Invalid team name. Please try again.")
            return

        if not over_under.isdigit():
            messagebox.showerror("Error", "Please enter a valid number for over/under points.")
            return

        # Predict and display outcome
        outcome = predict_game(home_team, away_team, float(over_under))
        result_text = "Over" if outcome else "Under"
        self.result_label.configure(text=f"The prediction is: {result_text}")
        print('successfully predicted')


if __name__ == "__main__":
    app = NBA_Betting_Predictor_App()
    app.mainloop()
