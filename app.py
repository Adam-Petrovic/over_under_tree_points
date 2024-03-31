"""GUI/frontend/app"""
from typing import Any, Optional
from dataclasses import dataclass
from tkinter import ttk, messagebox, PhotoImage
import customtkinter as ctk
from train import predict_game


# note for TA - we are aware these dataclasses are not necessary, it is just for bypassing pythonTA
@dataclass
class TeamData:
    """Dataclass for storing team abbreviations and logos"""
    team_abbreviations: dict[str, str]
    logos_path: str
    logos: dict[str, PhotoImage]


class SelectionUI:
    """Class for storing and initializing selection-related UI components."""

    home_team_label: Optional[ctk.CTkLabel] = None
    away_team_label: Optional[ctk.CTkLabel] = None
    home_team_selector: Optional[ttk.Combobox] = None
    away_team_selector: Optional[ttk.Combobox] = None

    def __init__(self, parent: Any, team_data: TeamData) -> None:

        self.home_team_label = ctk.CTkLabel(parent, text="Select Home Team:")
        self.home_team_label.grid(row=0, column=1, padx=10, sticky='w')

        self.away_team_label = ctk.CTkLabel(parent, text="Select Away Team:")
        self.away_team_label.grid(row=0, column=2, padx=10, sticky='w')

        self.home_team_selector = ttk.Combobox(parent, values=list(team_data.team_abbreviations.keys()),
                                               state="readonly", width=20)
        self.home_team_selector.grid(row=1, column=1, padx=10)

        self.away_team_selector = ttk.Combobox(parent, values=list(team_data.team_abbreviations.keys()),
                                               state="readonly", width=20)
        self.away_team_selector.grid(row=1, column=2, padx=10)


class PredictionUI:
    """Class for storing and initializing prediction related UI components"""

    over_under_entry: Optional[ctk.CTkEntry]
    predict_button: Optional[ctk.CTkButton]
    result_label: Optional[ctk.CTkLabel]

    def __init__(self, parent: Any) -> None:
        self.over_under_entry = ctk.CTkEntry(parent, placeholder_text="Betting Line")
        self.over_under_entry.grid(row=2, column=1, columnspan=2, pady=20)

        self.predict_button = ctk.CTkButton(parent, text="Predict")
        self.predict_button.grid(row=3, column=1, columnspan=2)

        self.result_label = ctk.CTkLabel(parent, text="", width=200, height=25)
        self.result_label.grid(row=4, column=1, columnspan=2, pady=10)


class LogoUI:
    """Class for storing and initializing logo related UI"""

    home_logo_label: Optional[ctk.CTkLabel]
    away_logo_label: Optional[ctk.CTkLabel]

    def __init__(self, parent: Any) -> None:
        self.home_logo_label = ctk.CTkLabel(parent, image=None, width=100, height=100, text="")
        self.home_logo_label.pack(side='left', anchor='sw', padx=20, pady=10)

        self.away_logo_label = ctk.CTkLabel(parent, image=None, width=100, height=100, text="")
        self.away_logo_label.pack(side='right', anchor='se', padx=20, pady=10)


class App(ctk.CTk):
    """Class for the tkinter app"""
    title: Optional[Any]
    geometry: Optional[Any]
    team_data: TeamData
    selection_ui: SelectionUI
    prediction_ui: PredictionUI
    logo_ui: LogoUI

    def __init__(self) -> None:
        super().__init__()
        self.title("Over Tree Points")
        self.geometry("600x440")

        self.team_data = TeamData(
            team_abbreviations={
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
                "New Orleans Pelicans": "NOP",
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
            },
            logos_path="logos/",
            logos={}
        )

        center_frame = ctk.CTkFrame(self)
        center_frame.pack(expand=True, padx=20, pady=20)

        self.selection_ui = SelectionUI(center_frame, self.team_data)
        self.prediction_ui = PredictionUI(center_frame)
        self.logo_ui = LogoUI(self)
        self.setup_logos()
        self.setup_actions()

    def setup_actions(self) -> None:
        """Sets up actions and event bindings for UI components"""
        self.selection_ui.home_team_selector.bind("<<ComboboxSelected>>", self.update_home_logo)
        self.selection_ui.away_team_selector.bind("<<ComboboxSelected>>", self.update_away_logo)
        self.prediction_ui.predict_button.configure(command=self.predict_outcome)

    def setup_logos(self) -> None:
        """Loads all team logos from 'logos/' folder"""
        for abbr in self.team_data.team_abbreviations.values():
            self.team_data.logos[abbr] = PhotoImage(file=f"{self.team_data.logos_path}{abbr}.png")

    def update_home_logo(self, _: Any) -> None:
        """Changes home team logo based off selector (left side)"""
        abbr = self.team_data.team_abbreviations.get(self.selection_ui.home_team_selector.get())
        if abbr in self.team_data.logos:
            self.logo_ui.home_logo_label.configure(image=self.team_data.logos[abbr])
        else:
            self.logo_ui.home_logo_label.configure(image="")

    def update_away_logo(self, _: Any) -> None:
        """Changes away team logo based off selector (right side)"""
        abbr = self.team_data.team_abbreviations.get(self.selection_ui.away_team_selector.get())
        if abbr in self.team_data.logos:
            self.logo_ui.away_logo_label.configure(image=self.team_data.logos[abbr])
        else:
            self.logo_ui.away_logo_label.configure(image="")

    def predict_outcome(self) -> None:
        """
        Selects the input data from the selectors/text fields and calls the
        predict_game() function to predict a game result
        """
        home_team_name = self.selection_ui.home_team_selector.get()
        away_team_name = self.selection_ui.away_team_selector.get()
        over_under = self.prediction_ui.over_under_entry.get()

        home_team_abbr = self.team_data.team_abbreviations[home_team_name]
        away_team_abbr = self.team_data.team_abbreviations[away_team_name]

        if home_team_name == away_team_name:
            messagebox.showerror("Error", "Please select two distinct teams.")
            return

        try:
            over_under_value = float(over_under)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for over/under points.")
            return

        outcome = predict_game(home_team_abbr, away_team_abbr, over_under_value)
        if outcome == 1:
            result_text = "Over 🎉"
        else:
            result_text = "Under ❌"
        self.prediction_ui.result_label.configure(text=f"Prediction: {result_text}")


if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config={
        'max-line-length': 120
    })
    app = App()
    app.mainloop()
