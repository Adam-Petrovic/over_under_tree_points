"""GUI/frontend/app"""
from typing import Any, Optional
from tkinter import ttk, messagebox, PhotoImage
import customtkinter as ctk
from train import predict_game


class App(ctk.CTk):
    """Class for the tkinter app"""

    #  NOTES FOR TA:
    # - the tkinter classes are not listed as official typing, so we are using Optional for these cases
    # - There is a current PythonTA error for too many instance variables in a class. We believe it
    #   makes more sense to have everything in the App class for consistency and efficiency reasons
    title: Optional[Any]
    geometry: Optional[Any]
    team_abbreviations: dict[str, str]
    logos_path: str
    preload_logos: Optional[Any]
    setup_ui: Optional[Any]
    logos: dict[str, Optional[Any]]
    center_frame: Optional[Any]
    home_team_label: Optional[Any]
    away_team_label: Optional[Any]
    home_team_selector: Optional[Any]
    away_team_selector: Optional[Any]
    over_under_entry: Optional[Any]
    predict_button: Optional[Any]
    result_label: Optional[Any]
    home_logo_label: Optional[Any]
    away_logo_label: Optional[Any]

    def __init__(self) -> None:
        super().__init__()
        self.title("Over Tree Points")
        self.geometry("600x440")

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
        }

        self.logos_path = "logos/"
        self.preload_logos()

        self.setup_ui()

    def preload_logos(self) -> None:
        """Loads all team logos from 'logos/' folder"""
        self.logos = {}
        for abbr in self.team_abbreviations.values():
            self.logos[abbr] = PhotoImage(file=f"{self.logos_path}{abbr}.png")

    def setup_ui(self) -> None:
        """Loads the user interface"""

        # layout frame
        self.center_frame = ctk.CTkFrame(self)
        self.center_frame.pack(expand=True, padx=20, pady=20)

        # team selector labels
        self.home_team_label = ctk.CTkLabel(self.center_frame, text="Select Home Team:")
        self.home_team_label.grid(row=0, column=1, padx=10, sticky='w')

        self.away_team_label = ctk.CTkLabel(self.center_frame, text="Select Away Team:")
        self.away_team_label.grid(row=0, column=2, padx=10, sticky='w')

        # team selectors
        self.home_team_selector = ttk.Combobox(self.center_frame, values=list(self.team_abbreviations.keys()),
                                               state="readonly", width=20)
        self.home_team_selector.grid(row=1, column=1, padx=10)
        self.home_team_selector.bind("<<ComboboxSelected>>", self.update_home_logo)
        self.away_team_selector = ttk.Combobox(self.center_frame, values=list(self.team_abbreviations.keys()),
                                               state="readonly", width=20)
        self.away_team_selector.grid(row=1, column=2, padx=10)
        self.away_team_selector.bind("<<ComboboxSelected>>", self.update_away_logo)

        # betting line entry
        self.over_under_entry = ctk.CTkEntry(self.center_frame, placeholder_text="Betting Line")
        self.over_under_entry.grid(row=2, column=1, columnspan=2, pady=20)

        # predict button
        self.predict_button = ctk.CTkButton(self.center_frame, text="Predict", command=self.predict_outcome)
        self.predict_button.grid(row=3, column=1, columnspan=2)

        # result label
        self.result_label = ctk.CTkLabel(self.center_frame, text="", width=200, height=25)
        self.result_label.grid(row=4, column=1, columnspan=2, pady=10)

        # logos
        self.home_logo_label = ctk.CTkLabel(self, image=None, width=100, height=100, text="")
        self.home_logo_label.pack(side='left', anchor='sw', padx=20, pady=10)

        self.away_logo_label = ctk.CTkLabel(self, image=None, width=100, height=100, text="")
        self.away_logo_label.pack(side='right', anchor='se', padx=20, pady=10)

    def update_home_logo(self, _: Any) -> None:
        """Changes home team logo based off selector (left side)"""
        abbr = self.team_abbreviations.get(self.home_team_selector.get())
        if abbr in self.logos:
            self.home_logo_label.configure(image=self.logos[abbr])
        else:
            self.home_logo_label.configure(image="")

    def update_away_logo(self, _: Any) -> None:
        """Changes away team logo based off selector (right side)"""
        abbr = self.team_abbreviations.get(self.away_team_selector.get())
        if abbr in self.logos:
            self.away_logo_label.configure(image=self.logos[abbr])
        else:
            self.away_logo_label.configure(image="")

    def predict_outcome(self) -> None:
        """
        Selects the input data from the selectors/text fields and calls the
        predict_game() function to predict a game result
        """
        home_team_name = self.home_team_selector.get()
        away_team_name = self.away_team_selector.get()
        over_under = self.over_under_entry.get()

        home_team_abbr = self.team_abbreviations[home_team_name]
        away_team_abbr = self.team_abbreviations[away_team_name]

        if not over_under.isdigit():
            messagebox.showerror("Error", "Please enter a valid number for over/under points.")
            return

        outcome = predict_game(home_team_abbr, away_team_abbr, float(over_under))
        if outcome == 1:
            result_text = "Over 🎉"
        else:
            result_text = "Under ❌"
        self.result_label.configure(text=f"Prediction: {result_text}")


if __name__ == "__main__":
    # import python_ta
    # python_ta.check_all(config={
    #   'max-line-length': 120
    # })
    app = App()
    app.mainloop()
