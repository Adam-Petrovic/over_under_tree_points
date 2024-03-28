"""This file is the dataprepping and training of the model"""

import numpy as np
from sklearn.utils import Bunch

from random_forest import RandomForest


class Team_Performance:
    """
    A class which stores the box score for the Team in the given _Game
    """

    def __init__(self, team: str, fgm, fga, fg_percentage,
                 three_pm, three_pa, three_p_percentage, ftm, fta, ft_percentage, oreb, dreb, reb,
                 assists, steals, block, turnovers, personal_fouls, plus_minus):
        self.team = team
        self.fgm = fgm
        self.fga = fga
        self.fg_percentage = fg_percentage
        self.three_pm = three_pm
        self.three_pa = three_pa
        self.three_p_percentage = three_p_percentage
        self.ftm = ftm
        self.fta = fta
        self.ft_percentage = ft_percentage
        self.oreb = oreb
        self.dreb = dreb
        self.reb = reb
        self.assists = assists
        self.steals = steals
        self.blocks = block
        self.turnovers = turnovers
        self.personal_fouls = personal_fouls
        self.plus_minus = plus_minus

    def to_list(self):
        """
        Represents the Team_Performance as a list
        """
        return [self.fgm, self.fga, self.fg_percentage,
                self.three_pm, self.three_pa, self.three_p_percentage, self.ftm, self.fta, self.ft_percentage,
                self.oreb, self.dreb, self.reb,
                self.assists, self.steals, self.blocks, self.turnovers, self.personal_fouls, self.plus_minus]


class _Game:
    """
    Represents an instance of a game.
    """

    def __init__(self, date: str, matchup: str, home_team: str = None, home_performance: Team_Performance = None,
                 away_team: str = None, away_performance: Team_Performance = None, total_points=None):
        self.date = date
        self.matchup = matchup
        self.home_team = home_team
        self.home_performance = home_performance
        self.away_team = away_team
        self.away_performance = away_performance
        self.total_points = total_points

    def __str__(self):
        return str([self.date, self.matchup, self.home_team, self.away_team]) + '\n'


class Games:
    """
    Stores a list of _Game, and has methods for easier data access
    """

    def __init__(self):
        self.games = []

    def __str__(self):
        return str(''.join([g.__str__() for g in self.games]))

    def add_game(self, row: list):
        """
        Creates a game, and fills game out with Team_Performance, correclty assigning the home and away team

        >>> lst = [
        ... ['DEN','DEN vs. PHX','03/05/2024','L',265,107,41,96,42.7,15,40,37.5,10,14,71.4,10,41,51,25,6,6,15,15,-10],
        ... ['PHX','PHX @ DEN','03/05/2024','W',265,117,42,96,43.8,15,34,44.1,18,22,81.8,11,41,52,32,9,7,10,19,10],
        ... ['IND','IND @ DAL','03/05/2024','W',240,137,50,93,53.8,18,39,46.2,19,24,79.2,9,34,43,34,3,2,9,21,17],
        ... ['DAL','DAL vs. IND','03/05/2024','L',240,120,45,93,48.4,13,39,33.3,17,21,81.0,9,32,41,23,5,6,9,23,-17],
        ... ['SAC','SAC @ ATL','12/29/2023','W',240,117,46,89,51.7,17,45,37.8,8,15,53.3,5,33,38,35,9,3,12,20,7],
        ... ['ATL','ATL vs. SAC','12/29/2023','L',240,110,37,94,39.4,15,43,34.9,21,25,84.0,18,33,51,29,9,4,15,16,-7]]
        >>> g = Games()
        >>> for r in lst:
        ...     g.add_game(r)
        >>> print(g)
        ['03/05/2024', 'DEN vs. PHX', 'DEN', 'PHX']
        ['03/05/2024', 'IND @ DAL', 'DAL', 'IND']
        ['12/29/2023', 'SAC @ ATL', 'ATL', 'SAC']
        <BLANKLINE>
        >>> print(g.games[0].total_points)
        224
        >>> print(g.games[1].total_points)
        257
        """

        team_name = row[0]
        matchup = row[1]

        team_performance = Team_Performance(row[0], fgm=int(row[6]), fga=int(row[7]),
                                            fg_percentage=float(row[8]), three_pm=int(row[9]), three_pa=int(row[10]),
                                            three_p_percentage=float(row[11]), ftm=int(row[12]), fta=int(row[13]),
                                            ft_percentage=float(row[14]), oreb=int(row[15]), dreb=int(row[16]),
                                            reb=int(row[17]), assists=int(row[18]), steals=int(row[19]),
                                            block=int(row[20]), turnovers=int(row[21]), personal_fouls=int(row[22]),
                                            plus_minus=int(row[23]))
        for game in self.games:
            if game.date == row[2]:

                if (team_name + ' @ ' == game.matchup[:6]) or (' vs. ' + team_name == game.matchup[3:]):
                    game.away_team = team_name
                    game.away_performance = team_performance
                    game.total_points += int(row[5])
                    return
                elif (team_name + ' vs. ' == game.matchup[:8]) or (' @ ' + team_name == game.matchup[3:]):
                    game.home_team = team_name
                    game.home_performance = team_performance
                    game.total_points += int(row[5])
                    return

        if (team_name + ' @ ' == matchup[:6]) or (' vs. ' + team_name == matchup[3:]):
            game = _Game(row[2], row[1], away_team=team_name, away_performance=team_performance,
                         total_points=int(row[5]))

        else:
            game = _Game(row[2], row[1], home_team=team_name, home_performance=team_performance,
                         total_points=int(row[5]))

        self.games.append(game)

    def prepare_data(self, score) -> (list, list):
        """
        Returns and prepares the data for the model
        """
        data_so_far = []
        target = []
        for game in self.games:
            assert len(data_so_far) == len(target)
            if game.away_performance is None or game.home_performance is None:
                print(game)
                raise ValueError
            data_so_far += [game.home_performance.to_list() + game.away_performance.to_list()]
            target.append(int(game.total_points > score))
        return np.array(data_so_far), np.array(target)

    def get_stats(self, home: str, away: str) -> list[int | float]:
        """
        Returns the Estimated Weighted Average for each team when they play against each other. We do this for recency

        Preconditions:
         - {home, away}.issubset(['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND',
            'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOL', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS',
            'TOR', 'UTA', 'WAS'])
         - home != away

         I made counter = 0.1 because the NBA has suddenly easened up on defensive rules, making teams score less
         Thus, I account for this by dividing by counter + 0.1 to delflate the average stats.
         (https://theathletic.com/5357318/2024/03/22/nba-scoring-decline-numbers-points/)

        """
        props = [0] * 36
        counter = 0.1
        for game in self.games:
            if game.home_team == home and game.away_team == away:

                counter += 1
                for i in range(0, 17):
                    props[i] += game.home_performance.to_list()[i]
                for i in range(18, len(props)):
                    props[i] += game.away_performance.to_list()[i - 18]
        if counter == 0:
            raise ValueError
        return [props[i] // counter for i in range(0, len(props))]


def load_data(files: list[str], bet_score, games: Games):
    """
    Loads data for both the model and Games
    """
    for file in files:
        with (open(file) as csv_file):
            lines = csv_file.readlines()
            header = lines.pop(0).split()
            header[1: 3] = [' '.join(header[1: 3])]
            for line in lines:
                un_preped_line = line.split()
                un_preped_line[1: 4] = [' '.join(un_preped_line[1: 4])]
                games.add_game(un_preped_line)
    feature_names = header
    dataset, target = games.prepare_data(bet_score)
    return Bunch(data=dataset, target=target, feature_names=feature_names)


def get_accuracy() -> float:
    """
    Returns the % accuracy of the bets of our model.
    Our bets have been taken from:
        - https://sportsbook.fanduel.com/navigation/nba
        - https://www.teamrankings.com/nba/odds/
        - https://www.pinnacle.ca/en/?utm_source=aff&utm_medium=Covers

    I then waited until after the game to mark the result, where 1 is 'Over' and 0 is 'Under'
    There are a total of 21 amount of bets: 61.1% Accuracy!
    """

    bets = [['TOR', 'ORL', 219.5], ['BOS', 'WAS', 242.5], ['BKN', 'SAS', 218.5], ['ATL', 'LAC', 221.0],
            ['CLE', 'IND', 224.5], ['POR', 'ATL', 228.5], ['IND', 'CHI', 218.5], ['DET', 'MIN', 218.5],
            ['LAL', 'MEM', 247.5], ['HOU', 'OKC', 223.5], ['ATL', 'BOS', 225.0], ['LAL', 'PHI', 224.5],
            ['LAL', 'MEM', 256.5], ['IND', 'CHI', 221.5], ['CLE', 'CHA', 207.5], ['CLE', 'CHA', 206.5],
            ['BKN', 'WAS', 221.5], ['POR', 'ATL', 219.5], ['TOR', 'NYK', 211.5], ['SAS', 'UTA', 231.5],
            ['SAS', 'UTA', 222.5]
            # ['DEN', 'PHX', 224.5],  # Predicted under :)
            ]

    results = [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    tree_results = []
    for bet in bets:
        games = Games()
        data = load_data(['datasets/2022_23', 'datasets/2023_24'], bet[2], games)
        X, y = data.data, data.target
        clf = RandomForest(max_depth=10)
        clf.fit(X, y)

        tree_results.append(clf.predict([games.get_stats(bet[0], bet[1])]))
        print(tree_results[-1])
    assert len(results) == len(tree_results)

    return len([x for x in range(len(results)) if tree_results[x] == results[x]]) / len(results)
