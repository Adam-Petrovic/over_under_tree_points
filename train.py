"""Over Under Tree Points: decision_tree.py

Module Description
==================
This module deals with the training of our model, which mainly includes the prepping of data/loading data from our
datasets, fitting the model with that data, placing predictions, and accuracy tests.

Copyright and Usage Information
===============================

This file is provided solely for the personal and use of Adam.
All forms of distribution of this code, whether as given or with any changes, are
expressly prohibited, unless permitted by either Adam or Grant.
For more information on copyright on this material,
please message Adam at adam.petrovic2005@gmail.com

This file is Copyright (c) 2024 Adam Petrovic
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn import datasets

from random_forest import RandomForest


@dataclass
class _FieldGoalStats:
    fgm: int
    fga: int
    fg_percentage: float


@dataclass
class _ThreePointStats:
    three_pm: int
    three_pa: int
    three_p_percentage: float


@dataclass
class _FreeThrowStats:
    ftm: int
    fta: int
    ft_percentage: float


@dataclass
class _ReboundStats:
    oreb: int
    dreb: int
    reb: float


@dataclass
class _OffBallStats:
    assists: int
    steals: int
    blocks: int
    turnovers: int
    personal_fouls: int
    plus_minus: float


@dataclass
class _OffensiveStats:
    field_goal_stats: _FieldGoalStats
    three_point_stats: _ThreePointStats
    free_throw_stats: _FreeThrowStats


@dataclass
class _DefensiveStats:
    rebound_stats: _ReboundStats
    offball_stats: _OffBallStats


class TeamPerformance:
    """
    A class which stores the box score for the Team in the given _Game
    """

    offensive_stats: _OffensiveStats
    defensive_stats: _DefensiveStats

    def __init__(self, offensive_stats: _OffensiveStats, defensive_stats: _DefensiveStats) -> None:
        self.offensive_stats = offensive_stats
        self.defensive_stats = defensive_stats

    def to_list(self) -> list[int | float]:
        """
        Represents the TeamPerformance as a list
        """
        return [self.offensive_stats.field_goal_stats.fgm,
                self.offensive_stats.field_goal_stats.fga,
                self.offensive_stats.field_goal_stats.fg_percentage,
                self.offensive_stats.three_point_stats.three_pm,
                self.offensive_stats.three_point_stats.three_pa,
                self.offensive_stats.three_point_stats.three_p_percentage,
                self.offensive_stats.free_throw_stats.ftm,
                self.offensive_stats.free_throw_stats.fta,
                self.offensive_stats.free_throw_stats.ft_percentage,
                self.defensive_stats.rebound_stats.oreb,
                self.defensive_stats.rebound_stats.dreb,
                self.defensive_stats.rebound_stats.reb,
                self.defensive_stats.offball_stats.assists,
                self.defensive_stats.offball_stats.steals,
                self.defensive_stats.offball_stats.blocks,
                self.defensive_stats.offball_stats.turnovers,
                self.defensive_stats.offball_stats.personal_fouls,
                self.defensive_stats.offball_stats.plus_minus]


@dataclass
class _Matchup:
    home_team: Optional[str]
    home_performance: Optional[TeamPerformance]
    away_team: Optional[str]
    away_performance: Optional[TeamPerformance]


class _Game:
    """
    Represents an instance of a game.
    """

    date: str
    matchup_name: str
    matchup: _Matchup
    total_points: int

    def __init__(self, date: str, matchup_name: str, matchup: _Matchup, total_points: int = None) -> None:
        self.date = date
        self.matchup_name = matchup_name
        self.matchup = matchup
        self.total_points = total_points

    def __str__(self) -> str:
        return str([self.date, self.matchup_name, self.matchup.home_team, self.matchup.away_team]) + '\n'


class Games:
    """
    Stores a list of _Game, and has methods for easier data access
    """

    games: list[_Game]

    def __init__(self) -> None:
        self.games = []

    def __str__(self) -> str:
        return str(''.join([g.__str__() for g in self.games]))

    def add_game(self, row: list) -> None:
        """
        Creates a game, and fills game out with TeamPerformance, correclty assigning the home and away team

        Preconditions:
            - len(row) == 24


        >>> lst = [
        ... ['DEN','DEN vs. PHX','03/05/2024','L','265','107','41','96','42.7','15','40','37.5','10','14','71.4','10',
        ... '41','51','25','6','6','15','15','-10'],
        ... ['PHX','PHX @ DEN','03/05/2024','W','265','117','42','96','43.8','15','34','44.1','18','22','81.8','11',41,
        ... '52','32','9','7','10','19','10'],
        ... ['IND','IND @ DAL','03/05/2024','W','240','137','50','93','53.8','18','39','46.2','19','24','79.2','9','34',
        ... '43','34','3','2','9','21','17'],
        ... ['DAL','DAL vs. IND','03/05/2024','L','240','120','45','93','48.4','13','39','33.3','17','21','81.0','9',
        ... '32','41','23','5','6','9','23','-17'],
        ... ['SAC','SAC @ ATL','12/29/2023','W','240','117','46','89','51.7','17','45','37.8','8','15','53.3','5','33',
        ... '38','35',9,'3','12','20','7'],
        ... ['ATL','ATL vs. SAC','12/29/2023','L','240','110','37','94','39.4','15','43','34.9','21','25','84.0','18',
        ... 33,'51',29,'9','4','15','16','-7']]
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

        team_performance = TeamPerformance(
            _OffensiveStats(
                _FieldGoalStats(fgm=int(row[6]), fga=int(row[7]), fg_percentage=float(row[8])),
                _ThreePointStats(three_pm=int(row[9]), three_pa=int(row[10]), three_p_percentage=float(row[11])),
                _FreeThrowStats(ftm=int(row[12]), fta=int(row[13]), ft_percentage=float(row[14]))),

            _DefensiveStats(
                _ReboundStats(oreb=int(row[15]), dreb=int(row[16]), reb=int(row[17])),
                _OffBallStats(assists=int(row[18]), steals=int(row[19]), blocks=int(row[20]),
                              turnovers=int(row[21]), personal_fouls=int(row[22]), plus_minus=int(row[23]))))

        for game in self.games:
            if game.date == row[2]:

                if (team_name + ' @ ' == game.matchup_name[:6]) or (' vs. ' + team_name == game.matchup_name[3:]):
                    game.matchup.away_team = team_name
                    game.matchup.away_performance = team_performance
                    game.total_points += int(row[5])
                    return
                elif (team_name + ' vs. ' == game.matchup_name[:8]) or (' @ ' + team_name == game.matchup_name[3:]):
                    game.matchup.home_team = team_name
                    game.matchup.home_performance = team_performance
                    game.total_points += int(row[5])
                    return

        if (team_name + ' @ ' == matchup[:6]) or (' vs. ' + team_name == matchup[3:]):
            game = _Game(row[2], row[1], _Matchup(home_team=None, home_performance=None, away_team=team_name,
                                                  away_performance=team_performance), total_points=int(row[5]))

        else:
            game = _Game(row[2], row[1], _Matchup(home_team=team_name, home_performance=team_performance,
                                                  away_team=None, away_performance=None), total_points=int(row[5]))

        self.games.append(game)

    def prepare_data(self, score: float) -> (list, list):
        """
        Returns and prepares the data for the model
        """
        data_so_far = []
        target = []
        for game in self.games:
            assert len(data_so_far) == len(target)
            if game.matchup.away_performance is None or game.matchup.home_performance is None:
                raise ValueError
            data_so_far += [game.matchup.home_performance.to_list() + game.matchup.away_performance.to_list()]
            target.append(int(game.total_points > score))
        return np.array(data_so_far), np.array(target)

    def get_stats(self, home: str, away: str) -> np.ndarray:
        """
        Returns the Estimated Weighted Average for each team when they play against each other. We do this for recency

        Preconditions:
         - {home, away}.issubset(['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND',
            'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOL', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS',
            'TOR', 'UTA', 'WAS'])
         - home != away
        """

        props = [0] * 36
        counter = 0
        for game in self.games:
            if game.matchup.home_team == home and game.matchup.away_team == away:
                counter += 1
                for i in range(0, 17):
                    props[i] += game.matchup.home_performance.to_list()[i]
                for j in range(18, len(props)):
                    props[j] += game.matchup.away_performance.to_list()[j - 18]
        if counter == 0:
            raise ValueError
        return np.asarray([prop // counter for prop in props])


def load_data(files: list[str], bet_score: float, games: Games) -> Bunch:
    """
    Loads data for both the model and Games
    """
    header = ''
    for file in files:
        with open(file) as csv_file:
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
        - https://www.nba.com/nbabet
        - https://www.oddsshark.com/nba/odds

    I then waited until after the game to mark the result, where 1 is 'Over' and 0 is 'Under'
    There are a total of 21 amount of bets:  Accuracy!

    We have 11 unders, 21 overs.
    """

    bets = [['TOR', 'ORL', 219.5], ['BOS', 'WAS', 242.5], ['ATL', 'LAC', 221.0],
            ['CLE', 'IND', 224.5], ['POR', 'ATL', 228.5], ['DET', 'MIN', 218.5],
            ['LAL', 'MEM', 247.5], ['HOU', 'OKC', 223.5], ['ATL', 'BOS', 225.0], ['LAL', 'PHI', 224.5],
            ['LAL', 'MEM', 256.5], ['IND', 'CHI', 221.5], ['CLE', 'CHA', 207.5], ['CLE', 'CHA', 206.5],
            ['BKN', 'WAS', 221.5], ['POR', 'ATL', 219.5], ['TOR', 'NYK', 211.5], ['SAS', 'UTA', 231.5],
            ['SAS', 'UTA', 222.5], ['DEN', 'PHX', 224.5], ['DEN', 'CLE', 211.0], ['DEN', 'CLE', 213.5],
            ['TOR', 'PHI', 218.5], ['WAS', 'MIA', 219.5], ['BKN', 'LAL', 226.5], ['BKN', 'LAL', 214.5],
            ['CHA', 'LAC', 215.0], ['CHA', 'LAC', 215.0], ['CHA', 'LAC', 213.0], ['MIN', 'CHI', 212.0],
            ['HOU', 'DAL', 231.5], ['SAC', 'UTA', 225.5], ['NYK', 'OKC', 217.5]]

    results = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    tree_results = []
    assert len(bets) == len(results)

    print('Starting to calculate bets')
    for bet in bets:
        tree_results.append(predict_game(bet[0], bet[1], bet[2]))
        print(tree_results)

    return len([x for x in range(len(results)) if tree_results[x] == results[x]]) / len(results)


def test_model() -> float:
    """
    :return: Accuracy based on a real data
    """
    # Uses a dataset from sklearn with a predetermined accuracy of data
    test_data = sklearn.datasets.load_breast_cancer()
    print('testing group')
    X = test_data.data
    y = test_data.target

    # Splits data int training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_trees=20)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return len([x for x in range(len(y_test)) if predictions[x] == y_test[x]]) / len(y_test)


def predict_game(home_team: str, away_team: str, bet: float) -> int:
    """
    Predicts the game outcome using the home_team, away_team, and bet.
    :param home_team: Name of Home team
    :param away_team: Name of Away team
    :param bet: The score we predict the game will be over/under
    :return: 1: Over, 0: Under
    """

    games = Games()
    data = load_data(['datasets/2022_23', 'datasets/2023_24'], bet, games)
    dataset, targets = data.data, data.target
    clf = RandomForest(max_depth=20)
    clf.fit(dataset, targets)
    return 1 if clf.predict([games.get_stats(home_team, away_team)]) else 0


# if __name__ == "__main__":
#     import python_ta
#
#     python_ta.check_all(config={
#         'extra-imports': [],  # the names (strs) of imported modules
#         'allowed-io': [],  # the names (strs) of functions that call print/open/input
#         'max-line-length': 120
#     })
