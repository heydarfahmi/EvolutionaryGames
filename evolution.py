import math

from player import Player
import numpy as np
from config import CONFIG
import copy
import random


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        w1 = child.nn.weights
        b1 = child.nn.bias
        pm = 0.3
        v = 0.5
        random.seed(1)
        p = random.random()
        if pm < p:
            return
        for i in [1, len(w1) - 1]:
            b_s = b1[i].shape
            w_s = w1[i].shape
            child.nn.bias[i] += np.random.normal(0, 0.2, b_s)
            child.nn.weights[i] += np.random.normal(0, 0.2, w_s)
        return child

    def cross_over(selfs, player1, player2):
        w1 = player1.nn.weights
        b1 = player1.nn.bias
        w2 = player2.nn.weights
        b2 = player2.nn.bias
        player1.nn.weights[1] = w2[1]
        player1.nn.bias[1] = b2[1]
        player2.nn.weights[1] = w1[1]
        player2.nn.bias[1] = b1[1]

    def generate_new_population(self, num_players, prev_players=None):
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            sum_fitness = float(sum(player.fitness for player in prev_players))
            prob_list = [player.fitness / sum_fitness for player in prev_players]
            new_players = list(np.random.choice(prev_players, int(num_players), prob_list))
            new_players = copy.deepcopy(new_players)

            for player_index in range(len(new_players)):
                new_players[player_index] = self.mutate(new_players[player_index])

            return new_players

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects
        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting
        return sorted(players, key=lambda player: player.fitness, reverse=True)[:num_players]
