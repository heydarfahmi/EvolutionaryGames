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
        pm = 0.4
        v = 0.5

        p = random.random()
        if pm < p:
            return
        for i in [1,len(w1)-1]:
            b_s = b1[i].shape
            w_s = w1[i].shape
            b1[i] += np.random.normal(0, v, b_s)
            w1[i] += np.random.normal(0, v, w_s)
        # pass
        # TODO
        # child: an object of class `Player`

    def cross_over(selfs, player1, player2):
        w1 = player1.nn.weights
        b1 = player1.nn.bias
        w2 = player2.nn.weights
        b2 = player2.nn.bias
        new_player1 = copy.deepcopy(player1)
        new_player2 = copy.deepcopy(player2)
        new_player1.nn.weights[1] = w2[1]
        new_player1.nn.bias[1] = b2[1]
        new_player2.nn.weights[1] = w1[1]
        new_player2.nn.bias[1] = b1[1]
        return new_player1, new_player2

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            pc = random.uniform(0.1, 0.6)
            sum_fitness = float(sum(player.fitness for player in prev_players))
            prob_list = [player.fitness / sum_fitness for player in prev_players]
            new_players = list(np.random.choice(prev_players, int(num_players), prob_list))
            new_players = copy.deepcopy(new_players)
            num_of_cross_over = num_players - int(num_players * pc)
            if num_of_cross_over % 2 == 1:
                num_of_cross_over += 1

            parents = list(np.random.choice(prev_players, num_of_cross_over,prob_list))
            parents = copy.deepcopy(parents)
            np.random.shuffle(parents)
            for k in range(0, int(num_of_cross_over / 2)):
                children = self.cross_over(parents[k], parents[k + int(num_of_cross_over / 2)])
                new_players.append(children[0])
                new_players.append(children[1])

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover

            for player in new_players:
                self.mutate(player)
            new_players = new_players[:num_players]
            if len(new_players) != num_players:
                print("WHAT THE")
                raise Exception
            return new_players[:num_players]

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects
        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting

        return sorted(players, key=lambda player: player.fitness, reverse=True)[:num_players]
