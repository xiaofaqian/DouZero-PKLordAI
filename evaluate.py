import os 
import argparse
import pprint

import numpy as np

from douzero.env.game import GameEnv, InfoSet
from douzero.evaluation.deep_agent import DeepAgent
from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    landlordAgent = DeepAgent(position='landlord', model_path='baselines/landlord_weights.ckpt')
    farmerAgent = DeepAgent(position='farmer', model_path='baselines/farmer_weights.ckpt')
    players = {'landlord': landlordAgent, 'farmer': farmerAgent}
    while True:
        deck = []
        for i in range(5, 15):#二斗发牌，牌库中去掉3、4，这里从5开始
                deck.extend([i for _ in range(4)])
        deck.extend([17 for _ in range(4)])
        deck.extend([20, 30])
        np.random.shuffle(deck)
        card_play_data = {
                'landlord': deck[:20],
                'farmer': deck[20:37],
                'close_cards': deck[37:46],
                'three_landlord_cards': deck[17:20]
                }
        for key in card_play_data:
                card_play_data[key].sort()
        pprint.pprint(card_play_data)
        env = GameEnv(players)
        env.card_play_init(card_play_data)
        while not env.game_over:
                env.step()
        env.reset()
#     infoset = InfoSet('landlord')
#     infoset.player_hand_cards = '56677778TJKAAAA2R'
#     landlordAgent.act(infoset)