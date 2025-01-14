from copy import deepcopy
from . import move_detector as md, move_selector as ms
from .move_generator import MovesGener
import random

EnvCard2RealCard = {5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {'5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

bombs = [[5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13], [14, 14, 14, 14],
         [17, 17, 17, 17], [20, 30]]

class GameEnv(object):

    def __init__(self, players):

        self.card_play_action_seq = []

        self.three_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.players = players

        self.last_move_dict = {'landlord': [],'farmer': [] }

        self.played_cards = {'landlord': [], 'farmer': [] }

        self.last_move = []
        self.last_two_moves = []

        self.num_wins = {'landlord': 0,
                         'farmer': 0}

        self.num_scores = {'landlord': 0,
                           'farmer': 0}

        self.info_sets = {'landlord': InfoSet('landlord'),'farmer': InfoSet('farmer')}

        self.bomb_num = 0
        
        self.last_pid = 'landlord'
        
        self.close_cards = []

    def card_play_init(self, card_play_data):
        self.info_sets['landlord'].player_hand_cards = \
            card_play_data['landlord']
        self.info_sets['farmer'].player_hand_cards = \
            card_play_data['farmer']
        self.close_cards = card_play_data['close_cards']
        self.three_landlord_cards = card_play_data['three_landlord_cards']
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()
        self.give_up_num = 2 if random.random() < 0.1 else 1
        #print(f"初始化游戏: {card_play_data}, 让牌数量: {self.give_up_num}")

    def game_done(self):
        if len(self.info_sets['landlord'].player_hand_cards) == 0 or \
                len(self.info_sets['farmer'].player_hand_cards) - self.give_up_num <= 0:#农民手牌数量减去地主让牌数量等于0  
            # 如果其中一个玩家打光了手牌，游戏结束。
            self.compute_player_utility()
            self.update_num_wins_scores()

            self.game_over = True

    def compute_player_utility(self):

        if len(self.info_sets['landlord'].player_hand_cards) == 0:
            self.player_utility_dict = {'landlord': 1,
                                        'farmer': -1}
        else:
            self.player_utility_dict = {'landlord': -1,
                                        'farmer': 1}

    def update_num_wins_scores(self):
        for pos, utility in self.player_utility_dict.items():
            base_score = 1 if pos == 'landlord' else 1
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner = pos
                self.num_scores[pos] += base_score * (2 ** self.bomb_num)
            else:
                self.num_scores[pos] -= base_score * (2 ** self.bomb_num)

    def get_winner(self):
        return self.winner

    def get_bomb_num(self):
        return self.bomb_num

    def step(self):
        action = self.players[self.acting_player_position].act(
            self.game_infoset)
        assert action in self.game_infoset.legal_actions

        if len(action) > 0:
            self.last_pid = self.acting_player_position

        if action in bombs:
            self.bomb_num += 1

        self.last_move_dict[
            self.acting_player_position] = action.copy()
        #print(f"{self.acting_player_position} 当前手牌：{self.info_sets[self.acting_player_position].player_hand_cards} \n 出牌: {action}")
        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)

        self.played_cards[self.acting_player_position] += action

        if self.acting_player_position == 'landlord' and \
                len(action) > 0 and \
                len(self.three_landlord_cards) > 0:
            for card in action:
                if len(self.three_landlord_cards) > 0:
                    if card in self.three_landlord_cards:
                        self.three_landlord_cards.remove(card)
                else:
                    break

        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

    def get_last_move(self):
        last_move = []
        if len(self.card_play_action_seq) != 0:
            if len(self.card_play_action_seq[-1]) == 0:
                last_move = self.card_play_action_seq[-2]
            else:
                last_move = self.card_play_action_seq[-1]

        return last_move

    def get_last_two_moves(self):
        last_two_moves = [[], []]
        for card in self.card_play_action_seq[-2:]:
            last_two_moves.insert(0, card)
            last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_acting_player_position(self):
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord'
        else:
            if self.acting_player_position == 'landlord':
                self.acting_player_position = 'farmer'
            else:
                self.acting_player_position = 'landlord'

        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        if action != []:
            for card in action:
                self.info_sets[
                    self.acting_player_position].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        mg = MovesGener(
            self.info_sets[self.acting_player_position].player_hand_cards)

        action_sequence = self.card_play_action_seq

        rival_move = []
        if len(action_sequence) != 0:
            rival_move = action_sequence[-1]
            # if len(action_sequence[-1]) == 0:
            #     rival_move = action_sequence[-2]
            # else:
            #     rival_move = action_sequence[-1]

        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
        rival_move_len = rival_type.get('len', 1)
        moves = list()

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()

        elif rival_move_type == md.TYPE_1_SINGLE:
            all_moves = mg.gen_type_1_single()
            moves = ms.filter_type_1_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_2_PAIR:
            all_moves = mg.gen_type_2_pair()
            moves = ms.filter_type_2_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_3_TRIPLE:
            all_moves = mg.gen_type_3_triple()
            moves = ms.filter_type_3_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_4_BOMB:
            all_moves = mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
            moves = ms.filter_type_4_bomb(all_moves, rival_move)

        elif rival_move_type == md.TYPE_5_KING_BOMB:
            moves = []

        elif rival_move_type == md.TYPE_6_3_1:
            all_moves = mg.gen_type_6_3_1()
            moves = ms.filter_type_6_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_7_3_2:
            all_moves = mg.gen_type_7_3_2()
            moves = ms.filter_type_7_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
            all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
            moves = ms.filter_type_8_serial_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_11_SERIAL_3_1:
            all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
            moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_12_SERIAL_3_2:
            all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
            moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_13_4_2:
            all_moves = mg.gen_type_13_4_2()
            moves = ms.filter_type_13_4_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_14_4_22:
            all_moves = mg.gen_type_14_4_22()
            moves = ms.filter_type_14_4_22(all_moves, rival_move)

        if rival_move_type not in [md.TYPE_0_PASS,
                                   md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
            moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

        if len(rival_move) != 0:  # rival_move is not 'pass'
            moves = moves + [[]]

        for m in moves:
            m.sort()

        return moves

    def reset(self):
        self.card_play_action_seq = []

        self.three_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.last_move_dict = {'landlord': [],
                               'farmer': []}

        self.played_cards = {'landlord': [],
                             'farmer': []}

        self.last_move = []
        self.last_two_moves = []

        self.info_sets = {'landlord': InfoSet('landlord'),'farmer': InfoSet('farmer')}

        self.bomb_num = 0
        self.last_pid = 'landlord'

    def get_infoset(self):
        self.info_sets[
            self.acting_player_position].last_pid = self.last_pid

        self.info_sets[
            self.acting_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.info_sets[
            self.acting_player_position].bomb_num = self.bomb_num

        self.info_sets[
            self.acting_player_position].last_move = self.get_last_move()

        self.info_sets[
            self.acting_player_position].last_two_moves = self.get_last_two_moves()

        self.info_sets[
            self.acting_player_position].last_move_dict = self.last_move_dict

        self.info_sets[self.acting_player_position].num_cards_left_dict = \
            {pos: len(self.info_sets[pos].player_hand_cards)
             for pos in ['landlord', 'farmer']}

        self.info_sets[self.acting_player_position].other_hand_cards = []
        for pos in ['landlord', 'farmer']:
            if pos != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += \
                    self.info_sets[pos].player_hand_cards
        self.info_sets[self.acting_player_position].other_hand_cards += self.close_cards
        self.info_sets[self.acting_player_position].other_hand_cards.sort()

        self.info_sets[self.acting_player_position].played_cards = \
            self.played_cards
        self.info_sets[self.acting_player_position].three_landlord_cards = \
            self.three_landlord_cards
        self.info_sets[self.acting_player_position].card_play_action_seq = \
            self.card_play_action_seq

        self.info_sets[
            self.acting_player_position].all_handcards = \
            {pos: self.info_sets[pos].player_hand_cards
             for pos in ['landlord', 'farmer']}

        return deepcopy(self.info_sets[self.acting_player_position])
    
class InfoSet(object):
    """
    游戏状态被描述为信息集(infoset)，
    包含了当前情况下的所有信息，
    例如所有玩家的手牌、历史出牌等。
    """
    def __init__(self, player_position):
        # 玩家位置，即地主、地主下家或地主上家
        self.player_position = player_position
        # 当前玩家的手牌列表
        self.player_hand_cards = None
        # 每个玩家剩余的牌数，格式为 字符串->整数 的字典
        self.num_cards_left_dict = None
        # 地主的三张底牌列表
        self.three_landlord_cards = None
        # 历史出牌记录，为二维列表
        self.card_play_action_seq = None
        # 对于当前玩家来说，其他玩家的手牌合集
        self.other_hand_cards = None
        # 当前回合的合法动作，为二维列表
        self.legal_actions = None
        # 最近一次有效的出牌
        self.last_move = None
        # 最近两次的出牌
        self.last_two_moves = None
        # 所有位置的最后一次出牌记录
        self.last_move_dict = None
        # 到目前为止所有打出的牌的列表
        self.played_cards = None
        # 所有玩家的手牌，为字典格式
        self.all_handcards = None
        # 最后一个打出有效牌的玩家位置（即非'过'的玩家）
        self.last_pid = None
        # 到目前为止打出的炸弹数量
        self.bomb_num = None
        # 地主让牌张数
        self.give_up_num = None
        

    @staticmethod
    def from_dict(data):
        """
        将字典转换为 InfoSet 实例
        Args:
            data (dict): 包含 InfoSet 属性的字典
        Returns:
            InfoSet: 新的 InfoSet 实例
        """
        info_set = InfoSet(data.get('player_position'))
        info_set.player_hand_cards = data.get('player_hand_cards')
        info_set.num_cards_left_dict = data.get('num_cards_left_dict')
        info_set.three_landlord_cards = data.get('three_landlord_cards')
        info_set.card_play_action_seq = data.get('card_play_action_seq')
        info_set.other_hand_cards = data.get('other_hand_cards')
        info_set.legal_actions = data.get('legal_actions')
        info_set.last_move = data.get('last_move')
        info_set.last_two_moves = data.get('last_two_moves')
        info_set.last_move_dict = data.get('last_move_dict')
        info_set.played_cards = data.get('played_cards')
        info_set.all_handcards = data.get('all_handcards')
        info_set.last_pid = data.get('last_pid')
        info_set.bomb_num = data.get('bomb_num')
        info_set.give_up_num = data.get('give_up_num')
        return info_set


