from collections import Counter
import numpy as np

from douzero.env.game import GameEnv

Card2Column = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5,
               11: 6, 12: 7, 13: 8, 14: 9, 17: 10}  # 删除3(3)和4(4)的映射

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(5, 15):#二斗发牌，牌库中去掉3、4，这里从5开始
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, objective):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['landlord', 'farmer']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players)

        self.infoset = None

    def reset(self):
        """
        每次调用reset时，环境将重新初始化，
        并生成一副新的牌。通常在游戏结束后调用。
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'landlord': _deck[:20],
                          'farmer': _deck[20:37],
                          'close_cards': _deck[37:46],
                          'three_landlord_cards': _deck[17:20],
                          }
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            elif self.objective == 'logadp':
                return bomb_num + 1.0
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            elif self.objective == 'logadp':
                return -bomb_num - 1.0
            else:
                return -1.0

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action

def get_obs(infoset):
    """
    此函数从信息集(infoset)中获取不完整信息的观察值。
    由于我们对不同位置编码不同的特征，所以有二个分支。
    
    此函数将返回一个名为`obs`的字典。它包含几个字段，
    这些字段将用于训练模型。可以调整这些特征来提升性能。

    `position`: 字符串，可以是 landlord/farmer

    `x_batch`: 特征批次(不包括历史动作)。
    同时也包含动作特征

    `z_batch`: 仅包含历史动作的特征批次

    `legal_actions`: 合法动作列表

    `x_no_action`: 特征(不包括历史动作和动作特征)。
    没有批次维度

    `z`: 与z_batch相同但不是批次形式
    """
    if infoset.player_position == 'landlord':
        return _get_obs_landlord(infoset)
    elif infoset.player_position == 'farmer':
        return _get_obs_farmer(infoset)
    else:
        raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    将动作（一个整数列表）转换为卡牌矩阵的实用函数。
    这里我们移除了6个总是为零的条目，并将表示形式展平。
    
    参数:
        list_cards: 包含卡牌数字的列表
        
    返回:
        一个长度为46的一维numpy数组，包含:
        - 44个值表示普通牌（4x11矩阵展平）
        - 2个值表示大小王
    """
    if len(list_cards) == 0:
        return np.zeros(46, dtype=np.int8)

    matrix = np.zeros([4, 11], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1  # 小王
        elif card == 30:
            jokers[1] = 1  # 大王
    return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 46))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(7, 92)
    return action_seq_array

def _process_action_seq(sequence, length=14):
    """
    一个用于编码历史动作的实用函数。
    我们编码最近14个动作。如果动作数量不足14个，
    则用空列表进行填充。
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    将炸弹数量编码为one-hot表示的实用函数。
    """
    one_hot = np.zeros(13)
    one_hot[bomb_num] = 1
    return one_hot

def _get_give_up_num(give_up_num):
    """
    获取地主让牌数量
    """
    one_hot = np.zeros(8)
    one_hot[give_up_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    farmer_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['farmer'], 17)
    
    farmer_num_cards_left_batch = np.repeat(
        farmer_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    farmer_played_cards = _cards2array(
        infoset.played_cards['farmer'])
    
    farmer_played_cards_batch = np.repeat(
        farmer_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    
    give_up_num = _get_give_up_num(
        infoset.give_up_num)
    give_up_num_batch = np.repeat(
        give_up_num[np.newaxis, :],
        num_legal_actions, axis=0)
    
    last_farmer_action = _cards2array(
        infoset.last_move_dict['farmer'])
    last_farmer_action_batch = np.repeat(
        last_farmer_action[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         last_farmer_action_batch,
                         farmer_played_cards_batch,
                         farmer_num_cards_left_batch,
                         bomb_num_batch,
                         give_up_num_batch,
                         my_action_batch))
    
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             last_farmer_action,
                             farmer_played_cards,
                             farmer_num_cards_left,
                             bomb_num,
                             give_up_num))
    
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    
    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_farmer(infoset):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    
    give_up_num = _get_give_up_num(
        infoset.give_up_num)
    give_up_num_batch = np.repeat(
        give_up_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         landlord_num_cards_left_batch,
                         bomb_num_batch,
                         give_up_num_batch,
                         my_action_batch))
    
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             last_action,
                             last_landlord_action,
                             landlord_num_cards_left,
                             bomb_num,
                             give_up_num))
    
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'farmer',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs
