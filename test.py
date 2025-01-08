from collections import Counter
import numpy as np

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(5, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])
_deck = deck.copy()
np.random.shuffle(_deck)
print(_deck)
card_play_data = {'landlord': _deck[:20],
                          'farmer': _deck[20:37],
                          'close_cards': _deck[37:46],
                          'three_landlord_cards': _deck[17:20],
                          }
for key in card_play_data:
    card_play_data[key].sort()
print(card_play_data['landlord'])
print(card_play_data['farmer'])
print(card_play_data['close_cards'])
print(card_play_data['three_landlord_cards'])

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array(action_seq_list):
    """
    此函数将历史动作序列编码为数组。
    我们编码历史15个动作。如果没有15个动作，
    则用0填充特征。由于斗地主中每三步为一个回合，
    我们将每个连续的三步动作的表示连接起来。
    最后，我们得到一个5x162的矩阵，
    该矩阵将用于LSTM进行编码。
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

if __name__ == '__main__':
    z = _action_seq_list2array(_process_action_seq([
    [14,14,14],     # 地主出三个A
    [],             # 农民"过"
    [],             # 另一个农民"过"
    [17,17],        # 地主出对2
    [20,30],        # 农民出王炸
    []             # 地主"过"
]))
    print(z)