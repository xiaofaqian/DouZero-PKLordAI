from bisect import bisect_left
import collections
from itertools import combinations
import os
import json
from collections import Counter, OrderedDict, defaultdict
import threading
import time
from typing import Dict, List
import uuid

import numpy as np
from pydantic import BaseModel



# 读取所需的JSON文件路径
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

# 读取特定动作映射的JSON文件
with open(os.path.join(ROOT_PATH, 'jsondata', 'specific_map.json'), 'r') as file:
    SPECIFIC_MAP = json.load(file, object_pairs_hook=OrderedDict)

# 读取动作空间的JSON文件，并获取所有动作列表
with open(os.path.join(ROOT_PATH, 'jsondata', 'action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())

# 读取牌型的JSON文件
with open(os.path.join(ROOT_PATH, 'jsondata', 'card_type.json'), 'r') as file:
    data = json.load(file, object_pairs_hook=OrderedDict)
    CARD_TYPE = data

# 读取类型对应卡牌的JSON文件
with open(os.path.join(ROOT_PATH, 'jsondata', 'type_card.json'), 'r') as file:
    TYPE_CARD = json.load(file, object_pairs_hook=OrderedDict)

# 定义牌的排名顺序（字符串表示）
CARD_RANK_STR = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
                'A', '2', 'B', 'R']

# 创建牌排名字符串到索引的映射
CARD_RANK_STR_INDEX = {card: index for index, card in enumerate(CARD_RANK_STR)}

# 定义牌的排名顺序（列表形式，包含大王和小王）
CARD_RANK = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
            'A', '2', 'BJ', 'RJ']
CARD_MAP = {
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'T': 10,
    'J': 11,
    'Q': 12,
    'K': 13,
    'A': 14,
    '2': 17,
    'B': 20,  # 小王
    'R': 30   # 大王
}

REVERSE_CARD_MAP = {v: k for k, v in CARD_MAP.items()}

# 按照牌的排名顺序排序
INDEX = OrderedDict(sorted(CARD_RANK_STR_INDEX.items(), key=lambda t: t[1]))

class PlayableCardsModel(BaseModel):
    cards: str
    seat: int
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
           cards = data.get('cards', ''),
           seat = data.get('seat', 0),
        ) 


class PredictPutCardModel(BaseModel):
    init_card: str
    current_hand: str
    opponent_hands: str
    fundcards: str
    other_hands: str
    current_multiplier: int
    self_seat: int
    landlord_seat: int
    pk_status: int
    self_win_card_num:int
    oppo_win_card_num:int
    playables: List[PlayableCardsModel]

    @classmethod
    def from_dict(cls, data: dict):
        """
        将字典转换为 PredictPutCardModel 对象的类方法
        
        Args:
            data (dict): 包含模型所需数据的字典
            
        Returns:
            PredictPutCardModel: 转换后的对象
        """
        # 确保 playables 字段正确转换
        playables = []
        if 'playables' in data:
            for playable in data['playables']:
                if isinstance(playable, dict):
                    # 假设 PlayableCardsModel 也有一个 from_dict 方法
                    playables.append(PlayableCardsModel.from_dict(playable))
                else:
                    playables.append(playable)
        
        return cls(
            init_card=data.get('init_card', ''),
            current_hand=data.get('current_hand', ''),
            opponent_hands=data.get('opponent_hands', ''),
            fundcards=data.get('fundcards', ''),
            other_hands=data.get('other_hands', ''),
            current_multiplier=data.get('current_multiplier', 0),
            self_seat=data.get('self_seat', 0),
            landlord_seat=data.get('landlord_seat', 0),
            pk_status=data.get('pk_status', 0),
            self_win_card_num=data.get('self_win_card_num', 0),
            oppo_win_card_num=data.get('oppo_win_card_num', 0),
            playables=playables
        )


async def generate_unique_task_id(db) -> str:
    """
    处理用户账单和任务相关的服务类
    """
    while True:
        task_id = uuid.uuid4().hex + uuid.uuid4().hex[:16]
        existing_task = await db.user_tasks.find_one({"task_id": task_id})
        if not existing_task:
            return task_id

# 获取当前时间戳（毫秒级）
def get_current_timestamp():
    """
    获取当前时间戳（毫秒级）

    返回:
        int: 当前时间戳（毫秒级）
    """
    return int(round(time.time() * 1000))

def convert_card_format(card_str: str) -> str:
    """
    将各种牌面格式转换为标准格式

    参数:
        card_str (str): 需要转换的牌面字符串，例如 "10,10", "10"

    返回:
        str: 转换后的标准牌面字符串，例如 "TT", "T"
    """
    # 定义转换映射
    convert_map = {
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
        '10': 'T',
        'J': 'J',
        'Q': 'Q', 
        'K': 'K',
        'A': 'A',
        '2': '2',
        'X': 'B',  # 小王
        'D': 'R'   # 大王
    }

    # 处理逗号分隔的情况
    if ',' in card_str:
        parts = card_str.split(',')
        converted_parts = [convert_map.get(part, part) for part in parts]
        return ''.join(converted_parts)

    # 处理单个数字的情况
    return convert_map.get(card_str, card_str)

def get_reverse_card_map(card_id:int):
    return REVERSE_CARD_MAP[card_id]

def get_card_type(card:str):
    '''
        获取卡牌类型
    '''
    return CARD_TYPE[card]

def get_card_str_by_action_id(action_id:int,card_list:List[str]):
    '''
        根据动作ID获取卡牌字符串
    '''
    action = 'pass'
    for key,value in ACTION_SPACE.items():
        if value == action_id:
            action = key
            break
    action = action.replace('*','')
    for card in card_list:
       if action in card:
           return card
    return None


def get_legal_action_ids(legal_actions):
    ''' Get all legal actions for current state

    Returns:
        legal_actions (list): a list of legal actions' id
    '''
    legal_action_id = []
    if legal_actions:
        for action in legal_actions:
            for abstract in SPECIFIC_MAP[action]:
                action_id = ACTION_SPACE[abstract]
                if action_id not in legal_action_id:
                    legal_action_id.append(action_id)
    return legal_action_id


class LocalObjs(threading.local):
    """
    本地线程对象，缓存候选卡牌相关数据
    """
    def __init__(self):
        self.cached_candidate_cards = None
        self.cached_candidate_cards_dict = None


_local_objs = LocalObjs()


def contains_cards(candidate, target):
    """
    判断候选卡牌是否包含目标卡牌

    :param candidate: 候选卡牌列表
    :param target: 目标卡牌列表
    :return: 如果candidate包含target，返回True；否则返回False
    """
    if _local_objs.cached_candidate_cards != candidate:
        # 如果缓存的候选卡牌不同，则重新计算
        _local_objs.cached_candidate_cards = candidate
        cards_dict = defaultdict(int)
        for card in candidate:
            cards_dict[card] += 1
        _local_objs.cached_candidate_cards_dict = cards_dict
    else:
        # 使用缓存的卡牌字典
        cards_dict = _local_objs.cached_candidate_cards_dict
    # 统计目标卡牌的数量
    target_count = defaultdict(int)
    for card in target:
        target_count[card] += 1
    # 检查每张目标卡牌的数量是否不超过候选卡牌中的数量
    return all(cards_dict[card] >= count for card, count in target_count.items())


def get_gt_cards(played_cards, current_hand):
    """
    获取比当前出牌更大的所有可能的卡牌组合

    :param played_cards: 当前已打出的卡牌
    :param current_hand: 玩家当前手牌的字符串表示
    :return: 可以比当前出牌更大的卡牌组合列表，包含 'pass'
    """
    if not played_cards or played_cards == "" or played_cards == "pass":
        return playable_cards_from_hand(current_hand)
    gt_cards = ['pass']
    target_types = CARD_TYPE[played_cards]
    type_dict = {}
    for card_type, weight in target_types:
        # 将 weight 转换为整数
        type_dict[card_type] = max(type_dict.get(card_type, -1), int(weight))
    if 'rocket' in type_dict:
        # 如果当前出牌是火箭，则不能再出更大的牌
        return gt_cards
    type_dict.setdefault('rocket', -1)
    type_dict.setdefault('bomb', -1)
    for card_type, weight in type_dict.items():
        candidates = TYPE_CARD.get(card_type, {})
        for can_weight, cards_list in candidates.items():
            if int(can_weight) > int(weight):
                for cards in cards_list:
                    if contains_cards(current_hand, cards) and cards not in gt_cards:
                        gt_cards.append(cards)
    return gt_cards

def get_bombs_rockets(current_hand: str) -> List[str]:
    '''获取手牌中的炸弹和王炸列表

    参数:
        current_hand (str): 手牌字符串。例如: '56888TTQKKKAA222RB'

    返回:
        List[str]: 炸弹和王炸列表。例如: ['KKKK', '2222', 'BR']
    '''
    bombs = []
    # 检查王炸
    if 'B' in current_hand and 'R' in current_hand:
        bombs.append('BR')
        
    # 检查普通炸弹
    length = len(current_hand)
    i = 0
    while i <= length - 4:
        # 如果连续4张牌相同，则为炸弹
        if current_hand[i] == current_hand[i+3]:
            bombs.append(current_hand[i] * 4)
            i += 4
        else:
            i += 1
            
    return bombs

def get_landlord_score(current_hand):
    ''' 粗略判断手牌的质量，并提供一个分数作为叫地主的依据。

    参数:
        current_hand (str): 手牌字符串。例如: '56888TTQKKKAA222R'

    返回:
        int: 分数
    '''
    score_map = {'A': 1, '2': 2, 'B': 3, 'R': 4}
    score = 0
    # 火箭
    if current_hand[-2:] == 'BR':
        score += 3
        current_hand = current_hand[:-2]
    length = len(current_hand)
    i = 0
    while i < length:
        # 炸弹
        if i <= (length - 4) and current_hand[i] == current_hand[i+3] and current_hand[i] not in score_map:
            score += 6
            i += 4
            continue
        # 2, 小王, 大王
        if current_hand[i] in score_map:
            score += score_map[current_hand[i]]
        i += 1
    return score

def chain_indexes(indexes_list):
        ''' 查找单张、对子和三张的顺子

        参数:
            indexes_list: 具有相同数量的牌的索引列表，数量可以是1, 2或3。

        返回:
            元组列表: [(start_index1, length1), (start_index1, length1), ...]

        '''
        chains = []
        prev_index = -100
        count = 0
        start = None
        for i in indexes_list:
            if (i[0] >= 12): # '2BR'没有顺子
                break
            if (i[0] == prev_index + 1):
                count += 1
            else:
                if (count > 1):
                    chains.append((start, count))
                count = 1
                start = i[0]
            prev_index = i[0]
        if (count > 1):
            chains.append((start, count))
        return chains
    
def pair_attachments(cards_count, chain_start, chain_length, size):
        ''' 查找三带对和四带二对的对子附件

        参数:
            cards_count: 牌的数量
            chain_start: 三带或四带的起始牌的索引
            chain_length: 顺子的长度，对于三带对或四带二对为1
            size: 附件对的数量

        返回:
            元组列表: [attachment1, attachment2, ...]
                      每个附件有两个元素，
                      第一个包含小于chain_start索引的附加牌的索引，
                      第二个包含大于chain_start索引的附加牌的索引
        '''
        attachments = set()
        candidates = []
        for i, _ in enumerate(cards_count):
            if (i >= chain_start and i < chain_start + chain_length):
                continue
            if (cards_count[i] == 2 or cards_count[i] == 3):
                candidates.append(i)
            elif (cards_count[i] == 4):
                candidates.append(i)
        for attachment in combinations(candidates, size):
            if (attachment[-1] == 14 and attachment[-2] == 13):
                continue
            i = bisect_left(attachment, chain_start)
            attachments.add((attachment[:i], attachment[i:]))
        return list(attachments)
    
def solo_attachments(hands, chain_start, chain_length, size):
        ''' 查找三带单和四带二的单牌附件

        参数:
            hands: 手牌
            chain_start: 三带或四带的起始牌的索引
            chain_length: 顺子的长度，对于三带单或四带二为1
            size: 附件单牌的数量

        返回:
            元组列表: [attachment1, attachment2, ...]
                      每个附件有两个元素，
                      第一个包含小于chain_start索引的附加牌的索引，
                      第二个包含大于chain_start索引的附加牌的索引
        '''
        attachments = set()
        candidates = []
        prev_card = None
        same_card_count = 0
        for card in hands:
            # 不计算顺子中的牌
            if (CARD_RANK_STR_INDEX[card] >= chain_start and CARD_RANK_STR_INDEX[card] < chain_start + chain_length):
                continue
            if (card == prev_card):
                # 附件不能有炸弹
                if (same_card_count == 3):
                    continue
                # 附件不能有与三带连续的3张相同牌（除了3张'222'）
                elif (same_card_count == 2 and (CARD_RANK_STR_INDEX[card] == chain_start - 1 or CARD_RANK_STR_INDEX[card] == chain_start + chain_length) and card != '2'):
                    continue
                else:
                    same_card_count += 1
            else:
                prev_card = card
                same_card_count = 1
            candidates.append(CARD_RANK_STR_INDEX[card])
        for attachment in combinations(candidates, size):
            if (attachment[-1] == 14 and attachment[-2] == 13):
                continue
            i = bisect_left(attachment, chain_start)
            attachments.add((attachment[:i], attachment[i:]))
        return list(attachments)
    
def playable_cards_from_hand(current_hand):
        ''' 获取手牌中可出的牌

        返回:
            set: 可出牌的字符串集合
        '''
        cards_dict = collections.defaultdict(int)
        for card in current_hand:
            cards_dict[card] += 1
        cards_count = np.array([cards_dict[k] for k in CARD_RANK_STR])
        playable_cards = set()

        non_zero_indexes = np.argwhere(cards_count > 0)
        more_than_1_indexes = np.argwhere(cards_count > 1)
        more_than_2_indexes = np.argwhere(cards_count > 2)
        more_than_3_indexes = np.argwhere(cards_count > 3)
        # 单牌
        for i in non_zero_indexes:
            playable_cards.add(CARD_RANK_STR[i[0]])
        # 对子
        for i in more_than_1_indexes:
            playable_cards.add(CARD_RANK_STR[i[0]] * 2)
        # 炸弹，四带二单，四带二对
        for i in more_than_3_indexes:
            cards = CARD_RANK_STR[i[0]] * 4
            playable_cards.add(cards)
            for left, right in solo_attachments(current_hand, i[0], 1, 2):
                pre_attached = ''
                for j in left:
                    pre_attached += CARD_RANK_STR[j]
                post_attached = ''
                for j in right:
                    post_attached += CARD_RANK_STR[j]
                playable_cards.add(pre_attached + cards + post_attached)
            for left, right in pair_attachments(cards_count, i[0], 1, 2):
                pre_attached = ''
                for j in left:
                    pre_attached += CARD_RANK_STR[j] * 2
                post_attached = ''
                for j in right:
                    post_attached += CARD_RANK_STR[j] * 2
                playable_cards.add(pre_attached + cards + post_attached)

        # 单顺5 -- 单顺12
        solo_chain_indexes = chain_indexes(non_zero_indexes)
        for (start_index, length) in solo_chain_indexes:
            s, l = start_index, length
            while(l >= 5):
                cards = ''
                curr_index = s - 1
                curr_length = 0
                while (curr_length < l and curr_length < 12):
                    curr_index += 1
                    curr_length += 1
                    cards += CARD_RANK_STR[curr_index]
                    if (curr_length >= 5):
                        playable_cards.add(cards)
                l -= 1
                s += 1

        # 连对3 -- 连对10
        pair_chain_indexes = chain_indexes(more_than_1_indexes)
        for (start_index, length) in pair_chain_indexes:
            s, l = start_index, length
            while(l >= 3):
                cards = ''
                curr_index = s - 1
                curr_length = 0
                while (curr_length < l and curr_length < 10):
                    curr_index += 1
                    curr_length += 1
                    cards += CARD_RANK_STR[curr_index] * 2
                    if (curr_length >= 3):
                        playable_cards.add(cards)
                l -= 1
                s += 1

        # 三张，三带一，三带二
        for i in more_than_2_indexes:
            playable_cards.add(CARD_RANK_STR[i[0]] * 3)
            for j in non_zero_indexes:
                if (j < i):
                    playable_cards.add(CARD_RANK_STR[j[0]] + CARD_RANK_STR[i[0]] * 3)
                elif (j > i):
                    playable_cards.add(CARD_RANK_STR[i[0]] * 3 + CARD_RANK_STR[j[0]])
            for j in more_than_1_indexes:
                if (j < i):
                    playable_cards.add(CARD_RANK_STR[j[0]] * 2 + CARD_RANK_STR[i[0]] * 3)
                elif (j > i):
                    playable_cards.add(CARD_RANK_STR[i[0]] * 3 + CARD_RANK_STR[j[0]] * 2)

        # 三带一，三带二，三张 -- 三顺2 -- 三顺6; 三带一顺2 -- 三带一顺5; 三带二顺2 -- 三带二顺4
        trio_chain_indexes = chain_indexes(more_than_2_indexes)
        for (start_index, length) in trio_chain_indexes:
            s, l = start_index, length
            while(l >= 2):
                cards = ''
                curr_index = s - 1
                curr_length = 0
                while (curr_length < l and curr_length < 6):
                    curr_index += 1
                    curr_length += 1
                    cards += CARD_RANK_STR[curr_index] * 3

                    # 三顺2到三顺6
                    if (curr_length >= 2 and curr_length <= 6):
                        playable_cards.add(cards)

                    # 三带一顺2到三带一顺5
                    if (curr_length >= 2 and curr_length <= 5):
                        for left, right in solo_attachments(current_hand, s, curr_length, curr_length):
                            pre_attached = ''
                            for j in left:
                                pre_attached += CARD_RANK_STR[j]
                            post_attached = ''
                            for j in right:
                                post_attached += CARD_RANK_STR[j]
                            playable_cards.add(pre_attached + cards + post_attached)

                    # 三带二顺2 -- 三带二顺4
                    if (curr_length >= 2 and curr_length <= 4):
                        for left, right in pair_attachments(cards_count, s, curr_length, curr_length):
                            pre_attached = ''
                            for j in left:
                                pre_attached += CARD_RANK_STR[j] * 2
                            post_attached = ''
                            for j in right:
                                post_attached += CARD_RANK_STR[j] * 2
                            playable_cards.add(pre_attached + cards + post_attached)
                l -= 1
                s += 1
        # 火箭
        if (cards_count[13] and cards_count[14]):
            playable_cards.add(CARD_RANK_STR[13] + CARD_RANK_STR[14])
        return playable_cards
    
    
class DataTransformer:
    @staticmethod
    def transform(predict_model_data: PredictPutCardModel) -> Dict:
        # 初始化结果字典
        result = {}
        # 确定玩家位置
        give_up_num = 1
        if predict_model_data.self_seat == predict_model_data.landlord_seat:
            result['player_position'] = 'landlord'
            give_up_num = predict_model_data.oppo_win_card_num
        else:
            result['player_position'] = 'farmer'
            give_up_num = predict_model_data.self_win_card_num
        result['give_up_num'] = give_up_num
        last_playable = predict_model_data.playables[-1].cards if predict_model_data.playables and predict_model_data.playables[-1].cards else "pass"
        
        if not predict_model_data.playables or predict_model_data.playables[-1].seat == predict_model_data.self_seat or last_playable == "pass":
            legal_actions = list(playable_cards_from_hand(predict_model_data.current_hand))
        else:
            legal_actions = get_gt_cards(last_playable, predict_model_data.current_hand)
        
        result['legal_actions'] = [DataTransformer.cards_to_nums(action) for action in legal_actions]
        
        result['other_hand_cards'] = DataTransformer.calculate_other_hand_cards(predict_model_data, result)
        # 处理玩家手牌
        player_hand_cards = DataTransformer.cards_to_nums(predict_model_data.current_hand)
        result['player_hand_cards'] = player_hand_cards

        # 初始化各玩家剩余手牌数量
        num_cards_left_dict = {
            'landlord': 0,
            'farmer': 0
        }
       
        # 假设初始手牌数为17张，底牌归地主
        if result['player_position'] == 'landlord':
            num_cards_left_dict['landlord'] = len(player_hand_cards)
            num_cards_left_dict['farmer'] = 17  # 农民初始手牌数为17
        else:
            num_cards_left_dict['landlord'] = 20     # 地主初始手牌数为20
            num_cards_left_dict['farmer'] = len(player_hand_cards)

        result['num_cards_left_dict'] = num_cards_left_dict

        # 处理底牌（fundcards）
        three_landlord_cards = DataTransformer.cards_to_nums(predict_model_data.fundcards)
        result['three_landlord_cards'] = three_landlord_cards

        # 处理出牌序列
        card_play_action_seq = []
        last_move_dict = {
            'landlord': [],
            'farmer': []
        }
        played_cards = {
            'landlord': [],
            'farmer': []
        }
        last_pid = ''

        bomb_num = 0  # 炸弹数量

        for playable in predict_model_data.playables:
            if playable.cards:
                # 获取出牌的玩家
                if playable.seat == predict_model_data.landlord_seat:
                    pid = 'landlord'
                else:
                    pid = 'farmer'

                # 记录最后一个出牌玩家
                last_pid = pid

                # 将出牌转为数字列表
                played_card_nums = DataTransformer.cards_to_nums(playable.cards)
                card_play_action_seq.append(played_card_nums)

                # 更新 last_move_dict
                last_move_dict[pid] = played_card_nums

                # 更新 played_cards
                played_cards[pid].extend(played_card_nums)

                # 更新剩余手牌数量
                num_cards_left_dict[pid] -= len(played_card_nums)

                # 计算炸弹数量
                card_type = get_card_type(playable.cards)
                if card_type and (card_type[0][0] == 'bomb' or card_type[0][0] == 'rocket'):
                    bomb_num += 1
            else:
                # 跳过不出牌的情况
                card_play_action_seq.append([])

        # 更新最后两手牌
        if len(card_play_action_seq) >= 2:
            last_two_moves = card_play_action_seq[-2:]
        else:
            last_two_moves = card_play_action_seq

        result['card_play_action_seq'] = card_play_action_seq
        result['last_move'] = last_move_dict[last_pid] if last_pid else []
        result['last_two_moves'] = last_two_moves
        result['last_move_dict'] = last_move_dict
        result['played_cards'] = played_cards

        # 二人斗地主，没有 'landlord_down' 的数据，置为空列表
        result['all_handcards'] = {
            'landlord': [],
            'farmer': []
        }
        if result['player_position'] == 'landlord':
            result['all_handcards']['landlord'] = player_hand_cards
            result['all_handcards']['farmer'] = []  # 无法知道农民的手牌
        else:
            result['all_handcards']['landlord'] = []  # 无法知道地主的手牌
            result['all_handcards']['farmer'] = player_hand_cards

        result['last_pid'] = last_pid
        result['bomb_num'] = bomb_num

        # 初始化手牌
        init_card = {
            'landlord': [],
            'farmer': [],
            'three_landlord_cards': three_landlord_cards
        }
        init_hand_cards = DataTransformer.cards_to_nums(predict_model_data.init_card)
        if result['player_position'] == 'landlord':
            init_card['landlord'] = init_hand_cards
        else:
            init_card['farmer'] = init_hand_cards
        result['init_card'] = init_card

        return result

    @staticmethod
    def cards_to_nums(cards_str: str) -> List[int]:
        """
        将卡牌字符串转换为数字列表
        """
        if cards_str == "pass":
            return []
        return [CARD_MAP[card] for card in cards_str]
    
    @staticmethod
    def nums_to_cards(nums: List[int]) -> str:
        """
        将数字列表转换为卡牌字符串
        
        参数:
            nums (List[int]): 卡牌数字列表
            
        返回:
            str: 转换后的卡牌字符串，如果列表为空则返回"pass"
        """
        if not nums:
            return "pass"
        return ''.join(REVERSE_CARD_MAP[num] for num in nums)
    
    @staticmethod
    def calculate_other_hand_cards(predict_model_data: PredictPutCardModel, result: Dict) -> List[int]:
        all_cards = ['5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2', 'B', 'R']
        full_deck = []
        for card in all_cards:
            count = 4  # Normal cards have 4
            if card in ['B', 'R']:
                count = 1  # Jokers have 1
            full_deck.extend([CARD_MAP[card]] * count)

        # Create a Counter for the full deck
        full_deck_counter = Counter(full_deck)

        # Extract and convert current hand to numbers
        current_hand_numbers = DataTransformer.cards_to_nums(predict_model_data.current_hand)
        current_hand_counter = Counter(current_hand_numbers)

        # Subtract hand cards from the full deck
        full_deck_counter -= current_hand_counter

        # Loop through playables and remove played cards
        for playable in predict_model_data.playables:
            played_cards_numbers = DataTransformer.cards_to_nums(playable.cards)
            played_cards_counter = Counter(played_cards_numbers)
            full_deck_counter -= played_cards_counter

        # The remaining cards in the full_deck_counter 
        # are the ones that have not appeared yet
        other_hand_cards = list(full_deck_counter.elements())
        return other_hand_cards


