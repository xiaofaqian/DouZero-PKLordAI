import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

from douzero.env.game import InfoSet
from douzero.evaluation.deep_agent import DeepAgent
from utils import DataTransformer, PredictPutCardModel, get_bombs_rockets, get_gt_cards

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="PKLord Game API",
    description="斗地主游戏预测接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PklordLocal:
    
    @staticmethod
    def play_cards(request):
        
        # print(f"request: {request}")
        if request.pk_status != 0:
            return 'pass'
        
        current_hand = request.current_hand
        lastMove = request.playables[-1].cards if request.playables else None
        actions =  get_gt_cards(lastMove,current_hand)
        # print(f"actions: {actions}")
        if len(actions) <= 1:
            return "pass"
        # 去掉 "pass"
        actions = [action for action in actions if action != "pass"]
        # print(f"actions: {actions}")
        # 只能出炸弹，就出炸弹
        bombs = get_bombs_rockets(current_hand)
        # print(f"bombs: {bombs}")
        if len(bombs) != 0 and len(bombs) >= len(actions):
            return bombs[0]
        # 去除炸弹后，最长的一个action
        non_bomb_actions = [action for action in actions if action not in bombs]
        # print(f"actions: {non_bomb_actions}")
        if non_bomb_actions:  # 如果非炸弹动作不为空
            return max(non_bomb_actions, key=len)  # 返回长度最长的一个action
        
        return "pass"
# 响应模型
class PredictResponse(BaseModel):
    code: int = 200
    msg: str = ""
    data: str = ""

farmer_agent = DeepAgent(position='farmer', model_path='baselines/farmer_weights.ckpt')
landlord_agent = DeepAgent(position='landlord', model_path='baselines/landlord_weights.ckpt')

@app.get("/")
async def root():
    """健康检查接口"""
    return {"status": "ok", "message": "PKLord Game API is running"}

@app.post("/play/pklord/predictPutCard", response_model=PredictResponse)
async def predict_put_card(request: PredictPutCardModel):
    """
    预测下一步最优出牌
    
    Args:
        request: 包含当前游戏状态的请求对象
        
    Returns:
        PredictResponse: 预测结果响应
    """
    try:
        # 将 request 转换为 JSON 并打印
        request_json = json.dumps(request.model_dump(), ensure_ascii=False, indent=2)
        logger.info(f"Received prediction request JSON:\n{request_json}")
        # 如果游戏状态不是2，则使用本地策略
        if request.pk_status != 2:
            playable = PklordLocal.play_cards(request)
            return PredictResponse(
                code=200,
                msg="预测成功",
                data=playable
            )
            
        infoset = DataTransformer.transform(request)
        info =  InfoSet.from_dict(infoset)
        if info.player_position == 'farmer':
            agent = farmer_agent
        else:
            agent = landlord_agent
        prediction_result = agent.act(info)
        print(f"上手牌: {info.last_move} \n 当前手牌: {info.player_hand_cards} \n 预测结果: {prediction_result}")
        prediction_result = DataTransformer.nums_to_cards(prediction_result)
        return PredictResponse(
            code=200,
            msg="预测成功",
            data=prediction_result
        )
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return PredictResponse(
            code=400,
            msg=str(ve),
            data=""
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return PredictResponse(
            code=500,
            msg=f"预测失败: {str(e)}",
            data=""
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"Global exception: {str(exc)}")
    return {
        "code": 500,
        "msg": "服务器内部错误",
        "data": ""
    }

# 启动服务器配置 uvicorn api:app --reload --host 0.0.0.0 --port 8003
if __name__ == "__main__":
#     putCard = PredictPutCardModel.from_dict(json.loads("""
#                                             {
#     "init_card": "578999TJQQKKKAA22",
#     "current_hand": "578999TJQQKKKAA22",
#     "opponent_hands": "",
#     "fundcards": "56T",
#     "other_hands": "",
#     "current_multiplier": 1,
#     "self_seat": 1,
#     "landlord_seat": 0,
#     "pk_status": 2,
#     "self_win_card_num": 1,
#     "oppo_win_card_num": 0,
#     "playables": [
#       {
#         "cards": "5",
#         "seat": 0
#       }
#     ]
#   }
#                                             """))
#     infoset = DataTransformer.transform(putCard)
#     info =  InfoSet.from_dict(infoset)
#     model_path = f"baselines/{info.player_position}_weights.ckpt"
#     agent = DeepAgent(position=info.player_position, model_path=model_path)
#     print(agent.act(info))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,  # 开发模式下启用热重载
        workers=1
    )
