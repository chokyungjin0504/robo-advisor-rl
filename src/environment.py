import gymnasium as gym                    # 강화학습 환경 표준 인터페이스
from gymnasium import spaces               # 관측/행동 공간 정의용
import numpy as np                         # 수치 계산
import pandas as pd                        # 데이터프레임 처리


class PortfolioEnv(gym.Env):
    """포트폴리오 자산배분 강화학습 환경.

    에이전트는 매일 자산별 목표 비중을 결정하고,
    그 결과로 발생한 수익률을 보상으로 받는다.
    """

    def __init__(
        self,
        prices: pd.DataFrame,              # 자산별 수정주가 (행=날짜, 열=자산)
        features: pd.DataFrame,            # 정규화된 피처 (행=날짜, 열=피처)
        window: int = 20,                  # 관측 윈도우 (명세 권장: 20~60 거래일)
        fee_rate: float = 0.00015,         # 거래 수수료 0.015% (명세 지정값)
        slippage: float = 0.0005,          # 슬리피지 0.05% (명세 지정값)
        reward_type: str = "simple",       # 보상 함수 종류: simple / sharpe / mdd
        mdd_lambda: float = 1.0,           # 변형 3의 MDD 페널티 강도
        mdd_limit: float = 0.15,           # Safe-Guard: MDD 15% 초과 시 조기 종료
        initial_cash: float = 1_000_000.0, # 초기 자본금
    ):
        super().__init__()                 # gym.Env 초기화 (필수)

        # ---------- 데이터 저장 ----------
        self.prices = prices.values                    # DataFrame → numpy (속도 향상)
        self.features = features.values                # 피처도 numpy로 변환
        self.n_assets = prices.shape[1]                 # 자산 개수 (예: 5)
        self.n_features = features.shape[1]             # 피처 개수 (예: 25)

        # ---------- 하이퍼파라미터 저장 ----------
        self.window = window
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.reward_type = reward_type
        self.mdd_lambda = mdd_lambda
        self.mdd_limit = mdd_limit
        self.initial_cash = initial_cash

        # ---------- 일간 수익률 미리 계산 ----------
        self.returns = np.diff(np.log(self.prices), axis=0)   # (T-1, n_assets)
        self.returns = np.vstack([np.zeros(self.n_assets), self.returns])  # 첫 행 0 추가

        # ---------- 행동 공간 정의 ----------
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # ---------- 관측 공간 정의 ----------
        obs_dim = self.n_features + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ---------- 에피소드 상태 변수 (reset에서 초기화) ----------
        self.current_step = None
        self.weights = None
        self.portfolio_value = None
        self.peak_value = None
        self.return_history = None

    def reset(self, seed=None, options=None):
        """에피소드를 처음 상태로 되돌린다."""
        super().reset(seed=seed)

        self.current_step = self.window
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.return_history = []

        return self._get_observation(), {}

    def step(self, action):
        """에이전트의 행동을 받아 하루를 진행한다."""

        target_weights = self._softmax(action)

        turnover = np.abs(target_weights - self.weights).sum()
        cost = turnover * (self.fee_rate + self.slippage)

        self.current_step += 1
        daily_returns = self.returns[self.current_step]
        portfolio_return = np.dot(target_weights, daily_returns)
        net_return = portfolio_return - cost

        self.portfolio_value *= (1.0 + net_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_mdd = (self.peak_value - self.portfolio_value) / self.peak_value

        self.weights = target_weights
        self.return_history.append(net_return)

        reward = self._compute_reward(net_return, current_mdd)

        terminated = bool(current_mdd > self.mdd_limit)
        truncated = bool(self.current_step >= len(self.returns) - 1)

        info = {
            "portfolio_value": self.portfolio_value,
            "mdd": current_mdd,
            "net_return": net_return,
            "weights": self.weights.copy(),
            "turnover": turnover,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """현재 시점의 관측값을 만든다."""
        market_features = self.features[self.current_step]
        obs = np.concatenate([market_features, self.weights])
        return obs.astype(np.float32)

    def _compute_reward(self, net_return, current_mdd):
        """보상 함수 3종 — 명세 요구사항."""

        if self.reward_type == "simple":
            return float(net_return)

        elif self.reward_type == "sharpe":
            if len(self.return_history) < 2:
                return float(net_return)
            sigma = np.std(self.return_history[-20:])
            return float(net_return / (sigma + 1e-8))

        elif self.reward_type == "mdd":
            return float(net_return - self.mdd_lambda * current_mdd)

        else:
            raise ValueError(f"알 수 없는 보상 함수: {self.reward_type}")

    @staticmethod
    def _softmax(x):
        """PPO가 출력한 임의의 실수를 합=1인 비중으로 변환."""
        x = np.clip(x, -10, 10)
        e_x = np.exp(x - np.max(x))
        return (e_x / e_x.sum()).astype(np.float32)