from stable_baselines3 import PPO                          # PPO 알고리즘
from stable_baselines3.common.vec_env import DummyVecEnv   # 벡터 환경 래퍼
from stable_baselines3.common.monitor import Monitor       # 에피소드 보상 기록
import pandas as pd
import numpy as np

from environment import PortfolioEnv    # 우리가 만든 환경


# ---------- 테스트용 가짜 데이터 (팀원 데이터 오면 이 부분만 교체) ----------
n_days = 1000                                               # 학습엔 데이터가 좀 더 필요해서 늘림
n_assets = 5
n_features = 25

dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

fake_prices = pd.DataFrame(
    100 * np.exp(np.cumsum(np.random.randn(n_days, n_assets) * 0.01, axis=0)),
    index=dates,
    columns=["SPY", "QQQ", "IEF", "GLD", "BIL"],
)

fake_features = pd.DataFrame(
    np.random.randn(n_days, n_features),
    index=dates,
)

# ---------- 환경 준비 ----------
env = PortfolioEnv(fake_prices, fake_features, reward_type="simple")  # 보상함수: 단순 수익률
env = Monitor(env)                          # 에피소드 보상을 기록해주는 래퍼 (학습곡선용)
env = DummyVecEnv([lambda: env])            # SB3는 벡터 환경 형태를 요구함

# ---------- PPO 모델 생성 ----------
model = PPO(
    policy="MlpPolicy",                     # 다층 퍼셉트론(기본 신경망) 정책
    env=env,
    learning_rate=3e-4,                     # 학습률 (PPO의 일반적인 기본값)
    n_steps=2048,                           # 한 번 업데이트하기 전에 모을 경험 개수
    batch_size=64,                          # 미니배치 크기
    n_epochs=10,                            # 모은 데이터를 몇 번 재사용할지
    gamma=0.99,                             # 할인율 (미래 보상을 얼마나 중요하게 볼지)
    clip_range=0.2,                         # PPO의 핵심: 정책을 한 번에 너무 크게 안 바꾸게 제한
    seed=42,                                # 재현성을 위한 시드 고정
    verbose=1,                              # 학습 중 로그 출력
    tensorboard_log="../logs/ppo_simple",   # 학습 로그 저장 경로
)

# ---------- 학습 실행 ----------
print("학습 시작...")
model.learn(total_timesteps=100_000)        # 명세 요구사항: 최소 100,000 스텝
print("학습 완료")

# ---------- 모델 저장 ----------
model.save("../models/ppo_reward_simple")
print("모델 저장 완료: models/ppo_reward_simple.zip")