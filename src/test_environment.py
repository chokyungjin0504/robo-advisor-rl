from stable_baselines3.common.env_checker import check_env
import pandas as pd
import numpy as np

from environment import PortfolioEnv    # 방금 만든 환경 불러오기


# ---------- 테스트용 가짜 데이터 만들기 ----------
n_days = 500
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

# ---------- 1) Gymnasium 규격 준수 확인 ----------
env = PortfolioEnv(fake_prices, fake_features, reward_type="simple")
check_env(env)
print("환경 규격 검증 통과")

# ---------- 2) 랜덤 에이전트로 한 에피소드 돌려보기 ----------
obs, info = env.reset(seed=42)
total_reward = 0.0

for step in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 50 == 0:
        print(f"스텝 {step:3d} | "
              f"자산가치 {info['portfolio_value']:,.0f} | "
              f"MDD {info['mdd']:.2%} | "
              f"비중합 {info['weights'].sum():.4f}")

    if terminated:
        print(f"스텝 {step}: MDD 한계 초과로 조기 종료")
        break
    if truncated:
        print(f"스텝 {step}: 데이터 종료")
        break

print(f"누적 보상: {total_reward:.6f}")