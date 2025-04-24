import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 페이지 설정
st.set_page_config(
    page_title="스마트 정수장",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💧"
)

st.title("💧 스마트 정수장 XGBoost: 약품 공정")

# 탭 UI
tab1, tab2 = st.tabs(["⚙️ 설정", "📊 결과 보기"])

with tab1:
    st.subheader("⚙️ 모델 설정")

    target = st.selectbox("■ 주요 target :", ("로그 응집제 주입률",))

    input_options = {
        "원수 탁도", "원수 pH", "원수 알칼리도", "원수 전기전도도",
        "원수 수온", "3단계 원수 유입 유량", "3단계 1계열 응집제 주입률",
        "3단계 침전지 탁도", "3단계 침전지 체류시간", "3단계 여과지 탁도"
    }

    selected_input = st.multiselect(
        "■ 입력 변수 선택 (로그 원수 탁도는 항상 포함):",
        options=sorted(input_options)
    )

    st.markdown("---")
    st.subheader("📈 XGBoost 파라미터")

    max_depth_1 = st.slider("max_depth:", 1, 20, 3)
    n_estimator1 = st.slider("n_estimators:", 10, 500, 100, step=10)
    learning_rate1 = st.slider("learning_rate:", 0.01, 1.00, 0.1, step=0.01)
    subsample1 = st.slider("subsample:", 0.1, 1.0, 0.8, step=0.1)

    # 실행 버튼 (탭2에서 사용을 위한 세션 상태 저장)
    if st.button("🚀 실행하기"):
        st.session_state.run_model = True
        st.session_state.target = target
        st.session_state.selected_input = selected_input
        st.session_state.params = {
            "max_depth": max_depth_1,
            "n_estimators": n_estimator1,
            "learning_rate": learning_rate1,
            "subsample": subsample1
        }

with tab2:
    st.subheader("📊 예측 결과")

    # 실행 버튼 눌렀을 때만 실행
    if st.session_state.get("run_model", False):
        target = st.session_state.target
        selected_input = st.session_state.selected_input
        params = st.session_state.params

        try:
            df = pd.read_csv("SN_total.csv", encoding='utf-8-sig')
        except FileNotFoundError:
            st.error("❌ 'SN_total.csv' 파일을 찾을 수 없습니다.")
            st.stop()

        if not selected_input:
            st.warning("⛔ 열을 하나 이상 선택해주세요.")
            st.stop()

        X = df[list(set(selected_input + ["로그 원수 탁도"]))]
        y = df[target]

        Xt, Xts, yt, yts = train_test_split(X, y, test_size=0.2, shuffle=False)

        xg_reg = XGBRegressor(
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            random_state=2
        )

        xg_reg.fit(Xt, yt)
        yt_pred = xg_reg.predict(Xt)
        yts_pred = xg_reg.predict(Xts)

        # 지표 출력
        mse_train = mean_squared_error(10**yt, 10**yt_pred)
        mse_test = mean_squared_error(10**yts, 10**yts_pred)
        r2_train = r2_score(10**yt, 10**yt_pred)
        r2_test = r2_score(10**yts, 10**yts_pred)

        st.write(f"✅ 학습 데이터 MSE: {mse_train:.4f}")
        st.write(f"✅ 테스트 데이터 MSE: {mse_test:.4f}")
        st.write(f"✅ 학습 데이터 R²: {r2_train:.4f}")
        st.write(f"✅ 테스트 데이터 R²: {r2_test:.4f}")

        # 시각화
        st.subheader("📈 예측 시각화")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(Xt["로그 원수 탁도"], yt, s=3, label="실제")
        ax.scatter(Xt["로그 원수 탁도"], yt_pred, s=3, label="예측", c="r")
        ax.grid()
        ax.legend(fontsize=10)
        ax.set_xlabel("로그 원수 탁도")
        ax.set_ylabel("로그 응집제 주입률")
        ax.set_title(f"Train MSE: {mse_train:.4f}, R²: {r2_train:.2f}")

        ax = axes[1]
        ax.scatter(Xts["로그 원수 탁도"], yts, s=3, label="실제")
        ax.scatter(Xts["로그 원수 탁도"], yts_pred, s=3, label="예측", c="r")
        ax.grid()
        ax.legend(fontsize=10)
        ax.set_xlabel("로그 원수 탁도")
        ax.set_ylabel("로그 응집제 주입률")
        ax.set_title(f"Test MSE: {mse_test:.4f}, R²: {r2_test:.2f}")

        st.pyplot(fig)

    else:
        st.warning("먼저 '설정' 탭에서 실행 버튼을 눌러주세요.")