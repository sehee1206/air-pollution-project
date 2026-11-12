import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="국가 대기오염물질 분석 및 예측", layout="wide")
st.title("🌫️ 대한민국 지역별 대기오염물질 배출 현황 및 초미세먼지 예측")

# 1. 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", encoding="utf-8")
    return df

raw_df = load_data()

# ==========================================
# 2. 데이터 전처리 및 탐색 (EDA) - 대폭 강화!
# ==========================================
st.header("1. 데이터 전처리 및 탐색 (EDA)")

# 2-1. 결측치 확인 및 전처리 과정 보여주기
st.subheader("📌 ① 데이터 전처리: 결측치(Null) 확인 및 정제")
features = ['일산화탄소(CO)', '질소산화물(NOx)', '황산화물(SOx)', '총 부유입자(TSP)', '휘발성유기화합물(VOC)']
target = '초미세먼지(PM-2_5)'
analysis_cols = features + [target]

# 결측치 제거 전후 비교
clean_df = raw_df.dropna(subset=analysis_cols).copy()

col_pre1, col_pre2, col_pre3 = st.columns(3)
col_pre1.metric("전처리 전 전체 행 개수", f"{len(raw_df):,} 개")
col_pre2.metric("결측치 제거 후 행 개수", f"{len(clean_df):,} 개")
col_pre3.metric("제거된 불완전 데이터", f"{len(raw_df) - len(clean_df):,} 개")

with st.expander("🔍 컬럼별 결측치 개수 및 기초 통계량(평균/최댓값 등) 자세히 보기"):
    c_null, c_desc = st.columns(2)
    with c_null:
        st.write("**[전처리 전] 컬럼별 결측치 개수**")
        null_df = pd.DataFrame(raw_df[analysis_cols].isnull().sum(), columns=["결측치 수"])
        st.dataframe(null_df, use_container_width=True)
    with c_desc:
        st.write("**[전처리 후] 주요 오염물질 기초 통계량**")
        st.dataframe(clean_df[analysis_cols].describe().round(2), use_container_width=True)

st.divider()

# 2-2. 탐색적 데이터 분석 (시각화 3종)
st.subheader("📊 ② 탐색적 데이터 분석 (시각화)")

tab1, tab2, tab3 = st.tabs(["지역별/배출원별 분석", "전국 시도별 배출량 순위", "오염물질 간 상관관계 히트맵"])

with tab1:
    selected_sido = st.selectbox("분석할 시도를 선택하세요:", clean_df["시도"].unique())
    filtered_df = clean_df[clean_df["시도"] == selected_sido]
    
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**{selected_sido} 정제 데이터 미리보기**")
        st.dataframe(filtered_df[['시군구', '배출원대분류', '연료대분류'] + analysis_cols].head(10), use_container_width=True)
    with c2:
        st.write(f"**{selected_sido} 배출원대분류별 초미세먼지 배출량**")
        grouped = filtered_df.groupby("배출원대분류")["초미세먼지(PM-2_5)"].sum().reset_index()
        fig_bar = px.bar(grouped, x="배출원대분류", y="초미세먼지(PM-2_5)", color="배출원대분류", text_auto='.2s')
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.write("**전국 시도별 초미세먼지(PM-2_5) 총 배출량 비교**")
    sido_group = clean_df.groupby("시도")["초미세먼지(PM-2_5)"].sum().sort_values(ascending=False).reset_index()
    fig_sido = px.bar(sido_group, x="시도", y="초미세먼지(PM-2_5)", color="초미세먼지(PM-2_5)", color_continuous_scale="Reds")
    st.plotly_chart(fig_sido, use_container_width=True)

with tab3:
    st.write("**주요 대기오염물질 간 상관관계 히트맵 (독립변수 선정 근거)**")
    st.caption("1에 가까울수록(진한 빨간색) 두 물질이 함께 증가하는 강한 양의 상관관계를 의미합니다.")
    corr_matrix = clean_df[analysis_cols].corr().round(2)
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# ==========================================
# 3. 선형회귀 모델링 및 평가
# ==========================================
st.header("2. 선형회귀(Linear Regression) 모델링 및 평가")

X = clean_df[features]
y = clean_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

m1, m2 = st.columns(2)
m1.metric("결정계수 (R² Score)", f"{r2:.4f}")
m2.metric("평균제곱오차 (MSE)", f"{mse:.2f}")

# ==========================================
# 4. 예측 시뮬레이터
# ==========================================
st.header("3. 초미세먼지 배출량 예측 시뮬레이터")
st.write("각 오염물질 배출량을 입력하면 초미세먼지(PM-2_5) 배출량을 예측합니다.")

c1, c2, c3, c4, c5 = st.columns(5)
co = c1.number_input("일산화탄소(CO)", value=100.0)
nox = c2.number_input("질소산화물(NOx)", value=100.0)
sox = c3.number_input("황산화물(SOx)", value=50.0)
tsp = c4.number_input("총 부유입자(TSP)", value=30.0)
voc = c5.number_input("휘발성유기화합물(VOC)", value=50.0)

if st.button("예측 결과 확인"):
    pred_val = model.predict([[co, nox, sox, tsp, voc]])[0]
    st.success(f"예상 초미세먼지(PM-2_5) 배출량: **{pred_val:.2f}**")
