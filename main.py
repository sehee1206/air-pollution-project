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

df = load_data()

# 2. 데이터 전처리 및 탐색
st.header("1. 데이터 탐색 및 시각화")
selected_sido = st.selectbox("분석할 시도를 선택하세요:", df["시도"].unique())
filtered_df = df[df["시도"] == selected_sido]

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{selected_sido} 데이터 미리보기")
    st.dataframe(filtered_df.head(10))

with col2:
    st.subheader(f"{selected_sido} 배출원별 초미세먼지(PM-2_5) 배출량")
    grouped = filtered_df.groupby("배출원대분류")["초미세먼지(PM-2_5)"].sum().reset_index()
    fig = px.bar(grouped, x="배출원대분류", y="초미세먼지(PM-2_5)", color="배출원대분류")
    st.plotly_chart(fig, use_container_width=True)

# 3. 선형회귀 모델링 및 평가
st.header("2. 선형회귀(Linear Regression) 모델링 및 평가")
features = ['일산화탄소(CO)', '질소산화물(NOx)', '황산화물(SOx)', '총 부유입자(TSP)', '휘발성유기화합물(VOC)']
target = '초미세먼지(PM-2_5)'

# 결측치 제거
model_df = df[features + [target]].dropna()
X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

m1, m2 = st.columns(2)
m1.metric("결정계수 (R² Score)", f"{r2:.4f}")
m2.metric("평균제곱오차 (MSE)", f"{mse:.2f}")

# 4. 직접 수치 입력해서 예측해보기
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
