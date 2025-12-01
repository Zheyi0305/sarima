#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from scipy.stats import mstats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
from datetime import timedelta, datetime
from io import BytesIO, StringIO


# In[ ]:


#页面基础设置
st.title("📊 SARIMA 时间序列预测工具")
st.subheader("基于味精出口价格数据的自动预测")
st.divider()  # 分割线，让UI更整洁


# In[ ]:


#侧边栏用户交互设置
with st.sidebar:
    st.header("🔧 操作设置")
    # 1. 文件上传
    uploaded_file = st.file_uploader("上传 CSV 数据文件", type="csv")
    # 2. 预测步长设置
    forecast_steps = st.number_input(
        "预测未来步数（按月）", 
        min_value=1, max_value=24, value=6, step=1
    )

# 容错处理：未上传文件时提示
if not uploaded_file:
    st.info("请先在左侧侧边栏上传 CSV 数据文件（\n1. 需包含 'date' 日期列和 'data' 数据列 \n2. date列格式为标准 YYYY-MM（示例：2023-01、2025-12））")
    st.stop()  # 停止后续代码运行


# In[ ]:


#核心代码功能
@st.cache_data  # 缓存数据，避免重复运行（提升速度）
def load_and_preprocess_data(file):
    """加载并预处理数据"""
    data = pd.read_csv(file, parse_dates=['date'], index_col='date')
    # 确保价格列存在且为数值型
    if 'data' not in data.columns:
        st.error("CSV 文件中缺少 'data' 列（价格数据）")
        st.stop()
    data['data'] = pd.to_numeric(data['data'], errors='coerce').dropna()
    return data

def check_stationarity(series):
    """平稳性检验（ADF检验）"""
    result = adfuller(series)
    return result[0], result[1], result  # 返回ADF统计量、p值、完整结果
# ---------------------- 1. 数据加载与预处理 ----------------------
with st.spinner("正在加载数据..."):
    data = load_and_preprocess_data(uploaded_file)

# 显示数据预览
st.subheader("📈 数据预览")
st.dataframe(data.tail(10), use_container_width=True)  # 显示最后10行数据

# ---------------------- 2. 平稳性检验 ----------------------
st.subheader("📊 ADF 平稳性检验")
adf_statistic, p_value, adf_result = check_stationarity(data['data'])

# 用Streamlit组件展示结果（替代print）
col1, col2 = st.columns(2)
with col1:
    st.metric("ADF 统计量", f"{adf_statistic:.6f}")
with col2:
    st.metric("p-value", f"{p_value:.6f}")

# 结果解读
st.write("### 检验结果解读")
if adf_statistic < 0:
    st.success("✅ ADF统计量为负值，数据倾向于平稳")
else:
    st.warning("⚠️ ADF统计量为正值，数据倾向于非平稳")

if p_value < 0.01:
    st.write("• p-value < 0.01：数据极大可能是平稳的")
elif p_value < 0.05:
    st.write("• p-value < 0.05：数据很可能是平稳的")
elif p_value < 0.1:
    st.write("• p-value < 0.1：数据可能是平稳的")
else:
    st.write("• p-value ≥ 0.1：数据很可能是非平稳的")

st.divider()

# ---------------------- 3. 自动选择SARIMA参数 ----------------------
st.subheader("🔍 自动选择最佳模型参数")
with st.spinner("正在搜索最佳模型参数...（可能需要1-3分钟）"):
    warnings.filterwarnings('ignore')
    auto_model = auto_arima(
        data['data'],
        seasonal=True,
        m=12,  # 季节性周期（按月）
        trace=False,  # 关闭详细输出
        error_action='ignore',
        suppress_warnings=True,
        stepwise=False,
        n_jobs=-1
    )

# 展示最佳模型
best_order = auto_model.order
best_seasonal_order = auto_model.seasonal_order
st.code(f"最佳模型：{auto_model}\n非季节性参数：order={best_order}\n季节性参数：seasonal_order={best_seasonal_order}")

# 参数解读
with st.expander("📖 模型参数详细解读", expanded=False):
    p, d, q = best_order
    st.write(f"### 非季节性部分 ARIMA({p},{d},{q})")
    st.write(f"• AR({p})：{'当前值受前{p}期数值影响' if p>0 else '无自回归项'}")
    st.write(f"• I({d})：{'对数据进行了{d}次差分以消除趋势' if d>0 else '数据本身平稳，无需差分'}")
    st.write(f"• MA({q})：{'当前值受前{q}期预测误差影响' if q>0 else '无移动平均项'}")

    has_seasonal = any(x > 0 for x in best_seasonal_order[:3])
    if has_seasonal:
        P, D, Q, m = best_seasonal_order
        st.write(f"\n### 季节性部分 SARIMA({P},{D},{Q})")
        st.write(f"• 季节性AR({P})：受前{P}个季节性周期同期值影响")
        st.write(f"• 季节性差分({D})：{D}次季节性差分消除季节性趋势")
        st.write(f"• 季节性MA({Q})：受前{Q}个季节性周期预测误差影响")
    else:
        st.write("\n### 季节性部分")
        st.write("• 未检测到显著季节性模式")

st.divider()

# ---------------------- 4. 模型拟合与评估 ----------------------
st.subheader("📊 模型拟合结果评估")
with st.spinner("正在拟合模型..."):
    model = SARIMAX(
        data['data'],
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit()

# 展示拟合指标
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("AIC", f"{model_fit.aic:.2f}")
with col2:
    st.metric("BIC", f"{model_fit.bic:.2f}")

# 参数显著性
pvalues = model_fit.pvalues.dropna()
significant_params = sum(pvalues < 0.05)
total_params = len(pvalues)
with col3:
    st.metric("显著参数占比", f"{significant_params}/{total_params}")

# 拟合效果判断
if model_fit.aic < 100 and significant_params > total_params * 0.5:  # 调整判断阈值，更贴合实际
    st.success("✅ 模型拟合效果良好")
else:
    st.warning("⚠️ 模型拟合效果一般，可尝试调整数据或参数")

st.divider()

# ---------------------- 5. 预测与可视化 ----------------------
st.subheader("🚀 预测结果展示")
with st.spinner("正在预测未来数据..."):
    # 预测
    forecast_result = model_fit.get_forecast(steps=forecast_steps)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # 生成未来日期（按月初）
    last_date = pd.to_datetime(data.index[-1])
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_steps,
        freq='MS'  # 按月初生成日期
    )

    # 处理拟合值（清理无效值）
    fitted_values = model_fit.fittedvalues
    first_valid_idx = next(
        (idx for idx, val in fitted_values.items() if not pd.isna(val) and val != 0),
        None
    )
    if first_valid_idx:
        fitted_values = fitted_values.loc[first_valid_idx:]
    else:
        fitted_values = fitted_values.dropna()

# 可视化（适配Streamlit展示）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制历史数据、拟合数据、预测数据
ax.plot(data.index, data['data'], 'b-', label='历史数据', linewidth=1.5)
ax.plot(fitted_values.index, fitted_values, 'r-', label='模型拟合数据', linewidth=2)
ax.plot(future_dates, forecast, 'r--', label=f'未来{forecast_steps}步预测', linewidth=2.5, markersize=6)

# 连接拟合与预测数据
if len(fitted_values) > 0:
    ax.plot([fitted_values.index[-1], future_dates[0]],
            [fitted_values.iloc[-1], forecast.iloc[0]],
            'r--', linewidth=2)

# 绘制置信区间
ax.fill_between(future_dates,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='pink', alpha=0.3, label='95%置信区间')

# 预测起点标记
ax.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
ax.text(last_date + timedelta(days=10), data['data'].iloc[-1], '预测起点', fontsize=11)

# 图表美化
ax.set_title('SARIMA 时间序列预测结果', fontsize=14, pad=20)
#ax.set_xlabel('日期', fontsize=12)#行标签
#ax.set_ylabel('预测量', fontsize=12)#列标签
ax.legend(loc='best', fontsize=10)#图例
#ax.grid(True, alpha=0.3)#网格线

# 在Streamlit中展示图表（替代plt.show()）
st.pyplot(fig, use_container_width=True)

# ---------------------- 6. 预测结果表格与下载 ----------------------
st.subheader("📋 预测结果详情")
# 构建预测结果DataFrame
forecast_df = pd.DataFrame({
    '预测日期': future_dates.strftime('%Y-%m-%d'),
    '预测价格': forecast.values.round(2),
    '95%置信区间下限': conf_int.iloc[:, 0].values.round(2),
    '95%置信区间上限': conf_int.iloc[:, 1].values.round(2)
})

# 展示表格
st.dataframe(forecast_df, use_container_width=True)

# 下载功能（CSV + 图表）
st.subheader("💾 结果下载")
col1, col2 = st.columns(2)

# 下载预测结果CSV
with col1:
    csv_buffer = StringIO()
    forecast_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    st.download_button(
        label="下载预测结果 CSV",
        data=csv_buffer.getvalue(),
        file_name=f"SARIMA预测结果_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# 下载可视化图表
with col2:
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    st.download_button(
        label="下载预测图表 PNG",
        data=img_buffer,
        file_name=f"SARIMA预测图表_{datetime.now().strftime('%Y%m%d')}.png",
        mime="image/png"
    )

