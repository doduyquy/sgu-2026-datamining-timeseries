import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="BTC Predictive Dashboard", layout="wide")
st.title("Phân Tích Thực Chiến Dự Báo BTC (Institutional XGBoost)")

# Đường dẫn tới thư mục Models nãy mình vừa lưu
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL_PATH = os.path.join(DIR_PATH, 'Models', 'xgboost_btc_model.pkl')
MODEL_PKL_ALIAS_PATH = os.path.join(DIR_PATH, 'Models', 'xgboost.pkl')
MODEL_JSON_PATH = os.path.join(DIR_PATH, 'Models', 'xgboost_btc_model.json')
BUFFER_PATH = os.path.join(DIR_PATH, 'Models', 'live_buffer_data.csv')
ABNORMAL_THRESHOLD = 0.1

# 1. LOAD MODEL & LỊCH SỬ BUFFER (Hàm cache giúp web không load lại từ đầu)
def load_assets():
    if not os.path.exists(BUFFER_PATH):
        st.error("⚠️ Chưa tìm thấy file Model hoặc Baseline Data! Bạn cần chạy Cell cuối cùng trong Notebook XgBoost.ipynb để xuất dữ liệu ra.")
        st.stop()

    if not (os.path.exists(MODEL_JSON_PATH) or os.path.exists(MODEL_PKL_PATH) or os.path.exists(MODEL_PKL_ALIAS_PATH)):
        st.error("⚠️ Chưa tìm thấy file model. Hãy chạy lại cell export ở Notebook để tạo model trong thư mục time-series/Models.")
        st.stop()

    # Load 500 nến cuối cùng từ lịch sử notebook
    buffer_df = pd.read_csv(BUFFER_PATH, index_col=0, parse_dates=True)
    return buffer_df


def load_model_from_path(model_path):
    if model_path.endswith('.json'):
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    return joblib.load(model_path)


def get_expected_features(model):
    if isinstance(model, xgb.Booster):
        return model.feature_names
    return list(model.feature_names_in_)


def raw_predict_log_return(model, X_row):
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X_row, feature_names=X_row.columns.tolist())
        return float(model.predict(dmat)[0])
    return float(model.predict(X_row)[0])


def predict_log_return(model, X_row):
    pred = raw_predict_log_return(model, X_row)

    # Fallback an toàn: nếu model trả về giá trị phi lý thì dùng log_return gần nhất từ dữ liệu thật.
    last_known_log_return = float(X_row['log_return'].iloc[0]) if 'log_return' in X_row.columns else 0.0
    safe_fallback = float(np.clip(last_known_log_return, -0.02, 0.02))

    # Log-return theo giờ BTC thường rất nhỏ; |pred| > 0.1 gần như chắc chắn do model/path sai.
    if (not np.isfinite(pred)) or abs(pred) > ABNORMAL_THRESHOLD:
        return safe_fallback, True, pred
    return pred, False, pred

buffer_df = load_assets()

# 2. HÀM TẠO FEATURES (Copy chuẩn 100% logic từ Notebook)
def create_features(data, is_train=False):
    df_feat = data.copy()
    # Tính log return
    df_feat['log_return'] = np.log(df_feat['price'] / df_feat['price'].shift(1))
    
    # Lag & MA
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]
    for i in lag_hours: df_feat[f'return_lag_{i}h'] = df_feat['log_return'].shift(i)
    for w in [5, 10, 20, 50]: df_feat[f'return_ma_{w}'] = df_feat['log_return'].rolling(w).mean()
    for w in [5, 10, 20, 50]: df_feat[f'return_std_{w}'] = df_feat['log_return'].rolling(w).std()
    
    df_feat['momentum_10'] = df_feat['log_return'] - df_feat['return_ma_10']
    
    # RSI & MACD trên return
    ema_12 = df_feat['log_return'].ewm(span=12, adjust=False).mean()
    ema_26 = df_feat['log_return'].ewm(span=26, adjust=False).mean()
    df_feat['return_macd'] = ema_12 - ema_26
    
    delta = df_feat['log_return'].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df_feat['return_rsi_14'] = 100 - (100 / (1 + (gain / loss)))
    
    # Date features
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_month'] = df_feat.index.day
    
    if is_train:
        df_feat['target_log_return'] = df_feat['log_return'].shift(-1)
        df_feat = df_feat.dropna()
    else:
        df_feat = df_feat.dropna()
        
    return df_feat


def probe_model_health(model, probe_df):
    probe_feat = create_features(probe_df.iloc[-250:], is_train=False).iloc[-48:]
    feats = get_expected_features(model)
    raw_preds = []
    abnormal_count = 0

    for i in range(len(probe_feat)):
        X_probe = probe_feat.drop(['price', 'target_log_return'], axis=1, errors='ignore')
        X_probe = X_probe[feats].iloc[[i]]
        raw_pred = raw_predict_log_return(model, X_probe)
        raw_preds.append(raw_pred)
        if (not np.isfinite(raw_pred)) or abs(raw_pred) > ABNORMAL_THRESHOLD:
            abnormal_count += 1

    return {
        'abnormal_count': abnormal_count,
        'total': len(raw_preds),
        'pred_min': float(np.min(raw_preds)) if raw_preds else np.nan,
        'pred_max': float(np.max(raw_preds)) if raw_preds else np.nan,
        'pred_mean': float(np.mean(raw_preds)) if raw_preds else np.nan,
    }


candidate_model_paths = [p for p in [MODEL_JSON_PATH, MODEL_PKL_PATH, MODEL_PKL_ALIAS_PATH] if os.path.exists(p)]
model = None
model_path_used = None
model_probe = None

for candidate_path in candidate_model_paths:
    candidate_model = load_model_from_path(candidate_path)
    candidate_probe = probe_model_health(candidate_model, buffer_df[['price']])
    if candidate_probe['abnormal_count'] < candidate_probe['total']:
        model = candidate_model
        model_path_used = candidate_path
        model_probe = candidate_probe
        break

if model is None:
    # Nếu tất cả model đều lỗi, vẫn chọn file ưu tiên cao nhất để app chạy với fallback.
    model_path_used = candidate_model_paths[0]
    model = load_model_from_path(model_path_used)
    model_probe = probe_model_health(model, buffer_df[['price']])

expected_features = get_expected_features(model)

# 3. LẤY DỮ LIỆU REAL-TIME TỪ YAHOO FINANCE
st.sidebar.header("Bảng Điều Khiển")

# Giữ dữ liệu đang xem qua các lần rerun (tránh reset về buffer khi bấm nút dự báo)
if 'full_df' not in st.session_state:
    st.session_state.full_df = buffer_df[['price']].copy()
    st.session_state.data_source_label = "Buffer Cũ"

if st.sidebar.button("Kéo Dữ Liệu Live Từ Yahoo Finance"):
    with st.spinner('Đang tải dữ liệu trực tiếp...'):
        live_data = yf.download(tickers='BTC-USD', period='5d', interval='1h')
        
        # Xử lý MultiIndex nếu bị yfinance trả về
        if hasattr(live_data.columns, 'levels'):
            close_prices = live_data['Close'].values.flatten()
        else:
            close_prices = live_data['Close'].values
            
        live_df = pd.DataFrame({'price': close_prices}, index=live_data.index)
        
        # Trừ hao múi giờ nếu có
        if getattr(live_df.index, 'tz', None) is not None:
            live_df.index = live_df.index.tz_localize(None)
        
        # Gộp Buffer cũ và Live data mới, loại bỏ trùng lặp thời gian
        merged_df = pd.concat([buffer_df[['price']], live_df])
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')].sort_index()
        st.session_state.full_df = merged_df[['price']].copy()
        st.session_state.data_source_label = "Live Update Nhất"

full_df = st.session_state.full_df.copy()

st.info("Bạn đang xem dữ liệu: **" + st.session_state.data_source_label + "**")
st.caption(f"Model đang dùng: {model_path_used}")
st.caption(
    "Probe model (48 điểm gần nhất) | "
    f"abnormal: {model_probe['abnormal_count']}/{model_probe['total']} | "
    f"pred min/max/mean: {model_probe['pred_min']:.6f} / {model_probe['pred_max']:.6f} / {model_probe['pred_mean']:.6f}"
)

# Hiển thị số liệu mới nhất
last_time = full_df.index[-1]
last_price = full_df['price'].iloc[-1]
st.write(f"### Giá BTC hiện tại (Cập nhật lúc {last_time.strftime('%Y-%m-%d %H:%M:%S')}): **${last_price:,.2f}**")

# === BÁO CÁO BACKTEST (THỰC TẾ VS DỰ BÁO LỊCH SỬ) ===
st.markdown("---")
st.subheader("🔍 So Sánh Thực Tế vs Dự Báo (Lịch Sử Gần Nhất)")
with st.expander("Bấm để xem lịch sử mô hình bám sát Giá Thực Tế trong 48 giờ qua thế nào"):
    with st.spinner("Đang trích xuất backtest..."):
        # Lấy 250 nến cuối để có đà cho 48 nến test
        df_eval_feat = create_features(full_df.iloc[-250:], is_train=False)
        
        # Chỉ lấy 48 giờ cuối
        df_eval_feat = df_eval_feat.iloc[-48:]
        
        eval_preds = []
        eval_actuals = []
        eval_dates = []
        eval_fallback_count = 0
        
        for i in range(len(df_eval_feat)):
            X_eval = df_eval_feat.drop(['price', 'target_log_return'], axis=1, errors='ignore')
            X_eval = X_eval[expected_features].iloc[[i]]
            pred_log, used_fallback, raw_pred = predict_log_return(model, X_eval)
            if used_fallback:
                eval_fallback_count += 1
            
            # Phục hồi giá gốc từ Price của bước trước
            curr_date = df_eval_feat.index[i]
            idx_pos = full_df.index.get_loc(curr_date)
            # Giá thực tế ở bước t-1
            prev_price = full_df['price'].iloc[idx_pos - 1]
                
            pred_p = prev_price * np.exp(pred_log)
            
            eval_preds.append(pred_p)
            eval_actuals.append(df_eval_feat['price'].iloc[i])
            eval_dates.append(curr_date)

        eval_preds_arr = np.array(eval_preds)
        eval_actuals_arr = np.array(eval_actuals)
        eval_rmse = np.sqrt(np.mean((eval_actuals_arr - eval_preds_arr) ** 2))
        eval_mae = np.mean(np.abs(eval_actuals_arr - eval_preds_arr))

        c1, c2 = st.columns(2)
        c1.metric("RMSE (Backtest 48h)", f"${eval_rmse:,.2f}")
        c2.metric("MAE (Backtest 48h)", f"${eval_mae:,.2f}")
            
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=eval_dates, y=eval_actuals, mode='lines+markers', name='Thực tế', line=dict(color='royalblue')))
        fig_eval.add_trace(go.Scatter(x=eval_dates, y=eval_preds, mode='lines+markers', name='Dự báo (1-step trước đó)', line=dict(color='darkorange', dash='dash')))
        fig_eval.update_layout(title="Mô phỏng Dự Báo t+1 so với Tín Hiệu Thực Tế (48 Giờ Qua)", template="plotly_dark", height=450, xaxis_title="Thời gian (Giờ)", yaxis_title="Mệnh Giá (USD)")
        st.plotly_chart(fig_eval, use_container_width=True)

        if eval_fallback_count > 0:
            st.warning(
                f"⚠️ Model đã trả về giá trị log_return bất thường {eval_fallback_count} lần trong phần backtest. "
                "App đang dùng fallback an toàn nên không bị crash. Hãy retrain và export lại model từ notebook."
            )

# 4. CHẠY PREDICTION ĐA BƯỚC (MULTI-STEP FORECASTING)
st.markdown("---")
forecast_hours = st.slider("⏳ Chọn số giờ muốn dự báo tới tương lai:", min_value=1, max_value=24, value=8)

if st.button(f"BẤM DỰ BÁO {forecast_hours} GIỜ TIẾP THEO", type="primary"):
    with st.spinner(f'Đang chạy vòng lặp Autoregressive cho {forecast_hours} giờ...'):
        
        # Tạo 1 bản sao tạm thời để "mớm" dần giá dự báo vào
        temp_df = full_df[['price']].copy()
        
        pred_times = []
        pred_prices = []
        future_fallback_count = 0
        
        for step in range(forecast_hours):
            # Tính lại Features trên dữ liệu đã gộp thêm dự báo của các bước trước
            df_feat = create_features(temp_df.iloc[-500:], is_train=False)
            
            drop_cols = ['price']
            if 'target_log_return' in df_feat.columns: drop_cols.append('target_log_return')
                
            X_step = df_feat.drop(drop_cols, axis=1, errors='ignore')
            X_step = X_step[expected_features].iloc[[-1]]
            
            # Dự báo log return cho t+1 kế tiếp
            pred_log_ret, used_fallback, raw_pred = predict_log_return(model, X_step)
            if used_fallback:
                future_fallback_count += 1
            
            # Giải mã ra Giá USD mới
            current_price = temp_df['price'].iloc[-1]
            next_price = current_price * np.exp(pred_log_ret)
            
            # Tạo index thời gian mới
            next_time = temp_df.index[-1] + timedelta(hours=1)
            
            # GHI NHẬN LẠI (Nối dự báo vào mạch dữ liệu để tính cho vòng lặp sau)
            new_row = pd.DataFrame({'price': [next_price]}, index=[next_time])
            temp_df = pd.concat([temp_df, new_row])
            
            pred_times.append(next_time)
            pred_prices.append(next_price)
            
        # TỔNG KẾT KẾT QUẢ CUỐI CÙNG
        final_pred_price = pred_prices[-1]
        diff = final_pred_price - last_price
        trend = "TĂNG 🟩" if diff > 0 else "GIẢM 🟥"
        
        st.success(f"###Đích đến sau {forecast_hours} giờ ({pred_times[-1].strftime('%H:%M')}): **${final_pred_price:,.2f}** ({trend})")
        st.metric(label=f"Tổng chênh lệch sau {forecast_hours}h", value=f"${diff:,.2f}", delta=f"{(final_pred_price/last_price - 1)*100:.2f}%")
        
        # 5. VẼ BIỂU ĐỒ BẰNG PLOTLY (Hiển thị nguyên đường ray)
        fig = go.Figure()
        
        # Plot 48h thực tế
        plot_df = full_df.iloc[-48:]
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['price'], 
            mode='lines+markers', name='Thực tế (48h qua)', 
            line=dict(color='royalblue')
        ))
        
        # Vẽ đường nối dự báo (Bao gồm điểm thực tế cuối cùng và toàn bộ mảng dự báo)
        connect_x = [plot_df.index[-1]] + pred_times
        connect_y = [plot_df['price'].iloc[-1]] + pred_prices
        
        fig.add_trace(go.Scatter(
            x=connect_x, y=connect_y, 
            mode='lines+markers', name=f'Quỹ đạo tương lai ({forecast_hours}h)', 
            line=dict(color='darkorange', dash='dash', width=3),
            marker=dict(size=8, symbol='diamond')
        ))
        
        fig.update_layout(
            title=f"Biểu Đồ Hành Vi Giá - Dự Kiến Quỹ Đạo {forecast_hours} Giờ", 
            xaxis_title="Thời gian (Giờ)", 
            yaxis_title="Mệnh giá (USD)",
            height=600,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        if future_fallback_count > 0:
            st.warning(
                f"Model trả về giá trị log_return bất thường {future_fallback_count}/{forecast_hours} bước dự báo tương lai. "
                "App đã tự fallback để giữ biểu đồ ổn định. Bạn nên chạy lại cell huấn luyện + export model trong notebook."
            )
