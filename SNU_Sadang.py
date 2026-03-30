import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import neurokit2 as nk

# ================== 캐싱 설정 ==================
@st.cache_data
def read_watch_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'time_kst' in df.columns:
        df['time_kst'] = pd.to_datetime(df['time_kst'])
    cols_to_fix = [c for c in df.columns if c in ['x', 'y', 'z', 'bvp', 'eda', 'temperature', 'temp']]
    for c in cols_to_fix:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=cols_to_fix).reset_index(drop=True)

# ================== 1. 핵심 전처리 함수 ==================
def winsorize_signal(data, lower=1, upper=99):
    data = np.asarray(data, dtype=float)
    if len(data) == 0: return data
    low_val, high_val = np.percentile(data, [lower, upper])
    return np.clip(data, low_val, high_val)

def preprocess_common_signal(df, col_names, fs):
    if len(df) == 0: return df
    for c in col_names:
        df[c] = df[c].interpolate(method='linear', limit_direction='both').ffill().bfill()
        df[c] = df[c].rolling(window=int(fs * 3), min_periods=1, center=True).mean()
        df[c] = winsorize_signal(df[c], 1, 99)
    return df

# HRV는 60초 간격의 데이터가 필요하므로 유지
def extract_hrv_from_signal(values, fs, prefix):
    feats = {f"{prefix}_Amp_Mean_60s": 0.0, f"{prefix}_RMSSD_60s": 0.0}
    try:
        v = winsorize_signal(np.asarray(values, dtype=float), 1, 99)
        b, a = signal.butter(3, [0.5/(fs/2), 8.0/(fs/2)], btype="band")
        clean_v = signal.filtfilt(b, a, v)
        env = np.abs(signal.hilbert(clean_v))
        feats[f"{prefix}_Amp_Mean_60s"] = float(np.mean(env))
        peaks, _ = signal.find_peaks(clean_v, distance=fs/2.5)
        if len(peaks) > 3:
            rr = np.diff(peaks) / fs * 1000.0
            rr = rr[(rr > 300) & (rr < 1300)]
            if len(rr) > 2:
                feats[f"{prefix}_RMSSD_60s"] = float(np.sqrt(np.mean(np.diff(rr)**2)))
    except: pass
    return feats

# EDA 전처리용 함수 (extract_eda_60s)는 삭제되었습니다! 
# 아래 메인 로직에서 통째로 1초 단위로 처리합니다.

# ================== 3. 메인 UI ==================
st.set_page_config(page_title="센서 데이터 시각화", layout="wide")
st.title("📊 통합 센서 데이터 시각화 분석기")

if 'final_df' not in st.session_state:
    st.session_state.final_df = None

st.sidebar.header("⚙️ 데이터 업로드")
uploaded_files = st.sidebar.file_uploader("CSV 파일들 업로드", type=["csv"], accept_multiple_files=True)
label_f = st.sidebar.file_uploader("정답 레이블 파일 업로드", type=["csv", "xlsx"])

if st.sidebar.button("🚀 시각화 시작"):
    if uploaded_files:
        try:
            with st.status("데이터 분석 중...") as status:
                f_map = {f.name.upper(): f for f in uploaded_files}
                df_acc = read_watch_csv(next(f for k, f in f_map.items() if 'ACC' in k))
                df_bvp = read_watch_csv(next(f for k, f in f_map.items() if 'BVP' in k))
                df_eda = read_watch_csv(next(f for k, f in f_map.items() if 'EDA' in k))
                df_tmp = read_watch_csv(next(f for k, f in f_map.items() if 'TEMP' in k or 'TEMPERATURE' in k))

                # --- 1. ACC, BVP, TEMP 전처리 ---
                df_acc = preprocess_common_signal(df_acc, ['x', 'y', 'z'], 32.0)
                df_acc['mag'] = np.sqrt(df_acc['x']**2 + df_acc['y']**2 + df_acc['z']**2)
                df_bvp = preprocess_common_signal(df_bvp, ['bvp'], 64.0)
                temp_col = 'temp' if 'temp' in df_tmp.columns else 'temperature'
                df_tmp = preprocess_common_signal(df_tmp, [temp_col], 4.0)

                # --- 2. EDA 전처리: 전체 신호 통째로 분리 ---
                # 결측치 채우고 튀는 값 잡기
                df_eda['eda'] = df_eda['eda'].interpolate(method='linear', limit_direction='both').ffill().bfill()
                df_eda['eda'] = winsorize_signal(df_eda['eda'], 1, 99)
                
                # NeuroKit2로 전체 신호 클리닝 및 분리
                eda_clean = nk.eda_clean(df_eda['eda'].values, sampling_rate=4.0, method='neurokit')
                eda_decomposed = nk.eda_phasic(eda_clean, sampling_rate=4.0)
                
                # 분리된 Tonic과 Phasic을 원본 df_eda에 컬럼으로 추가
                df_eda['EDA_Tonic'] = eda_decomposed['EDA_Tonic']
                df_eda['EDA_Phasic'] = eda_decomposed['EDA_Phasic']

                # --- 3. 1초 단위로 시간 버림 ---
                for d in [df_acc, df_bvp, df_eda, df_tmp]: 
                    d['time_sec'] = d['time_kst'].dt.floor('1s')

                # --- 4. 1초 단위로 묶기(Groupby) ---
                agg_acc = df_acc.groupby('time_sec')['mag'].mean().rename('ACC_MAG_mean')
                agg_bvp = df_bvp.groupby('time_sec')['bvp'].mean().rename('BVP_BVP_mean')
                # EDA는 이제 Tonic과 Phasic을 각각 1초 평균 냅니다!
                agg_eda = df_eda.groupby('time_sec')[['EDA_Tonic', 'EDA_Phasic']].mean()
                agg_tmp = df_tmp.groupby('time_sec')[temp_col].mean().rename('TEMP_TEMP_mean')

                common_idx = agg_acc.index.intersection(agg_bvp.index).intersection(agg_eda.index)
                win_res = []
                
                # --- 5. HRV (BVP)만 60초 윈도우로 계산 ---
                for t in common_idx:
                    start_t = t - pd.Timedelta(seconds=60)
                    bvp_win = df_bvp[(df_bvp['time_sec'] > start_t) & (df_bvp['time_sec'] <= t)]['bvp']
                    hrv = extract_hrv_from_signal(bvp_win, 64.0, "BVP")
                    win_res.append({**hrv, 'time_sec': t})

                # 최종 병합
                final_df = pd.concat([agg_acc, agg_bvp, agg_eda, agg_tmp], axis=1).join(pd.DataFrame(win_res).set_index('time_sec')).reset_index()
                
                final_df['label'] = 0
                if label_f:
                    ldf = pd.read_csv(label_f) if label_f.name.endswith('csv') else pd.read_excel(label_f)
                    ldf['time_sec'] = pd.to_datetime(ldf['실제 날짜'].astype(str) + ' ' + ldf['실제 시각'].astype(str)).dt.floor('1s')
                    final_df.loc[final_df['time_sec'].isin(ldf['time_sec']), 'label'] = 1
                
                st.session_state.final_df = final_df
                status.update(label="✅ 1초 단위 처리 완료!", state="complete")
        except Exception as e: st.error(f"에러: {e}")

# ================== 4. 시각화 영역 ==================
if st.session_state.final_df is not None:
    vis_df = st.session_state.final_df.sort_values('time_sec').reset_index(drop=True)
    
    # 다운로드 버튼 추가
    st.subheader("💾 전처리 완료 데이터 다운로드")
    csv_data = vis_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 1초 단위 통합 데이터 CSV로 다운로드하기",
        data=csv_data,
        file_name="preprocessed_1sec_sensor_data.csv",
        mime="text/csv"
    )
    st.divider()
    
    # 60s 꼬리표를 떼고 1초 단위 컬럼명으로 변경
    plot_configs = {
        "BVP Pulse": ("BVP_BVP_mean", "#FF4B4B"),
        "HRV RMSSD": ("BVP_RMSSD_60s", "#B22222"), # HRV는 여전히 60초
        "EDA Tonic (1sec)": ("EDA_Tonic", "#FF8C00"),
        "EDA Phasic (1sec)": ("EDA_Phasic", "#FFA500"),
        "Movement": ("ACC_MAG_mean", "#28A745"),
        "Temperature": ("TEMP_TEMP_mean", "#FFD700")
    }
    
    selected = st.multiselect("지표 선택", list(plot_configs.keys()), default=list(plot_configs.keys()))
    
    if selected:
        fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True, subplot_titles=selected, vertical_spacing=0.05)
        
        # --- 음영 계산 로직 ---
        if 'label' in vis_df.columns and vis_df['label'].any():
            diffs = np.diff(vis_df['label'].astype(int), prepend=0, append=0)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            
            for s, e in zip(starts, ends):
                s_time = vis_df['time_sec'].iloc[s].isoformat()
                e_time = vis_df['time_sec'].iloc[min(e, len(vis_df)-1)].isoformat()
                for r in range(1, len(selected) + 1):
                    fig.add_shape(type="rect", x0=s_time, x1=e_time, y0=0, y1=1,
                                xref=f"x{r if r > 1 else ''}", yref=f"y{r if r > 1 else ''} domain",
                                fillcolor="gray", opacity=0.2, line_width=0, layer="below")

        for i, key in enumerate(selected, 1):
            col, color = plot_configs[key]
            if col in vis_df.columns:
                fig.add_trace(go.Scattergl(x=vis_df['time_sec'], y=vis_df[col], name=key, line=dict(color=color, width=1.5),
                                            hovertemplate='시간: %{x|%H:%M:%S}<br>값: %{y:.4f}<extra></extra>'), row=i, col=1)
                fig.update_xaxes(showticklabels=True, tickformat="%H:%M:%S", row=i, col=1)

        fig.update_layout(height=250*len(selected), hovermode="x unified", showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
