import re

def evaluate_distance_shift(horse_name, target_dist, trainer_name, df_hist):
    """
    量化途程轉換的影響，回傳一個乘數 (Multiplier)。
    預設為 1.0 (中性無影響)。
    """
    if df_hist.empty: return 1.0
    
    # 找出該馬匹的歷史紀錄
    name_cols = [c for c in df_hist.columns if 'name' in c.lower() or '馬名' in c or '馬匹' in c]
    if not name_cols: return 1.0
    
    h_df = df_hist[df_hist[name_cols[0]] == horse_name].head(3) # 取近 3 仗
    if len(h_df) == 0: return 1.0
    
    # 取得上仗途程
    last_run = h_df.iloc[0]
    last_dist_str = str(last_run.get('Distance', '')).replace('米', '').strip()
    if not last_dist_str.isdigit(): return 1.0
    last_dist = int(last_dist_str)
    
    # 計算途程變化
    target_dist = int(str(target_dist).replace('米', '').strip())
    dist_diff = target_dist - last_dist
    
    if abs(dist_diff) < 200:
        return 1.0 # 變化不大，視為同程
        
    # 解析上仗走位 (Running_Pos)
    running_pos_str = str(last_run.get('Running_Pos', ''))
    pos_list = [int(p) for p in re.findall(r'\d+', running_pos_str)]
    if len(pos_list) < 2: return 1.0
    
    early_pos = pos_list[0]       # 早段位置 (前速)
    finish_pos = pos_list[-1]     # 終點位置
    position_change = early_pos - finish_pos # 正數代表追勢強 (例如 12 變 4)，負數代表力弱 (例如 1 變 8)

    multiplier = 1.0
    
    # ---------------------------------------------------
    # 邏輯 1：增程檢定 (Stepping Up)
    # ---------------------------------------------------
    if dist_diff >= 200:
        if position_change > 3: 
            # 早段落後，但末段狂追 -> 增程有利
            multiplier = 1.15
        elif early_pos <= 3 and position_change < -3:
            # 早段領放，但末段乏力 -> 增程不利 (長力盲點)
            multiplier = 0.85
            
    # ---------------------------------------------------
    # 邏輯 2：縮程檢定 (Dropping Down)
    # ---------------------------------------------------
    elif dist_diff <= -200:
        if early_pos <= 3 and position_change < -3:
            # 上仗放頭斷氣 -> 縮程有利 (利用天然前速)
            multiplier = 1.15
        elif early_pos >= 10:
            # 慢腳馬縮程 -> 步速太快跟不上，極不利
            multiplier = 0.80

    # ---------------------------------------------------
    # 練馬師意圖疊加 (Trainer Behavior Overlay)
    # ---------------------------------------------------
    # 練馬師在安排「轉途程」時的勝率也是一門學問
    # 若該練馬師擅長「刻意跑錯途程減分，再轉回首本途程出擊」
    strategic_trainers = ['蔡約翰', '告東尼', '呂健威', '大衛希斯']
    if trainer_name in strategic_trainers and multiplier > 1.0:
        multiplier += 0.05 # 練馬師意圖加成：強勢馬房的合理部署，信號更強
        
    return multiplier
