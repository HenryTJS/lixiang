# -*- coding: utf-8 -*-
"""T5 v6 / T7: 损伤度 D——组合证据（损伤型事件 + 全能量加速 + 应变(降权辅证)）

证据工程结论(2026-09-06):
  1. "损伤型事件"(能量显著超背景)证据在 AE 全程活跃组(012/013/023)早期干净:
     早期磨合/普通 AE 不算, 损伤型事件在 30-35% 才出现 → 解决 AE 磨合污染;
  2. 静默型(020/017)无巨事件, 需"全能量累积加速"证据 → e_ae = max(e_dmg, e_full);
  3. 应变单调漂移(载荷)与真发散同形态, 权重无法区分 → e_strain 降权(辅证)。

T7 参数化: 全部可调参数集中在 __init__ kwargs(默认=v6 配置)。
调用方(E/在线)不传参即默认; 校准脚本通过 params dict 注入。

证据:
  e_dmg    = 块内损伤型事件能量 log EWMA      (损伤型: logE>背景p60+LIFT)
  e_full   = 全能量累积 log 加速度(短/长窗斜率差 EWMA)
  e_shape  = 无量纲形状/比值列相对校准段基线的尾部抬升(默认 `PeakFactor`, 见 SHAPE_DEFAULTS)
  e_ae     = max(e_dmg, e_full[, e_shape])
  e_strain = estrain_w × 应变 std 相对滚动低分位发散
  e_fo     = fo_w × 光纤多通道"去共模空间极差"相对运行中位发散
  risk     = max(启用源的证据)   —— 源由 sources 掩码逐组声明
  D        = 升快降慢累积(rise 追 risk, fall 回落)

源掩码(sources): 声发射为主源且每组必做('ae'); 应变('strain')/光纤('fo')为辅助源,
  逐组按数据有无声明。sources=('ae',) 即声发射单源基线。默认 ('ae','strain') 与
  e_fo 引入前的行为一致(fo 不在默认掩码内), 保证历史结果可复现。
"""
import numpy as np
from .config import EXT_BLOCK_PTS

# --- 形状/比值证据的默认配置(v9.1, 2026-09-20 起为项目默认口径) ---
# 默认列 = `PeakFactor`(Peak/RMS): 逐列对照(9 组)中唯一同时做到
#   ① 救活 025(D_end 0.202→0.823, 预警从「无」→ 68.0% 寿命)
#   ② 016 零影响、018~020 变动 ≤1.5 个百分点
#   ③ 主样本 5 组分级仍 5/5 级 A
#   ④ 026 的 `t85` 保持「未达」
# 且它是**无量纲**峰值因子 → 免疫跨试件幅值尺度不可比。
SHAPE_DEFAULTS = {'shape_col': 'PeakFactor', 'shape_q': 95.0, 'shape_w': 0.6}


class _AmpChannel:
    """单通道“循环幅值”跟踪器。

    背景: 载荷 5 Hz、采样 10 Hz → **每周期恰好 2 个采样点**，未解调的通道会交替
    采到峰/谷，此时 块均值=直流电平、**块内 std=幅值**；已解调的通道则数值本身就是幅值。
    两种约定在同批数据里混存(实测 016 应变已解调, 其余 25 个通道均未解调),
    故按通道自动判别, 否则会把“直流电平”错当幅值。

    判定量(对采样“翻相”免疫): alt = mean|Δx₁| / mean|Δx₂|
      alt ≫ 1 → 未解调振荡 (相邻差大、隔点差≈零)
      alt ≈ 1 → 已解调平滑
    实测两簇分离清楚: 016 应变=0.94, 其余通道 4.9~352。
    """

    def __init__(self, demod_win, demod_th):
        self.demod_win = int(demod_win)
        self.demod_th = float(demod_th)
        self._p1 = None
        self._p2 = None
        self._d1 = 0.0
        self._d2 = 0.0
        self._dn = 0
        self.raw = None          # None=未判别; True=未解调; False=已解调
        self._bn = 0
        self._bsum = 0.0
        self._bsq = 0.0

    def add(self, v):
        v = float(v)
        if not np.isfinite(v):
            return
        if self.raw is None:
            if self._p1 is not None and self._p2 is not None:
                self._d1 += abs(v - self._p1)
                self._d2 += abs(v - self._p2)
                self._dn += 1
            self._p2 = self._p1
            self._p1 = v
            if self._dn >= self.demod_win:
                self.raw = bool(self._d1 / max(self._d2, 1e-12) > self.demod_th)
        self._bn += 1
        self._bsum += v
        self._bsq += v * v

    def block_amp(self):
        """结算本块幅值并清零; 本块无数据→None。"""
        if self._bn == 0:
            return None
        n = self._bn
        m = self._bsum / n
        sd = float(np.sqrt(max(0.0, self._bsq / n - m * m)))
        amp = sd if (self.raw is None or self.raw) else abs(m)
        self._bn = 0
        self._bsum = 0.0
        self._bsq = 0.0
        return amp

    def mode(self):
        if self.raw is None:
            return '?'
        return 'RAW' if self.raw else 'DEMOD'


class OnlineDamageIndex:
    def __init__(self, params=None):
        p = dict(params or {})
        # --- 证据级(可校准) ---
        self.bg_warm = int(p.get('bg_warm', 200))       # 背景建立事件数
        self.bg_max = int(p.get('bg_max', 800))         # 背景滚动窗上限
        self.lift = float(p.get('lift', 2.0))           # 损伤型判据: 背景p60+lift(log)
        self.e_dmg_scale = float(p.get('e_dmg_scale', 2.5))
        self.acc_scale = float(p.get('acc_scale', 0.02))
        self.estrain_w = float(p.get('estrain_w', 0.6))
        self.strain_ratio_gain = float(p.get('strain_ratio_gain', 2.0))
        self.estrain_min_blocks = int(p.get('estrain_min_blocks', 20))
        # --- AE 失效自适应(默认关; 仅当调用方显式开启) ---
        # 适用场景: 某些试件的 AE 信号"整体高位且平稳"，不存在显著超背景的突发事件
        # → e_ae 恒 ≈ 0。此时应变辅证(权重上限 estrain_w)成为唯一可用证据，
        #   D 被锁在 estrain_w 以下（如 L1-56 停在 0.58），无法表达更高损伤等级。
        # 机制: 维护最近 ae_dead_win 个块(已结算)的 e_ae 历史；若该窗口内最大值仍
        #   < ae_dead_thr，则判定 AE 源无信息，将应变权重提升至 estrain_w_hi。
        # 纯因果: 只用已结算块的信息, 零未来信息; 窗口按**块数**而非寿命比例,
        #   故短寿命试件(总块数 < 窗口)永不触发, 行为与原先完全一致。
        self.ae_auto = bool(p.get('ae_auto', False))
        self.ae_dead_win = int(p.get('ae_dead_win', 20))
        self.ae_dead_thr = float(p.get('ae_dead_thr', 0.1))
        self.estrain_w_hi = float(p.get('estrain_w_hi', 1.0))
        # --- 块长(点) = 证据结算粒度, 固定 EXT_BLOCK_PTS ---
        # 注: 曾引入可调 blk_pts(按寿命比例缩放)以对齐 L1 的 n_f 跨度,
        #     实测无效且使预警更早(块越细, 越界判据的 max 统计量抽样越密)
        #     → 已回退; 块长不是 D 过早临危的原因。
        # --- 源掩码融合 ---
        # 声发射为主源(ae, 每组必做); 应变(strain)与光纤(fo)为辅助源, 逐组按数据有无声明。
        # 默认 ('ae','strain') == 既有行为(fo 未引入前的口径), 保证历史结果可复现。
        self.sources = tuple(p.get('sources', ('ae', 'strain')))
        # 融合算子: max(任一源报警) | mean(多源平均) | min(需全源共识)
        self.fusion = str(p.get('fusion', 'max'))
        # e_fo(光纤局部化): 多通道块均值 → 减通道均值(消共模: 温度/整体应变) → 跨通道极差
        #                    → 相对运行中位的增幅 → clip → ×fo_w
        self.fo_w = float(p.get('fo_w', 0.6))
        self.fo_ratio_gain = float(p.get('fo_ratio_gain', 1.0))
        # --- 应变证据双模式 ---
        # 'std'   : 块内应变 std 相对运行中位发散（旧口径，默认，向后兼容）
        # 'stiff' : **刚度损失率** = |块均值| 相对运行低分位基线的相对增长
        #           载荷(应力)控制疲劳下 应变幅值 ∝ 1/刚度, 故相对增长 = 刚度损失率。
        self.strain_mode = str(p.get('strain_mode', 'std'))
        # 刚度损失基线: 取**固定校准段** [stiff_cal_lo, stiff_cal_hi) 块的 |块均值| 低分位。
        # 校准段避开开机瞬态且位于损伤起始之前 —— 物理上对应"初始健康刚度 E0"。
        # 用运行分位作基线会随退化漂移, 不符合"相对初始刚度"的定义, 故不用。
        self.stiff_base_pct = float(p.get('stiff_base_pct', 30))
        self.stiff_cal_lo = int(p.get('stiff_cal_lo', 20))
        self.stiff_cal_hi = int(p.get('stiff_cal_hi', 45))
        # --- 级别层闸门(异源分级) ---
        # >0 时: L2 需 stiff_loss≥lvl2_stiff, L3 需 stiff_loss≥lvl3_stiff。
        # 设计意图: AE 决定检测级(L1)与 D; 刚度退化只作为 L2/L3 的"确认闸门"，
        #           从而把"检测"与"退化确认"在时间上分开(分级梯度)。
        # 默认 0 = 不启用闸门 = 旧行为(级别仅由 D 阈值决定)。
        self.lvl2_stiff = float(p.get('lvl2_stiff', 0.0))
        self.lvl3_stiff = float(p.get('lvl3_stiff', 0.0))
        # --- 通道幅值跟踪(自动判别解调状态) ---
        # demod_win: 判别所需点数(10Hz 下 5000 点=500s=10 块, 远早于校准段)
        # demod_th : alt 阈值(实测两簇: 0.94 vs ≥4.9, 取 2.0 余量充足)
        self.demod_win = int(p.get('demod_win', 5000))
        self.demod_th = float(p.get('demod_th', 2.0))
        # --- D 状态机(可校准) ---
        self.rise = float(p.get('rise', 0.12))
        self.fall = float(p.get('fall', 0.008))
        self.levels = [0.25, 0.55, 0.85]
        # --- T8 消融模式 ---
        # full        完整 v6: e_ae=max(e_dmg,e_full); risk=max(e_ae,e_strain); 升快降慢
        # no_dmg      去损伤型事件分类(e_ae=max(e_full, e_shape))
        # no_full     去全能量加速(e_ae=max(e_dmg, e_shape))
        # no_strain   去应变证据(risk=e_ae)
        # only_strain 仅应变证据(risk=e_strain)
        # no_accum    去单调累积(D=risk, 无记忆)
        self.abl = str(p.get('abl', 'full'))
        # --- v9: 形状/比值证据 e_shape (可选, 默认关) ---
        # 动机: 25 个 AE 列以往只用 Peak。逐列证据筛查(2026-09-20)显示：
        #   **无量纲比值**列 (MarginFactor=Peak/RootMAV, PeakFactor=Peak/RMS,
        #   ImpulseFactor=Peak/MAV) 在 **9/9 组**上趋势为正，且在 Peak 失效的
        #   025/026 上仍有 1.7~3.1 倍尾部抬升；而原始量级列(RMS/MeanSquare…)趋势反向、
        #   频域 6 列无一致趋势。比值列天然免疫"幅值类跨试件尺度不可比"。
        # 逐列端到端对照(6 列 × 9 组, q95 w0.6)选出 PeakFactor 为默认，见 SHAPE_DEFAULTS。
        # 构造(纯因果, 与 stiff_loss 同一习惯):
        #   块内事件特征取 p_shape_q 分位 → 用**固定校准段** [shape_cal_lo, shape_cal_hi)
        #   块值的 shape_base_pct 分位作基线 → e_shape = clip((cur/base-1)/shape_gain, 0, 1)
        # shape_col=None → 完全关闭(恢复到 v6「只用 Peak」口径)，行为与历史逐位一致。
        # shape_col=None → 完全关闭，行为与历史**逐位一致**。
        # shape_w 默认 0.6 = 与 estrain_w 同一"辅助证据"权重约定(NOT 调参):
        #   实测 w=1.0 时 e_shape 单独即可把 risk 顶到 1.0, 9 组 D_end 全部饱和到
        #   ~1.0 → 跨试件区分度归零; w=0.6 时 025 D_end 0.202→0.824 而其余组不变。
        self.shape_col = p.get('shape_col', SHAPE_DEFAULTS['shape_col'])
        self.shape_q = float(p.get('shape_q', SHAPE_DEFAULTS['shape_q']))
        self.shape_gain = float(p.get('shape_gain', 2.0))
        self.shape_cal_lo = int(p.get('shape_cal_lo', 20))
        self.shape_cal_hi = int(p.get('shape_cal_hi', 45))
        self.shape_base_pct = float(p.get('shape_base_pct', 30))
        self.shape_w = float(p.get('shape_w', SHAPE_DEFAULTS['shape_w']))
        # --- v7 结构: 磨合/加载抑制窗(点) ---
        # 前 suppress_pts 点内证据照常学习(基线/背景), 但 risk 不累积进 D(排除加载/磨合瞬态)。
        # 在线无总长时传固定点; 离线评估可传 n*frac。
        self._suppress_pts = int(p.get('suppress_pts', 0))
        # --- v8 结构: 在线就绪检测(route A, 磨合/加载结束 = AE能量峰值回落 AND 应变稳定) ---
        # ready_mode='off' 不用(纯 v6/v7); 'auto' 用在线检测替代寿命分数抑制。
        # 语义: 抑制期 = 就绪前(证据照常学基线, 但 risk 不累积); 就绪后永不抑制。
        self.ready_mode = str(p.get('ready_mode', 'off'))
        # v8b: 就绪 = 最近连续 ready_quiet_blk 块"平静"(无 AE 事件 且 应变 std 不高)
        #       → 加载/磨合/漂移过渡段活动高不平静; 进入稳定服役期才就绪(latch)。
        self.ready_quiet_blk = int(p.get('ready_quiet_blk', 8))
        self.ready_st_fac = float(p.get('ready_st_fac', 1.5))
        # --- 方向A: 损伤不可逆确认后加速追赶(latch) —— 默认开(解决 0.85 不可达) ---
        # 目的: 解决 D 慢 rise 追不满断裂前脉冲证据 → 0.85(临危)级不可达。
        # 机制: 当 D 经确认(最近 conf_blk 块 min≥conf_drop 且 ≥conf_low)后, 对 risk 用快 rise 追赶。
        # 主样本 5 组验证: 达 0.85 级全部可达, 且预警 onset/分级不受影响。
        self.latch_enable = bool(p.get('latch_enable', True))
        self.latch_conf_low = float(p.get('latch_conf_low', 0.30))
        self.latch_conf_drop = float(p.get('latch_conf_drop', 0.15))
        self.latch_conf_blk = int(p.get('latch_conf_blk', 3))
        self.latch_rise_fast = float(p.get('latch_rise_fast', 0.5))
        self.reset()

    def reset(self):
        self._bpts = 0
        self._seen_pts = 0
        self._block_dmg = 0.0        # 块损伤型能量
        self._block_all = 0.0        # 块全能量(事件)
        self._strain_vals = []
        self._st_amps = []
        self.stiff_loss = 0.0
        self._amp_st = _AmpChannel(self.demod_win, self.demod_th)
        self._amp_fo = {}
        self._fo_amp = {}
        self._bg_loge = []
        self._bg_ready = False
        self._dlog_s = 0.0           # 损伤型块 log EWMA
        self._cum_energy = 0.0
        self._cum_logs = []
        self._accel_ewma = 0.0
        self._strain_stds = []
        self.risk = 0.0
        self.damage = 0.0
        self._bcount = 0
        self.level = 0
        self._last_e_ae = 0.0
        self._last_e_strain = 0.0
        self._last_e_fo = 0.0
        self._ready = False
        self._ae_ch = False
        self._st_ch = False
        self._quiet_run = 0
        self._recent_aelog = []
        self._aelog_this = 0.0
        self._has_st_this = False
        self._latched = False
        self._d_hist = []
        self._ae_hist = []                 # 自适应用的 e_ae 滚动窗口
        self._estrain_w = self.estrain_w   # 实际生效的应变权重(可变)
        self._block_shape = []             # 本块事件的特征值(e_shape 用)
        self._shape_hist = []              # 逐块的 p_q(特征) 历史
        self._last_e_shape = 0.0

    def _on_event(self, peak):
        peak = max(float(peak), 0.0)
        loge = np.log10(peak ** 2 + 1e-12)
        if self._bg_ready:
            if loge > float(np.percentile(self._bg_loge, 60)) + self.lift:
                self._block_dmg += peak ** 2        # 损伤型
                return
            self._bg_loge.append(loge)
        else:
            self._bg_loge.append(loge)
            if len(self._bg_loge) >= self.bg_warm:
                self._bg_ready = True
        if len(self._bg_loge) > self.bg_max:
            self._bg_loge.pop(0)

    def _block_settle(self):
        self._bcount += 1
        self._aelog_this = np.log10(self._block_all + 1.0)   # 本块 AE 事件能量(块内累积), 供就绪检测
        # --- e_dmg ---
        dlog = np.log10(self._block_dmg + 1.0)
        self._block_dmg = 0.0
        self._dlog_s = 0.7 * self._dlog_s + 0.3 * dlog
        e_dmg = float(min(max(self._dlog_s / self.e_dmg_scale, 0.0), 1.0))
        # --- e_full (全能量累积加速) ---
        self._cum_energy += self._block_all
        self._block_all = 0.0
        self._cum_logs.append(np.log10(self._cum_energy + 1.0))
        if len(self._cum_logs) > 40:
            self._cum_logs.pop(0)
        e_full = 0.0
        logs = self._cum_logs
        if len(logs) >= 14:
            sl = (logs[-1] - logs[-13]) / 12.0
            ss = (logs[-1] - logs[-4]) / 3.0
            self._accel_ewma = 0.6 * self._accel_ewma + 0.4 * (ss - sl)
            e_full = float(min(max(self._accel_ewma / self.acc_scale, 0.0), 1.0))
        # --- e_shape (无量纲形状/比值证据, 可选; 默认关) ---
        e_shape = 0.0
        if self.shape_col is not None:
            q = float(np.percentile(self._block_shape, self.shape_q)) \
                if len(self._block_shape) >= 3 else np.nan
            self._block_shape = []
            if np.isfinite(q):
                self._shape_hist.append(q)
                if len(self._shape_hist) > 200:
                    self._shape_hist.pop(0)
            slo, shi = int(self.shape_cal_lo), int(self.shape_cal_hi)
            if np.isfinite(q) and self._bcount > shi and len(self._shape_hist) > shi:
                w = np.asarray(self._shape_hist[slo:shi], dtype=float)
                w = w[np.isfinite(w)]
                if len(w) >= 5:
                    sb = float(np.percentile(w, self.shape_base_pct))
                    if abs(sb) > 1e-12:
                        e_shape = float(min(max(
                            (q / sb - 1.0) / max(self.shape_gain, 1e-9),
                            0.0), 1.0)) * self.shape_w
        self._last_e_shape = e_shape
        # --- 按消融模式组合 e_ae ---
        # 语义: 每个消融模式**只**去掉它名字对应的那一条证据, 保留其余 AE 证据。
        # ⚠️ 2026-09-20 修正: e_shape 升为默认后, 旧写法(no_dmg=e_full, no_full=e_dmg)
        #    会**连带丢掉 e_shape**, 导致消融表把形状证据的贡献错记到 e_full 名下。
        if self.abl == 'no_dmg':
            self._last_e_ae = max(e_full, e_shape)
        elif self.abl == 'no_full':
            self._last_e_ae = max(e_dmg, e_shape)
        elif self.abl == 'only_shape':
            self._last_e_ae = e_shape
        elif self.abl == 'no_shape':
            self._last_e_ae = max(e_dmg, e_full)
        else:
            self._last_e_ae = max(e_dmg, e_full, e_shape)
        # --- 刚度损失率(因果, 载荷控制不变量) ---
        # 幅值 = 本块 std(未解调通道) 或 |本块均值|(已解调通道), 由 _AmpChannel 自动判定。
        # 基线 = 固定校准段 [stiff_cal_lo, stiff_cal_hi) 块幅值的低分位(≈初始健康刚度)。
        amp_st = self._amp_st.block_amp()
        stiff = 0.0
        if amp_st is not None:
            self._st_amps.append(amp_st)
            if len(self._st_amps) > 200:
                self._st_amps.pop(0)
        lo, hi = int(self.stiff_cal_lo), int(self.stiff_cal_hi)
        if amp_st is not None and self._bcount > hi and len(self._st_amps) > hi:
            win = np.asarray(self._st_amps[lo:hi], dtype=float)
            win = win[np.isfinite(win)]
            if len(win) >= 5:
                sbm = float(np.percentile(win, self.stiff_base_pct))
                if sbm > 1e-9:
                    stiff = float(max(0.0, amp_st / sbm - 1.0))
        self.stiff_loss = stiff
        # --- AE 失效自适应: 窗口内 e_ae 全低于阈值 → 应变升为主证(纯因果) ---
        if self.ae_auto:
            self._ae_hist.append(self._last_e_ae)
            if len(self._ae_hist) > self.ae_dead_win:
                self._ae_hist.pop(0)
            if len(self._ae_hist) >= self.ae_dead_win \
               and max(self._ae_hist) < self.ae_dead_thr:
                self._estrain_w = self.estrain_w_hi
            else:
                self._estrain_w = self.estrain_w
        # e_strain (降权): 按 strain_mode 选择构造
        e_strain = 0.0
        self._has_st_this = bool(self._strain_vals)
        if self._strain_vals:
            self._strain_stds.append(float(np.std(self._strain_vals)))
            self._strain_vals = []
            if len(self._strain_stds) > 120:
                self._strain_stds.pop(0)
        if self.strain_mode == 'stiff':
            e_strain = float(min(stiff / max(self.strain_ratio_gain, 1e-9), 1.0)) * self._estrain_w
        elif len(self._strain_stds) >= self.estrain_min_blocks:
            sbase = float(np.percentile(self._strain_stds, 50))
            if sbase > 1e-9:
                ratio = self._strain_stds[-1] / sbase
                e_strain = float(min(max((ratio - 1.0) / self.strain_ratio_gain, 0.0), 1.0))
                e_strain *= self._estrain_w
        self._last_e_strain = e_strain
        # --- e_fo (光纤多通道局部化, 辅助源) ---
        # 各通道“循环幅值”(std/|mean| 自动判定) 相对自身校准段基线的增长;
        # 取**跨通道最大增长** = 退化最严重的局部位置 → 局部化证据。
        e_fo = 0.0
        if self._amp_fo:
            gains = []
            for ch, tr in self._amp_fo.items():
                a_ch = tr.block_amp()
                if a_ch is None:
                    continue
                hist = self._fo_amp.setdefault(ch, [])
                hist.append(a_ch)
                if len(hist) > 200:
                    hist.pop(0)
                if self._bcount > hi and len(hist) > hi:
                    w = np.asarray(hist[lo:hi], dtype=float)
                    w = w[np.isfinite(w)]
                    if len(w) >= 5:
                        fb = float(np.percentile(w, self.stiff_base_pct))
                        if fb > 1e-9:
                            gains.append(max(0.0, a_ch / fb - 1.0))
            if gains:
                g = float(max(gains))
                e_fo = float(min(g / max(self.fo_ratio_gain, 1e-9), 1.0)) * self.fo_w
        self._last_e_fo = e_fo
        # --- 在线就绪检测(route A) ---
        if self.ready_mode != 'off':
            self._check_ready()
        # --- risk / D(源掩码融合 + 消融模式) ---
        # risk = max(启用源的证据)。声发射为主源; strain/fo 按 sources 声明参与。
        # abl='only_strain'/'no_strain' 保留为消融专用覆盖(不改变既有消融口径)。
        if self.abl == 'only_strain':
            self.risk = e_strain
        elif self.abl == 'no_strain':
            self.risk = self._last_e_ae
        else:
            evs = []
            if 'ae' in self.sources:
                evs.append(self._last_e_ae)
            if 'strain' in self.sources:
                evs.append(e_strain)
            if 'fo' in self.sources:
                evs.append(e_fo)
            if not evs:
                self.risk = 0.0
            elif self.fusion == 'mean':
                self.risk = float(np.mean(evs))
            elif self.fusion == 'min':
                self.risk = float(np.min(evs))
            else:
                self.risk = float(np.max(evs))
        suppressed = (self._seen_pts <= self._suppress_pts) or \
                     (self.ready_mode != 'off' and not self._ready)
        if suppressed:
            # 磨合/加载/就绪前抑制期: 证据照常学, 但不进入损伤度
            self.damage = 0.0
            self.level = 0
            return
        if self.abl == 'no_accum':
            self.damage = max(0.0, min(1.0, self.risk))
        else:
            if self.risk > self.damage:
                self.damage += self.rise * (self.risk - self.damage)
            else:
                self.damage -= self.fall * self.damage
            self.damage = max(0.0, min(1.0, self.damage))
        # --- 方向A: 损伤不可逆确认后加速追赶(解决 0.85 不可达) ---
        if self.latch_enable and self.abl != 'no_accum':
            self._d_hist.append(self.damage)
            if len(self._d_hist) > self.latch_conf_blk:
                self._d_hist.pop(0)
            if not self._latched and len(self._d_hist) >= self.latch_conf_blk \
               and self.damage >= self.latch_conf_low \
               and min(self._d_hist) >= self.latch_conf_drop:
                self._latched = True
            if self._latched and self.risk > self.damage:
                self.damage += self.latch_rise_fast * (self.risk - self.damage)
                self.damage = max(0.0, min(1.0, self.damage))
        self._update_level()

    def _check_ready(self):
        """在线就绪检测 v8b(route A): 就绪 = 最近连续 ready_quiet_blk 块"平静"。
        平静块定义: 无显著 AE 事件能量(aelog<0.05) 且 (应变 std 不高于运行中位 ready_st_fac 倍
        —— 或本块无应变/应变史不足时不判活跃)。
        加载/磨合/漂移段活动高 → 不平静; 进入稳定服役期后连续平静 → ready(latch, 此后永不抑制)。
        应变数据全程缺失的组: 仅由 AE 通道决定(此时 AE 也是其唯一证据源)。
        """
        if self._ready:
            return
        if self._bcount < 6:
            return
        # --- 本块是否"活跃" ---
        ae_active = self._aelog_this > 0.05                       # 本块有显著 AE 事件能量
        st_active = False
        if self._has_st_this and len(self._strain_stds) >= 8:
            base = float(np.percentile(self._strain_stds, 50))
            if base > 0 and self._strain_stds[-1] > base * self.ready_st_fac:
                st_active = True                                  # 应变 std 显著高于运行中位
        active = ae_active or st_active
        # --- 连续平静计数 ---
        if active:
            self._quiet_run = 0
        else:
            self._quiet_run += 1
            if self._quiet_run >= self.ready_quiet_blk:
                self._ready = True

    def _update_level(self):
        """D 阈值 → 原始级别; 再用刚度损失率作为 L2/L3 闸门(分级梯度)。"""
        lv = 0
        for i, th in enumerate(self.levels):
            if self.damage >= th:
                lv = i + 1
        if self.lvl3_stiff > 0 and lv >= 3 and self.stiff_loss < self.lvl3_stiff:
            lv = 2
        if self.lvl2_stiff > 0 and lv >= 2 and self.stiff_loss < self.lvl2_stiff:
            lv = 1
        self.level = lv

    def shape_value(self, ae_dict):
        """按 shape_col 从事件字典取形状/比值特征值; 未启用或缺值→None。

        供调用方一行接入: di.update(strain, peak, fo, di.shape_value(p['ae']))
        shape_col=None 时恒返回 None → update 行为与历史逐位一致。
        """
        if self.shape_col is None or not ae_dict:
            return None
        v = ae_dict.get('ae_' + self.shape_col, None)
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    def update(self, strain, ae_peak=None, fo=None, ae_shape=None):
        """逐点推进。fo = {通道名: 值} (光纤, 可选; 缺省或全 NaN 时该块无光纤证据)。

        ae_shape: 可选的无量纲形状/比值特征值(与 ae_peak 同事件同刻传入)。
                  shape_col=None 时忽略，行为与历史逐位一致。
        """
        self._bpts += 1
        self._seen_pts += 1
        if ae_peak is not None:
            peak = max(float(ae_peak), 0.0)
            self._block_all += peak ** 2
            self._on_event(peak)
        if self.shape_col is not None and ae_shape is not None:
            try:
                sv = float(ae_shape)
            except (TypeError, ValueError):
                sv = np.nan
            if np.isfinite(sv):
                self._block_shape.append(sv)
        if strain is not None and not np.isnan(strain):
            self._strain_vals.append(strain)
            self._amp_st.add(strain)
        if fo:
            for k, v in fo.items():
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fv):
                    continue
                t = self._amp_fo.get(k)
                if t is None:
                    t = _AmpChannel(self.demod_win, self.demod_th)
                    self._amp_fo[k] = t
                t.add(fv)
        if self._bpts >= EXT_BLOCK_PTS:
            self._bpts = 0
            self._block_settle()
        return self.damage

    def get_level(self):
        return self.level
