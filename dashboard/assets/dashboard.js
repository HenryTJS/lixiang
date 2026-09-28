/* ============================================================
 * SHM-ONLINE · 疲劳机多源在线损伤监测看板  (纯自研, 无外部依赖)
 * ------------------------------------------------------------
 * 数据: dashboard/data/*.js  (由 export_dashboard.py 离线导出)
 *   D / risk / eae / est : ×1000      st / fo : ×100
 *   ael : log10(peak)×1000, 无事件 = -99999
 *   aen : 帧内 AE 事件数       lv : 0/1/2/3
 * ============================================================ */
(function () {
  'use strict';

  /* ---------------- 常量 ---------------- */
  var DEF_FRAME_DT = 0.5;             // 每帧代表的真实时间步:
                                      //   主样本 = 0.5 s (STEP=5 @10Hz); L1 = 50 cycle(数据包自带 frameDt 覆盖)
  var WIN = 1200;                     // 子面板滑动窗口(帧)
  var LV_COLORS = ['#00e39a', '#ffc93c', '#ff8a1f', '#ff3b47'];
  var LV_NAMES = ['正常', '注意', '预警', '临危'];
  var LV_DESC = [
    ['结构状态正常', '损伤度位于安全区间，各源信号平稳'],
    ['注意级 · 损伤累积启动', 'D 越过 0.25，建议提高巡检频次'],
    ['预警级 · 损伤加速扩展', 'D 越过 0.55，请安排停机检查'],
    ['临危级 · 逼近结构失效', 'D 越过 0.85，立即停机并隔离试件']
  ];
  var FO_COLORS = ['#25d4f0', '#7c9dff', '#b07cff', '#ff9de2', '#5fe08a',
    '#ffd166', '#ef6f6c', '#8bd450', '#f78fb3', '#6ad4dd'];
  var C_D = '#25d4f0', C_RISK = '#ff8a1f', C_EST = '#00e39a', C_AE = '#ff8a1f';

  /* ---------------- 全局状态 ---------------- */
  var S = {
    gid: null, ds: null, data: null, frame: 0, playing: false, speed: 120,
    last: 0, dirty: true, muted: false, ac: null,
    warnIdx: 0, logCount: 0, booted: false,
    unit: 's', frameDt: DEF_FRAME_DT, datasets: null, chanCells: null,
    acc: { ae: 0, fo: 0, st: 0 }, cache: {}
  };

  /* ---------------- DOM ---------------- */
  var $ = function (id) { return document.getElementById(id); };
  var el = {};
  ['gidSel', 'dsSel', 'sysState', 'sysStateText', 'clock', 'btnRun', 'btnReset', 'btnSound', 'sndText',
    'speedSeg', 'seek', 'tlPct', 'tNow', 'tTotal', 'roIdx', 'roAe', 'roRate',
    'gaugeArc', 'gaugeMark25', 'gaugeMark55', 'gaugeMark85', 'dVal', 'dBadge', 'dMax', 'dLife',
    'dMargin', 'gaugeTag', 'tLv1', 'tLv2', 'tLv3', 'verdict', 'verdictSub',
    'barAE', 'barST', 'barRK', 'valAE', 'valST', 'valRK',
    'vAE', 'vFO', 'vST', 'vEV', 'logList', 'logCount', 'hAEn', 'hFOn', 'hSTn', 'hDIn',
    'hFOc', 'hFOs', 'chanBody', 'equipRig', 'equipMode', 'tAE', 'tFO', 'tST', 'boot', 'bootText',
    'tDmg', 'tTrend', 'dUnit', 'lgD', 'dMarginLbl',
    'dfosRow', 'tDF', 'vDF', 'cvDFHeat', 'cvDFProf',
    'cvTrend', 'cvAE', 'cvFO', 'cvST', 'cvEV', 'lgB2', 'lgB3', 'lgC0', 'lgRef'
  ].forEach(function (k) { el[k] = $(k); });

  /* 主指标相关文案的**初始默认值** —— 数据包未提供覆盖字段时必须还原，
     否则从一个数据集切到另一个会残留上一个的名称（如 L1 的“离线复评”）。 */
  var DEF_LB = {
    tDmg: el.tDmg ? el.tDmg.textContent : '',
    tTrend: el.tTrend ? el.tTrend.textContent : '',
    dUnit: el.dUnit ? el.dUnit.textContent : '',
    lgD: el.lgD ? el.lgD.textContent : '',
    dMarginLbl: el.dMarginLbl ? el.dMarginLbl.textContent : ''
  };

  /* ============================================================
   *  一、绘图工具
   * ============================================================ */
  function prep(cv) {
    var w = cv.clientWidth || 300, h = cv.clientHeight || 120;
    var dpr = window.devicePixelRatio || 1;
    if (cv._w !== w || cv._h !== h) {
      cv._w = w; cv._h = h;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function box(W, H, l, t, r, b) { return { x: l, y: t, w: W - l - r, h: H - t - b }; }

  function bg(ctx, b, fill) {
    ctx.fillStyle = fill || 'rgba(3,7,11,.72)';
    ctx.fillRect(b.x, b.y, b.w, b.h);
  }

  function grid(ctx, b, nx, ny, yFmt, color) {
    ctx.save();
    ctx.strokeStyle = color || 'rgba(37,212,240,.09)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var i = 1; i < nx; i++) {
      var x = Math.round(b.x + b.w * i / nx) + .5;
      ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h);
    }
    for (var j = 1; j < ny; j++) {
      var y = Math.round(b.y + b.h * j / ny) + .5;
      ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y);
    }
    ctx.stroke();
    // 边框
    ctx.strokeStyle = 'rgba(37,212,240,.2)';
    ctx.strokeRect(b.x + .5, b.y + .5, b.w - 1, b.h - 1);
    // y 轴刻度文字
    if (yFmt) {
      ctx.fillStyle = '#5b7288';
      ctx.font = '9px "Cascadia Mono",Consolas,monospace';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (var k = 0; k <= ny; k++) {
        var yy = b.y + b.h * k / ny;
        ctx.fillText(yFmt(1 - k / ny), b.x - 4, yy);
      }
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    }
    ctx.restore();
  }

  function yOf(v, lo, hi, b) {
    var t = (v - lo) / ((hi - lo) || 1);
    return b.y + b.h - Math.max(0, Math.min(1, t)) * b.h;
  }

  /* 折线(逐像素取样 + 区间末值), 支持空值断线 */
  function plotLine(ctx, b, arr, i0, i1, lo, hi, color, opt) {
    opt = opt || {};
    var n = i1 - i0;
    if (n <= 1) return;
    var px = Math.max(2, Math.round(b.w));
    var step = n / px, empty = opt.empty;
    ctx.save();
    ctx.lineWidth = opt.width || 1.4;
    ctx.strokeStyle = color;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (opt.glow) { ctx.shadowColor = color; ctx.shadowBlur = opt.glow; }
    if (opt.alpha != null) ctx.globalAlpha = opt.alpha;
    ctx.beginPath();
    var pen = false, lastX = 0, lastY = 0;
    for (var p = 0; p <= px; p++) {
      var j = Math.min(i1 - 1, Math.max(i0, Math.ceil(i0 + (p + 1) * step) - 1));
      var v = arr[j];
      if (v === null || v === undefined || (empty !== undefined && v === empty)) { pen = false; continue; }
      var x = b.x + b.w * p / px;
      var y = yOf(v, lo, hi, b);
      if (!pen) { ctx.moveTo(x, y); pen = true; } else { ctx.lineTo(x, y); }
      lastX = x; lastY = y;
    }
    ctx.stroke();
    ctx.restore();
    return { x: lastX, y: lastY };
  }

  /* 面积填充(渐变) */
  function plotFill(ctx, b, arr, i0, i1, lo, hi, color) {
    var n = i1 - i0;
    if (n <= 1) return;
    var g = ctx.createLinearGradient(0, b.y, 0, b.y + b.h);
    g.addColorStop(0, color + '55'); g.addColorStop(1, color + '00');
    ctx.save();
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(b.x, b.y + b.h);
    var px = Math.max(2, Math.round(b.w));
    var step = n / px;
    for (var p = 0; p <= px; p++) {
      var j = Math.min(i1 - 1, Math.max(i0, Math.ceil(i0 + (p + 1) * step) - 1));
      ctx.lineTo(b.x + b.w * p / px, yOf(arr[j], lo, hi, b));
    }
    ctx.lineTo(b.x + b.w, b.y + b.h);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  /* 柱状 */
  function plotBars(ctx, b, arr, i0, i1, lo, hi, color) {
    var n = i1 - i0;
    if (n <= 1) return;
    var px = Math.max(2, Math.round(b.w));
    var step = n / px;
    var bw = Math.max(1, b.w / px - 1);
    ctx.save();
    ctx.fillStyle = color;
    var zero = yOf(lo, lo, hi, b);
    for (var p = 0; p < px; p++) {
      var j = Math.min(i1 - 1, Math.max(i0, Math.ceil(i0 + (p + 1) * step) - 1));
      var v = arr[j];
      if (!v) continue;
      var y = yOf(v, lo, hi, b);
      ctx.globalAlpha = .28 + .6 * Math.min(1, v / (hi || 1));
      ctx.fillRect(b.x + b.w * p / px, y, bw, zero - y);
    }
    ctx.restore();
  }

  function hline(ctx, b, y, color, dash, label, W) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(b.x, Math.round(y) + .5); ctx.lineTo(b.x + b.w, Math.round(y) + .5);
    ctx.stroke();
    ctx.setLineDash([]);
    if (label) {
      ctx.fillStyle = color;
      ctx.font = '9px "Cascadia Mono",Consolas,monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(label, b.x + b.w - (W || 0) - 2, y - 2);
    }
    ctx.restore();
  }

  function vline(ctx, b, x, color, dash, label) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + .5, b.y); ctx.lineTo(Math.round(x) + .5, b.y + b.h);
    ctx.stroke();
    ctx.setLineDash([]);
    if (label) {
      ctx.fillStyle = color;
      ctx.font = '9px "Cascadia Mono",Consolas,monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(label, x + 3, b.y + 2);
    }
    ctx.restore();
  }

  /* ============================================================
   *  二、数据解码 / 派生量
   * ============================================================ */
  function dec(arr, k) {                        // 解码 ×k
    var o = new Float32Array(arr.length);
    for (var i = 0; i < arr.length; i++) o[i] = arr[i] / k;
    return o;
  }

  function getSeries() {
    var d = S.data, c = S.cache;
    if (c.gid !== d.gid) {
      c.gid = d.gid;
      c.D = dec(d.D, 1000); c.risk = dec(d.risk, 1000);
      c.hiF = (d.hiF && d.hiF.length) ? dec(d.hiF, 1000) : null;
      c.eae = dec(d.eae, 1000); c.est = dec(d.est, 1000);
      c.st = dec(d.st, 100);
      c.ael = dec(d.ael, 1000);
      c.fo = {};
      for (var k in d.fo) c.fo[k] = dec(d.fo[k], 100);
      // DFOS 块级序列(L1 才有; 主样本无)
      c.dloc = d.dfos ? dec(d.dfos.local, 100) : null;
      c.dhi = d.dfos ? dec(d.dfos.hi, 1000) : null;
      // DFOS 空间分布(×100 编码 → 解码为 με) 与基线
      c.dfProf = (d.dfos && d.dfos.prof) ? dec(d.dfos.prof, 100) : null;
      c.dfBase = (d.dfos && d.dfos.base) ? dec(d.dfos.base, 100) : null;
      // 应变滑动波动(std, 窗口 300 帧) —— 仅连续采样源有意义
      c.stStd = rollingStd(c.st, 300);
      // AE 事件率(主样本 次/秒; L1 无"秒" → 次/帧) 与 累计前缀和(健康表用)
      var rk = rateK();
      c.rate = new Float32Array(d.nfr);
      c.aeSum = new Float64Array(d.nfr);
      var run = 0;
      for (var q = 0; q < d.nfr; q++) {
        c.rate[q] = d.aen[q] * rk;
        run += d.aen[q];
        c.aeSum[q] = run;
      }
      // 各通道全局范围
      c.foRange = {};
      for (var kk in c.fo) {
        var a = c.fo[kk], mn = Infinity, mx = -Infinity;
        for (var i = 0; i < a.length; i++) { if (a[i] < mn) mn = a[i]; if (a[i] > mx) mx = a[i]; }
        c.foRange[kk] = [mn, mx];
      }
      c.stRange = rangeOf(c.st);
    }
    return c;
  }

  function rangeOf(a) {
    var mn = Infinity, mx = -Infinity;
    for (var i = 0; i < a.length; i++) { if (a[i] < mn) mn = a[i]; if (a[i] > mx) mx = a[i]; }
    if (!isFinite(mn)) { mn = 0; mx = 1; }
    if (mx - mn < 1e-9) { mx = mn + 1; }
    return [mn, mx];
  }

  function rollingStd(a, w) {
    var n = a.length, out = new Float32Array(n);
    var s = 0, s2 = 0;
    for (var i = 0; i < n; i++) {
      s += a[i]; s2 += a[i] * a[i];
      if (i >= w) { s -= a[i - w]; s2 -= a[i - w] * a[i - w]; }
      var m = i >= w ? w : i + 1;
      var v = s2 / m - (s / m) * (s / m);
      out[i] = v > 0 ? Math.sqrt(v) : 0;
    }
    return out;
  }

  /* ---- 时间基 / 单位自适应: 主样本 = 秒; L1 = cycle ---- */
  /* 每帧推进的"播放步"。
     主样本: = 数据包 frameDt (0.5 s/帧, 与 10Hz+STEP=5 对应)。
     L1: 无实时概念, frameDt 是"50 cycle/帧"的语义值, 直接拿来做播放速率会使全程耗时
         约 20 min → 改按**全长归一**: 默认 120× 时全长 ≈ 40 s (与主样本同量级)。 */
  function frameDt() {
    var d = S.data;
    if (isCycle()) return d ? Math.max(0.15, 4800 / Math.max(1, d.nfr)) : DEF_FRAME_DT;
    return (d && d.frameDt) || DEF_FRAME_DT;
  }
  function isCycle() { return S.unit === 'cycle'; }
  function rateK() { return isCycle() ? 1 : 2; }   // 主样本 2 帧/s; L1 1 帧 = 50 cycle

  /* ============================================================
   *  三、渲染
   * ============================================================ */
  function render() {
    if (!S.data) return;
    var d = S.data, c = getSeries();
    var f = Math.floor(S.frame);
    if (f < 0) f = 0; if (f > d.nfr - 1) f = d.nfr - 1;

    drawTrend(d, c, f);
    drawAE(d, c, f);
    drawFO(d, c, f);
    drawST(d, c, f);
    drawEV(d, c, f);
    drawDFOS(d, c, f);
    drawGauge(d, c, f);
    drawSide(d, c, f);
    drawStatus(d, c, f);
  }

  /* ---- 主趋势 D(t) ---- */
  function drawTrend(d, c, f) {
    var o = prep(el.cvTrend), ctx = o.ctx;
    var b = box(o.w, o.h, 34, 10, 66, 18);
    bg(ctx, b);
    grid(ctx, b, 8, 4, function (t) { return t.toFixed(2); });

    var W = d.nfr - 1;
    var xOf = function (i) { return b.x + b.w * (i / W); };

    // 阈值色带
    var bands = [[0, .25, 'rgba(0,227,154,.045)'], [.25, .55, 'rgba(255,201,60,.05)'],
    [.55, .85, 'rgba(255,138,31,.06)'], [.85, 1, 'rgba(255,59,71,.08)']];
    bands.forEach(function (bd) {
      ctx.fillStyle = bd[2];
      var y1 = yOf(bd[1], 0, 1, b), y2 = yOf(bd[0], 0, 1, b);
      ctx.fillRect(b.x, y1, b.w, y2 - y1);
    });
    // 阈值线
    hline(ctx, b, yOf(.25, 0, 1, b), 'rgba(255,201,60,.5)', [3, 3], '0.25 注意', 52);
    hline(ctx, b, yOf(.55, 0, 1, b), 'rgba(255,138,31,.5)', [3, 3], '0.55 预警', 52);
    hline(ctx, b, yOf(.85, 0, 1, b), 'rgba(255,59,71,.55)', [3, 3], '0.85 临危', 52);

    // b2 / b3 参考锚 —— 在线语义: 仅当回放已越过该时刻才标出(不预知未来)
    var b2 = d.meta.b2, b3 = d.meta.b3;
    var b2f = b2 != null ? W * b2 / 100 : -1;
    var b3f = b3 != null ? W * b3 / 100 : -1;
    var showB2 = b2f > 0 && f >= b2f, showB3 = b3f > 0 && f >= b3f;
    if (showB2) vline(ctx, b, xOf(b2f), 'rgba(176,124,255,.6)', [5, 4], 'b2');
    if (showB3) vline(ctx, b, xOf(b3f), 'rgba(255,59,71,.55)', [5, 4], 'b3');
    if (el.lgB2) el.lgB2.style.display = showB2 ? '' : 'none';
    if (el.lgB3) el.lgB3.style.display = showB3 ? '' : 'none';

    // c0(基线重定义终点) / 论文检测点(L1) —— 同样遵循"回放越过才出现"的在线语义
    var c0f = d.meta.c0Pct != null ? W * d.meta.c0Pct / 100 : -1;
    var showC0 = c0f > 0 && f >= c0f;
    if (showC0) vline(ctx, b, xOf(c0f), 'rgba(0,227,154,.5)', [2, 3], 'c0');
    if (el.lgC0) el.lgC0.style.display = showC0 ? '' : 'none';
    var refs = d.meta.refs || [], showR = false;
    refs.forEach(function (r) {
      var rf = r.pct != null ? W * r.pct / 100 : -1;
      if (rf > 0 && f >= rf) {
        vline(ctx, b, xOf(rf), 'rgba(150,170,190,.4)', [1, 3], r.label || 'ref');
        showR = true;
      }
    });
    if (el.lgRef) el.lgRef.style.display = showR ? '' : 'none';

    // 已回放区域高亮 + 扫描光带(强化实时感)
    var cx0 = xOf(f);
    ctx.save();
    ctx.fillStyle = 'rgba(234,252,255,.045)';
    ctx.fillRect(b.x, b.y, cx0 - b.x, b.h);
    if (f > 2) {
      var sg = ctx.createLinearGradient(cx0 - 90, 0, cx0, 0);
      sg.addColorStop(0, 'rgba(37,212,240,0)');
      sg.addColorStop(1, 'rgba(37,212,240,.10)');
      ctx.fillStyle = sg;
      ctx.fillRect(Math.max(b.x, cx0 - 90), b.y, Math.min(90, cx0 - b.x), b.h);
    }
    ctx.restore();

    // 仅绘制"当前时刻及之前"的数据 (在线实时语义: 不预显未来曲线)
    if (f > 1) {
      plotFill(ctx, b, c.D, 0, f + 1, 0, 1, C_D);
      if (c.hiF) {
        plotLine(ctx, b, c.hiF, 0, f + 1, 0, 1, 'rgba(150,170,190,.95)', { width: 1.3 });
      }
      plotLine(ctx, b, c.risk, 0, f + 1, 0, 1, 'rgba(255,138,31,.85)', { width: 1.2 });
      plotLine(ctx, b, c.D, 0, f + 1, 0, 1, C_D, { width: 2, glow: 8 });
    }
    // 论文 HI_F 参考曲线：图例 + 与 D 的提前量标注（仅 L1 数据集有此字段）
    var lgHI = document.getElementById('lgHI');
    if (lgHI) {
      lgHI.style.display = c.hiF ? '' : 'none';
      var lgHIt = document.getElementById('lgHIt');
      if (lgHIt && c.hiF) {
        var lp = (d.meta && d.meta.hiLeadPt != null) ? d.meta.hiLeadPt : null;
        lgHIt.textContent = '论文 HI_F 参考（离线）' +
          (lp != null ? ' · D 早 ' + Number(lp).toFixed(1) + ' 个百分点' : '');
      }
    }
    // 光标
    var cx = xOf(f);
    vline(ctx, b, cx, 'rgba(234,252,255,.75)');
    var cy = yOf(c.D[f], 0, 1, b);
    ctx.save();
    ctx.fillStyle = LV_COLORS[d.lv[f]] || C_D;
    ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(cx, cy, 3.4, 0, 6.2832); ctx.fill();
    ctx.restore();

    // 当前值跟随标签
    var col = LV_COLORS[d.lv[f]] || C_D;
    var lbl = c.D[f].toFixed(3);
    ctx.font = '11px "Cascadia Mono",Consolas,monospace';
    var tw = ctx.measureText(lbl).width + 12;
    var lx = Math.min(cx + 7, b.x + b.w - tw - 2);
    ctx.fillStyle = 'rgba(6,14,20,.86)';
    ctx.fillRect(lx, cy - 9, tw, 18);
    ctx.strokeStyle = col; ctx.lineWidth = 1;
    ctx.strokeRect(lx + .5, cy - 8.5, tw - 1, 17);
    ctx.fillStyle = col;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(lbl, lx + tw / 2, cy + .5);

    // x 轴寿命刻度
    ctx.fillStyle = '#5b7288';
    ctx.font = '9px "Cascadia Mono",Consolas,monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    for (var i = 0; i <= 4; i++) {
      var xi = xOf(W * i / 4);
      ctx.fillText((i * 25) + '%', xi + 2, b.y + b.h + 4);
    }
    ctx.textAlign = 'right';
    ctx.fillText('寿命', b.x + b.w, b.y + b.h + 4);
  }

  /* ---- 子面板窗口: 右端始终锁定当前帧(在线实时滚动) ---- */
  function win(d, f) {
    var i1 = Math.min(d.nfr, f + 1);
    var i0 = Math.max(0, i1 - WIN);
    return [i0, i1];
  }

  /* ---- AE ---- */
  function drawAE(d, c, f) {
    var o = prep(el.cvAE), ctx = o.ctx;
    var b = box(o.w, o.h, 26, 8, 6, 14);
    bg(ctx, b);
    var aax = (d.ax && d.ax.ael) ? d.ax.ael : [-3.2, 1.05];
    var LO = aax[0], HI = aax[1];
    var rhi = (d.ax && d.ax.rate) || 6;
    grid(ctx, b, 6, 2, function (t) { return (t * rhi).toFixed(0); });
    var w = win(d, f), i0 = w[0], i1 = w[1];
    // 事件率柱
    plotBars(ctx, b, c.rate, i0, i1, 0, rhi, C_AE);
    // log 峰值
    hline(ctx, b, yOf(0, LO, HI, b), 'rgba(255,138,31,.22)', null, null);
    plotLine(ctx, b, c.ael, i0, i1, LO, HI, '#ffd08a', { width: 1.3, empty: -99.999, glow: 5 });
    ctx.fillStyle = '#5b7288';
    ctx.font = '9px "Cascadia Mono",Consolas,monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('log10(峰值能量)', b.x + 3, b.y + b.h + 11);
    ctx.textAlign = 'right';
    ctx.fillText(isCycle() ? '事件率 (次/帧)' : '事件率 ×0.5/s', b.x + b.w, b.y + b.h + 11);
  }

  /* ---- FBG ---- */
  function drawFO(d, c, f) {
    var o = prep(el.cvFO), ctx = o.ctx;
    var cols = Object.keys(c.fo);
    var rows = Math.max(1, Math.ceil(cols.length / 5));
    var b = box(o.w, o.h, 26, 8, 6, 6 + rows * 10);
    bg(ctx, b);
    if (!cols.length) {
      ctx.fillStyle = '#3d5566';
      ctx.font = '11px "Cascadia Mono",Consolas,monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('光纤通道未接入', b.x + b.w / 2, b.y + b.h / 2);
      ctx.strokeStyle = 'rgba(91,114,136,.35)';
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(b.x + .5, b.y + .5, b.w - 1, b.h - 1);
      ctx.setLineDash([]);
      return;
    }
    var w = win(d, f), i0 = w[0], i1 = w[1];
    var lo = Infinity, hi = -Infinity;
    cols.forEach(function (k) { var r = c.foRange[k]; if (r[0] < lo) lo = r[0]; if (r[1] > hi) hi = r[1]; });
    var pad = (hi - lo) * .08 || 1;
    lo -= pad;
    hi += pad;
    grid(ctx, b, 6, 2, function (t) { return (lo + (hi - lo) * t).toFixed(1); });
    cols.forEach(function (k, idx) {
      plotLine(ctx, b, c.fo[k], i0, i1, lo, hi, FO_COLORS[idx % FO_COLORS.length],
        { width: 1.3, glow: 4 });
    });
    // 图例(每行最多 5 个, 自动换行)
    ctx.font = '9px "Cascadia Mono",Consolas,monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    cols.forEach(function (k, idx) {
      var x = b.x + 2 + (idx % 5) * Math.max(30, Math.min(64, b.w / 5));
      var y = b.y + b.h + 5 + Math.floor(idx / 5) * 10;
      ctx.fillStyle = FO_COLORS[idx % FO_COLORS.length];
      ctx.fillRect(x, y, 7, 2);
      ctx.fillStyle = '#5b7288';
      ctx.fillText(k, x + 10, y + 6);
    });
  }

  /* ---- 应变 ---- */
  function drawST(d, c, f) {
    var o = prep(el.cvST), ctx = o.ctx;
    var dfos = !!c.dloc;
    var b = box(o.w, o.h, 26, 8, 6, 14);
    bg(ctx, b);
    var w = win(d, f), i0 = w[0], i1 = w[1];
    var r = c.stRange;
    var pad = (r[1] - r[0]) * .1 || 1;
    var lo = r[0] - pad, hi = r[1] + pad;
    grid(ctx, b, 6, 2, function (t) { return (lo + (hi - lo) * t).toFixed(0); });
    plotLine(ctx, b, c.st, i0, i1, lo, hi, C_EST, { width: 1.2, glow: 4 });
    // 第二曲线(独立比例): L1 = DFOS 块级局部峰; 主样本 = 应变滑动波动 σ
    var s2 = dfos ? c.dloc : c.stStd, ii;
    var mx = 0;
    for (ii = i0; ii < i1; ii++) if (s2[ii] > mx) mx = s2[ii];
    mx = mx || 1;
    plotLine(ctx, b, s2, i0, i1, 0, mx, 'rgba(176,124,255,.9)', { width: dfos ? 1.4 : 1.2 });
    ctx.font = '9px "Cascadia Mono",Consolas,monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillStyle = C_EST; ctx.fillRect(b.x + 4, b.y + b.h + 5, 7, 2);
    ctx.fillStyle = '#5b7288';
    ctx.fillText(dfos ? 'FBG 块均值' : '应变', b.x + 14, b.y + b.h + 11);
    ctx.fillStyle = 'rgba(176,124,255,.9)'; ctx.fillRect(b.x + 66, b.y + b.h + 5, 7, 2);
    ctx.fillStyle = '#5b7288';
    ctx.fillText(dfos ? 'DFOS 峰' : '波动 σ', b.x + 76, b.y + b.h + 11);
  }

  /* ---- 证据层 ---- */
  function drawEV(d, c, f) {
    var o = prep(el.cvEV), ctx = o.ctx;
    var b = box(o.w, o.h, 26, 8, 6, 14);
    bg(ctx, b);
    grid(ctx, b, 6, 4, function (t) { return t.toFixed(2); });
    var w = win(d, f), i0 = w[0], i1 = w[1];
    hline(ctx, b, yOf(.25, 0, 1, b), 'rgba(255,201,60,.35)', [3, 3]);
    hline(ctx, b, yOf(.55, 0, 1, b), 'rgba(255,138,31,.35)', [3, 3]);
    hline(ctx, b, yOf(.85, 0, 1, b), 'rgba(255,59,71,.4)', [3, 3]);
    plotLine(ctx, b, c.eae, i0, i1, 0, 1, 'rgba(255,208,138,.9)', { width: 1.1 });
    plotLine(ctx, b, c.est, i0, i1, 0, 1, 'rgba(0,227,154,.85)', { width: 1.1 });
    plotLine(ctx, b, c.risk, i0, i1, 0, 1, C_RISK, { width: 1.8, glow: 7 });
    var lg = [['e_ae', 'rgba(255,208,138,.95)'], ['e_st', 'rgba(0,227,154,.95)'], ['risk', C_RISK]];
    ctx.font = '9px "Cascadia Mono",Consolas,monospace';
    ctx.textBaseline = 'bottom';
    lg.forEach(function (p, idx) {
      var x = b.x + 4 + idx * 40;
      ctx.fillStyle = p[1]; ctx.fillRect(x, b.y + b.h + 5, 7, 2);
      ctx.fillStyle = '#5b7288'; ctx.fillText(p[0], x + 10, b.y + b.h + 11);
    });
  }

  /* ---- 分布式应变空间分布 / 相对基线偏离热图 (L1 专有) ---- */
  function dfColor(t) {                       // t∈[-1,1] → 青(负) / 暗(0) / 橙(正)
    if (t > 1) t = 1; else if (t < -1) t = -1;
    if (t >= 0) return [(16 + 239 * t) | 0, (28 + 62 * t) | 0, (38 + 22 * t) | 0];
    var s = -t;
    return [(16 + 21 * s) | 0, (28 + 184 * s) | 0, (38 + 202 * s) | 0];
  }

  /* 分位数(最多抽样 2000 点, 足够稳定) —— 用于鲁棒归一, 抵抗 DFOS 掉点 */
  function dfQuantile(arr, q, n) {
    var m = (n === undefined) ? arr.length : n;
    var step = Math.max(1, Math.floor(m / 2000));
    var tmp = [];
    for (var i = 0; i < m; i += step) tmp.push(arr[i]);
    if (!tmp.length) return 0;
    tmp.sort(function (a, b) { return a - b; });
    var k = (tmp.length - 1) * q;
    var f = Math.floor(k), c2 = Math.min(tmp.length - 1, f + 1);
    return tmp[f] + (tmp[c2] - tmp[f]) * (k - f);
  }

  /* 离屏预渲染整条寿命的偏离热图(x=空间, y=块), 每帧仅 drawImage 裁剪 → 流畅 */
  function dfHeat(d, c) {
    if (c.heat && c.heat._gid === d.gid) return c.heat;
    var df = d.dfos, nblk = df.nblk, npos = df.npos;
    var prof = c.dfProf, base = c.dfBase;
    var cv = document.createElement('canvas');
    cv.width = npos; cv.height = nblk;
    var cx = cv.getContext('2d');
    var img = cx.createImageData(npos, nblk);
    var absdev = new Float32Array(prof.length);
    var i, b, p, rgb;
    for (i = 0; i < prof.length; i++) absdev[i] = Math.abs(prof[i] - base[i % npos]);
    // 鲁棒色标: p98(抗 DFOS 掉点); 超出者饱和显示
    var mx = Math.max(1e-6, dfQuantile(absdev, 0.98));
    for (b = 0; b < nblk; b++) {
      for (p = 0; p < npos; p++) {
        rgb = dfColor((prof[b * npos + p] - base[p]) / mx);
        i = (b * npos + p) * 4;
        img.data[i] = rgb[0]; img.data[i + 1] = rgb[1]; img.data[i + 2] = rgb[2];
        img.data[i + 3] = 240;
      }
    }
    cx.putImageData(img, 0, 0);
    cv._gid = d.gid;
    c.heatMx = mx;
    c.heat = cv;
    return cv;
  }

  function drawDFOS(d, c, f) {
    if (!el.dfosRow) return;
    if (!d.dfos || !c.dfProf) { el.dfosRow.style.display = 'none'; return; }
    el.dfosRow.style.display = '';

    var df = d.dfos, nblk = df.nblk, npos = df.npos, pos = df.pos;
    var prof = c.dfProf, base = c.dfBase;
    var bIdx = Math.max(0, Math.min(nblk - 1, df.blkOf[f]));
    var off = bIdx * npos;
    var heat = dfHeat(d, c);

    /* ---------- 左: 相对基线偏离热图 (x=位置, y=循环数) ---------- */
    var o1 = prep(el.cvDFHeat), x1 = o1.ctx;
    var b1 = box(o1.w, o1.h, 44, 8, 8, 26);
    bg(x1, b1);
    x1.save();
    x1.imageSmoothingEnabled = false;
    x1.drawImage(heat, 0, 0, npos, bIdx + 1, b1.x, b1.y, b1.w, b1.h * (bIdx + 1) / nblk);
    x1.restore();
    // 边框 + 水平网格
    x1.save();
    x1.strokeStyle = 'rgba(37,212,240,.2)';
    x1.strokeRect(b1.x + .5, b1.y + .5, b1.w - 1, b1.h - 1);
    x1.strokeStyle = 'rgba(37,212,240,.08)';
    x1.beginPath();
    for (var g = 1; g < 4; g++) {
      x1.moveTo(b1.x, Math.round(b1.y + b1.h * g / 4) + .5);
      x1.lineTo(b1.x + b1.w, Math.round(b1.y + b1.h * g / 4) + .5);
    }
    x1.stroke();
    x1.restore();
    // 轴: y = 循环数 / x = 位置
    x1.fillStyle = '#5b7288';
    x1.font = '9px "Cascadia Mono",Consolas,monospace';
    x1.textAlign = 'right'; x1.textBaseline = 'middle';
    var yLbl = ['0', fmtT(df.cyc[(nblk - 1) >> 1]), fmtT(df.cyc[nblk - 1])];
    for (var k = 0; k <= 2; k++) x1.fillText(yLbl[k], b1.x - 4, b1.y + b1.h * k / 2);
    x1.textAlign = 'left'; x1.textBaseline = 'top';
    for (var q = 0; q <= 2; q++) {
      var pp = Math.min(npos - 1, Math.round((npos - 1) * q / 2));
      x1.fillText(Math.round(pos[pp]) + (q === 2 ? ' mm' : ''), b1.x + b1.w * q / 2 + 2, b1.y + b1.h + 4);
    }
    x1.fillText('偏离基线  青=负 / 橙=正', b1.x + 2, b1.y + b1.h + 15);

    /* ---------- 右: 当前块空间分布 vs 基线 ---------- */
    var o2 = prep(el.cvDFProf), x2 = o2.ctx;
    var b2 = box(o2.w, o2.h, 46, 8, 8, 16);
    bg(x2, b2);
    var lo, hi;
    var l1v = Math.min(dfQuantile(base, 0.01), dfQuantile(prof, 0.01));
    var h1v = Math.max(dfQuantile(base, 0.99), dfQuantile(prof, 0.99));
    lo = l1v; hi = h1v;
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) { lo = -1; hi = 1; }
    var pd = (hi - lo) * .08 || 1;
    lo -= pd; hi += pd;
    grid(x2, b2, 4, 2, function (t) { return (lo + (hi - lo) * t).toFixed(0); });
    // 当前块 − 基线 的偏离填充
    x2.save();
    x2.beginPath();
    x2.moveTo(b2.x, yOf(prof[off], lo, hi, b2));
    for (var p1 = 1; p1 < npos; p1++) {
      x2.lineTo(b2.x + b2.w * p1 / (npos - 1), yOf(prof[off + p1], lo, hi, b2));
    }
    for (var p2 = npos - 1; p2 >= 0; p2--) {
      x2.lineTo(b2.x + b2.w * p2 / (npos - 1), yOf(base[p2], lo, hi, b2));
    }
    x2.closePath();
    x2.fillStyle = 'rgba(255,138,31,.16)';
    x2.fill();
    x2.restore();
    // 基线(块 0, 健康)
    x2.save();
    x2.strokeStyle = 'rgba(150,170,190,.85)';
    x2.lineWidth = 1;
    x2.setLineDash([4, 3]);
    x2.beginPath();
    for (var p3 = 0; p3 < npos; p3++) {
      var xa = b2.x + b2.w * p3 / (npos - 1), ya = yOf(base[p3], lo, hi, b2);
      if (p3) x2.lineTo(xa, ya); else x2.moveTo(xa, ya);
    }
    x2.stroke();
    x2.setLineDash([]);
    x2.restore();
    // 当前块分布
    plotLine(x2, b2, prof.slice(off, off + npos), 0, npos, lo, hi, '#ffd08a',
      { width: 1.6, glow: 6 });
    // 图例
    x2.font = '9px "Cascadia Mono",Consolas,monospace';
    x2.textAlign = 'left'; x2.textBaseline = 'bottom';
    x2.fillStyle = 'rgba(150,170,190,.95)'; x2.fillRect(b2.x + 2, b2.y + b2.h + 5, 7, 2);
    x2.fillStyle = '#5b7288'; x2.fillText('基线(块0)', b2.x + 12, b2.y + b2.h + 11);
    x2.fillStyle = '#ffd08a'; x2.fillRect(b2.x + 68, b2.y + b2.h + 5, 7, 2);
    x2.fillStyle = '#5b7288'; x2.fillText('当前块', b2.x + 78, b2.y + b2.h + 11);

    /* ---------- 读数 ---------- */
    var mxdev = 0, dv2;
    for (var p4 = 0; p4 < npos; p4++) {
      dv2 = Math.abs(prof[off + p4] - base[p4]);
      if (dv2 > mxdev) mxdev = dv2;
    }
    el.vDF.textContent = '块 ' + (bIdx + 1) + '/' + nblk + ' · ' + fmtT(df.cyc[bIdx]) +
      ' · 峰值偏离 ' + mxdev.toFixed(0) + ' με' +
      ' (色标 ±' + Math.round(c.heatMx || 0) +
      (df.spikeN ? ' · 去尖峰 ' + df.spikeN + ' 点' : '') + ')';
  }

  /* ---- 仪表 ---- */
  function drawGauge(d, c, f) {    var D = c.D[f], lv = d.lv[f], col = LV_COLORS[lv];
    var R = 76, CIRC = 2 * Math.PI * R;
    el.gaugeArc.setAttribute('stroke-dasharray', (CIRC * Math.max(0.001, D)).toFixed(1) + ' ' + CIRC.toFixed(1));
    el.gaugeArc.style.stroke = col;
    [[el.gaugeMark25, .25], [el.gaugeMark55, .55], [el.gaugeMark85, .85]].forEach(function (p) {
      p[0].setAttribute('stroke-dasharray', '2 ' + CIRC.toFixed(1));
      p[0].setAttribute('stroke-dashoffset', (-CIRC * p[1]).toFixed(1));
    });
    el.dVal.textContent = D.toFixed(3);
    el.dVal.style.color = lv ? col : '#eafcff';
    el.dVal.style.textShadow = '0 0 22px ' + col + '66';
    el.dBadge.textContent = LV_NAMES[lv];
    el.dBadge.style.color = col;
    el.dBadge.style.borderColor = col + '88';
    el.dBadge.style.background = col + '1a';

    var mx = 0;
    for (var i = 0; i <= f; i++) if (c.D[i] > mx) mx = c.D[i];
    if (mx < D) mx = D;
    el.dMax.textContent = mx.toFixed(3);
    var life = f / (d.nfr - 1) * 100;
    el.dLife.textContent = life.toFixed(1) + '%';
    el.dMargin.textContent = Math.max(0, 1 - D).toFixed(3);
    el.gaugeTag.textContent = S.playing ? 'RUNNING' : (S.frame > 0 ? 'PAUSED' : 'IDLE');
  }

  /* ---- 右侧分级/证据 ---- */
  function drawSide(d, c, f) {
    var lv = d.lv[f];
    var lampNodes = document.querySelectorAll('.lamp');
    for (var i = 0; i < lampNodes.length; i++) {
      var L = i + 1;
      lampNodes[i].classList.toggle('on', lv >= L);
    }
    // 报警时刻
    [1, 2, 3].forEach(function (L) {
      var w = d.warn.filter(function (x) { return x.lv === L; })[0];
      var node = el['tLv' + L];
      if (w && f >= w.f) node.textContent = fmtT(w.t);
      else node.textContent = '--';
    });
    var nm = LV_NAMES[lv];
    el.verdict.textContent = (lv > 0 ? '● ' : '○ ') + LV_DESC[lv][0];
    el.verdict.style.color = lv > 0 ? LV_COLORS[lv] : 'var(--txt)';
    el.verdictSub.textContent = LV_DESC[lv][1];

    var eae = c.eae[f], est = c.est[f], rk = c.risk[f];
    el.barAE.style.width = (eae * 100).toFixed(1) + '%';
    el.barST.style.width = (est * 100).toFixed(1) + '%';
    el.barRK.style.width = (rk * 100).toFixed(1) + '%';
    el.valAE.textContent = eae.toFixed(3);
    el.valST.textContent = est.toFixed(3);
    el.valRK.textContent = rk.toFixed(3);
  }

  /* ---- 通道健康表: 由数据包 chans 驱动(L1); 主样本还原静态行 ---- */
  var CHAN_HTML0 = null;                 // 主样本静态通道表(首次构建时缓存)
  function buildChanTable(d) {
    if (!el.chanBody) { S.chanCells = null; return; }
    if (CHAN_HTML0 === null) CHAN_HTML0 = el.chanBody.innerHTML;
    if (!d.chans) {                      // 主样本: 还原静态行并重新取引用
      el.chanBody.innerHTML = CHAN_HTML0;
      ['hAEn', 'hFOn', 'hSTn', 'hDIn', 'hFOc', 'hFOs'].forEach(function (k) { el[k] = $(k); });
      S.chanCells = null;
      return;
    }
    el.chanBody.innerHTML = '';
    S.chanCells = {};
    d.chans.forEach(function (ch) {
      var tr = document.createElement('tr');
      tr.innerHTML = '<td>' + ch.name + '</td><td>' + ch.mode + '</td><td>' +
        (ch.n == null ? '—' : ch.n) + '</td><td class="n">0</td>' +
        '<td><span class="st ok">在线</span></td>';
      el.chanBody.appendChild(tr);
      S.chanCells[ch.key] = { n: tr.children[3], s: tr.children[4].firstChild };
    });
  }

  function setChan(key, txt) {
    var c = S.chanCells && S.chanCells[key];
    if (c) c.n.textContent = txt;
  }

  function setChanState(key, on) {
    var c = S.chanCells && S.chanCells[key];
    if (!c) return;
    c.s.textContent = on ? '在线' : '未接入';
    c.s.className = 'st ' + (on ? 'ok' : 'off');
  }

  /* ---- 状态栏/健康表 ---- */
  function drawStatus(d, c, f) {
    var lv = d.lv[f];
    el.seek.value = Math.round(f / (d.nfr - 1) * 1000);
    el.seek.style.background = 'linear-gradient(90deg,' + (LV_COLORS[lv]) + ' 0%,' +
      (LV_COLORS[lv]) + ' ' + (f / (d.nfr - 1) * 100).toFixed(2) + '%,#132029 ' +
      (f / (d.nfr - 1) * 100).toFixed(2) + '%,#132029 100%)';
    el.tlPct.textContent = (f / (d.nfr - 1) * 100).toFixed(1) + '%';
    el.tNow.textContent = fmtT(d.t[f]);
    // 统一口径: 已回放的【原始】采样点数 = 帧数 × 降采样步长
    var rawPts = Math.min(d.n, (f + 1) * d.step);
    var nCh = d.foCols.length;
    var aeN = c.aeSum[f];
    el.roIdx.textContent = fmtNum(rawPts);
    el.roAe.textContent = fmtNum(aeN);
    el.roRate.textContent = (S.playing ? (S.speed / frameDt()) : 0).toFixed(0) +
      (isCycle() ? ' 帧/s' : ' /s');
    if (S.chanCells) {
      setChan('ae', fmtNum(aeN) + ' 事件');
      setChan('fo', nCh ? fmtNum(rawPts) + ' 点 × ' + nCh + ' 通道' : '—');
      setChan('dfos', d.dfos
        ? fmtNum(Math.min(d.dfos.nblk, (d.dfos.blkOf[f] || 0) + 1)) + ' / ' +
          fmtNum(d.dfos.nblk) + ' 块'
        : '—');
      setChan('engine', fmtNum(Math.floor(rawPts / 500)) + ' 块结算');
      setChanState('fo', !!nCh);
    } else {
      el.hAEn.textContent = fmtNum(aeN) + ' 事件';
      el.hFOn.textContent = nCh ? fmtNum(rawPts) + ' 点 × ' + nCh + ' 通道' : '—';
      el.hSTn.textContent = fmtNum(rawPts) + ' 点';
      el.hDIn.textContent = fmtNum(Math.floor(rawPts / 500)) + ' 块';
      el.hFOc.textContent = nCh || '0';
      if (nCh) { el.hFOs.textContent = '在线'; el.hFOs.className = 'st ok'; }
      else { el.hFOs.textContent = '未接入'; el.hFOs.className = 'st off'; }
    }

    el.vAE.textContent = (d.aen[f] * rateK()) + (isCycle() ? ' 次/帧' : ' 次/s');
    var fo = c.fo, ks = Object.keys(fo);
    if (ks.length > 3) {
      var msum = 0;
      ks.forEach(function (k) { msum += fo[k][f]; });
      el.vFO.textContent = ks.length + ' 通道 · 均值 ' + (msum / ks.length).toFixed(1);
    } else {
      el.vFO.textContent = ks.length
        ? ks.map(function (k) { return k + ' ' + fo[k][f].toFixed(2); }).join('  ') : '离线';
    }
    el.vST.textContent = c.st[f].toFixed(1) + ' με' +
      (c.dloc ? ' | DFOS峰 ' + c.dloc[f].toFixed(0) : '');
    el.vEV.textContent = 'e_ae ' + c.eae[f].toFixed(2) + ' | risk ' + c.risk[f].toFixed(2);

    var st = el.sysState;
    st.className = 'sys-state' + (lv >= 2 ? ' alarm' : (S.playing ? ' run' : (S.frame > 0 ? ' pause' : '')));
    el.sysStateText.textContent = lv >= 2 && S.playing ? LV_NAMES[lv] + '报警'
      : (S.playing ? '监测中' : (S.frame > 0 ? '已暂停' : '待机'));
  }

  function fmtT(t) {
    if (isCycle()) {                        // L1: 时间基 = 循环数
      t = Math.max(0, t);
      if (t >= 10000) return (t / 1000).toFixed(1) + 'k cyc';
      return Math.round(t) + ' cyc';
    }
    t = Math.max(0, Math.round(t));
    var h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60;
    return (h < 10 ? '0' : '') + h + ':' + (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
  }

  function fmtNum(v) {
    return String(Math.round(v)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /* ============================================================
   *  四、播放引擎
   * ============================================================ */
  function play() {
    if (!S.data) return;
    if (S.frame >= S.data.nfr - 1) S.frame = 0;
    S.playing = true;
    S.last = 0;
    el.btnRun.classList.add('playing');
    el.btnRun.querySelector('b').textContent = '暂停监测';
    pushLog(0, '系统', '在线监测已启动 · 试件 ' + S.gid);
    S.dirty = true;
  }

  function pause(auto) {
    S.playing = false;
    el.btnRun.classList.remove('playing');
    el.btnRun.querySelector('b').textContent = '开始监测';
    if (auto) pushLog(0, '系统', '数据回放结束 · 试件 ' + S.gid + ' 全程监测完成');
    invalidate();
  }

  function reset() {
    pause(false);
    S.frame = 0; S.warnIdx = 0;
    el.logList.innerHTML = '<div class="log-empty">— 暂无记录 —</div>';
    S.logCount = 0; el.logCount.textContent = '0 条';
    pushLog(0, '系统', '系统复位 · 回到试验起点');
    invalidate();
  }

  /* 标记需重绘；暂停态直接同步渲染(不等动画帧, 拖动进度条即时生效) */
  function invalidate() {
    S.dirty = true;
    if (!S.playing && S.data) { S.dirty = false; syncWarn(); render(); }
  }

  function loop(ts) {
    requestAnimationFrame(loop);
    var dt = S.last ? (ts - S.last) / 1000 : 0;
    S.last = ts;
    if (S.playing && S.data) {
      S.frame += Math.min(.25, dt) * S.speed / frameDt();
      var end = S.data.nfr - 1;
      if (S.frame >= end) { S.frame = end; pause(true); }
      syncWarn();
      render();
    } else if (S.dirty) {
      S.dirty = false; syncWarn(); render();
    }
  }

  /* ---- 预警日志同步 ---- */
  function syncWarn() {
    var d = S.data; if (!d) return;
    var f = Math.floor(S.frame);
    var changed = false;
    while (S.warnIdx < d.warn.length && d.warn[S.warnIdx].f <= f) {
      var w = d.warn[S.warnIdx++];
      pushLog(w.lv, LV_NAMES[w.lv], '损伤度 D 突破 ' + ['.25', '.55', '.85'][w.lv - 1] +
        ' 阈值 → 进入「' + LV_NAMES[w.lv] + '」级  (t=' + fmtT(w.t) + ')');
      beep(w.lv);
      changed = true;
    }
    if (changed) S.dirty = true;
  }

  function pushLog(lv, tag, msg) {
    if (S.logCount === 0) el.logList.innerHTML = '';
    var row = document.createElement('div');
    row.className = 'log-row l' + lv;
    var t = S.data ? (S.data.t[Math.floor(S.frame)] || 0) : 0;
    row.innerHTML = '<span class="lt">' + fmtT(t) + '</span>' +
      '<span class="ll">' + tag + '</span>' +
      '<span class="lm">' + msg + '</span>';
    el.logList.insertBefore(row, el.logList.firstChild);
    S.logCount++;
    el.logCount.textContent = S.logCount + ' 条';
    while (el.logList.childNodes.length > 240) el.logList.removeChild(el.logList.lastChild);
  }

  function beep(lv) {
    if (S.muted) return;
    try {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      S.ac = S.ac || new AC();
      var ac = S.ac;
      if (ac.state === 'suspended') ac.resume();
      var freq = lv >= 3 ? 900 : (lv === 2 ? 620 : 440);
      var n = lv >= 3 ? 3 : (lv === 2 ? 2 : 1);
      for (var i = 0; i < n; i++) {
        var t0 = ac.currentTime + i * .19;
        var osc = ac.createOscillator(), g = ac.createGain();
        osc.type = lv >= 3 ? 'square' : 'sine';
        osc.frequency.setValueAtTime(freq, t0);
        g.gain.setValueAtTime(0, t0);
        g.gain.linearRampToValueAtTime(lv >= 3 ? .09 : .05, t0 + .012);
        g.gain.exponentialRampToValueAtTime(.0001, t0 + .17);
        osc.connect(g); g.connect(ac.destination);
        osc.start(t0); osc.stop(t0 + .2);
      }
    } catch (e) { /* 忽略音频异常 */ }
  }

  /* ============================================================
   *  五、数据加载 / 交互
   * ============================================================ */
  function loadGroup(gid) {
    return new Promise(function (res, rej) {
      if (window.SHM_DATA && window.SHM_DATA[gid]) {
        S.data = window.SHM_DATA[gid]; res(S.data); return;
      }
      var s = document.createElement('script');
      s.src = ((S.ds && S.ds.path) || 'data/') + gid + '.js';
      s.onload = function () {
        S.data = window.SHM_DATA[gid]; res(S.data);
      };
      s.onerror = function () { rej(new Error('数据包加载失败: ' + gid)); };
      document.head.appendChild(s);
    });
  }

  function selectGroup(gid, keepLog) {
    pause(false);
    S.frame = 0;
    el.boot.classList.remove('hide');
    el.bootText.textContent = '正在加载试件 ' + gid + ' 数据包 …';
    loadGroup(gid).then(function (d) {
      S.gid = gid; S.warnIdx = 0;
      S.unit = d.unit || 's';
      S.frameDt = d.frameDt || DEF_FRAME_DT;
      S.cache = {};
      getSeries();
      el.gidSel.value = gid;
      el.btnRun.disabled = false; el.btnReset.disabled = false; el.seek.disabled = false;
      el.tTotal.textContent = '/ ' + fmtT(d.dur);
      if (el.equipRig && d.rig) el.equipRig.textContent = d.rig;
      if (el.equipMode && d.mode) el.equipMode.textContent = d.mode;
      // 主指标名称可由数据包覆盖（L1 第二批 = 「离线复评 HI_AE」，非“损伤度 D(t)”）；
      // 未提供时必须**还原默认**，否则跨数据集切换会残留上一个的名称。
      var LB = d.labels || {};
      if (el.tDmg) el.tDmg.textContent = d.indexName || DEF_LB.tDmg;
      if (el.tTrend) el.tTrend.textContent = d.trendName || DEF_LB.tTrend;
      if (el.dUnit) el.dUnit.textContent = LB.unit || DEF_LB.dUnit;
      if (el.lgD) el.lgD.textContent = LB.legend || DEF_LB.lgD;
      if (el.dMarginLbl) el.dMarginLbl.textContent = LB.margin || DEF_LB.dMarginLbl;
      buildChanTable(d);
      var TT = isCycle()
        ? { ae: '声发射 · 事件率 / 峰值', fo: '光纤光栅 · 多通道', st: '分布式应变 · 块均值 / 局部峰' }
        : { ae: '声发射 · 事件率 / 峰值', fo: '光纤光栅 · 多通道', st: '应变 · 波形 / 波动' };
      if (el.tAE) el.tAE.textContent = TT.ae;
      if (el.tFO) el.tFO.textContent = TT.fo;
      if (el.tST) el.tST.textContent = TT.st;
      el.boot.classList.add('hide');
      S.dirty = true;
      if (!keepLog) {
        el.logList.innerHTML = '<div class="log-empty">— 暂无记录 —</div>';
        S.logCount = 0; el.logCount.textContent = '0 条';
        var desc = d.chans
          ? d.chans.filter(function (x) { return x.key !== 'engine'; })
            .map(function (x) { return x.name.split(' ')[0] + ' ' + x.n + ' ' + (x.unit || '通道'); }).join(' / ')
          : ('光纤 ' + (d.foCols.length || 0) + ' 通道 / 声发射 25 通道 / 应变 1 通道');
        pushLog(0, '系统', '试件 ' + gid + ' 传感通道已接入 · ' + desc + ' · 等待在线数据');
      }
      render();
    }).catch(function (e) {
      el.bootText.textContent = e.message;
      pushLog(3, '异常', e.message);
      setTimeout(function () { el.boot.classList.add('hide'); }, 1400);
    });
  }

  function bind() {
    if (el.dsSel) el.dsSel.addEventListener('change', function () { selectDataset(this.value); });
    el.gidSel.addEventListener('change', function () { selectGroup(this.value); });
    el.btnRun.addEventListener('click', function () { S.playing ? pause(false) : play(); });
    el.btnReset.addEventListener('click', function () { reset(); });
    el.btnSound.addEventListener('click', function () {
      S.muted = !S.muted;
      el.sndText.textContent = S.muted ? '声音 关' : '声音 开';
      this.classList.toggle('playing', S.muted);
    });
    el.speedSeg.addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      S.speed = parseInt(b.dataset.sp, 10);
      Array.prototype.forEach.call(el.speedSeg.children, function (x) {
        x.classList.toggle('on', x === b);
      });
      invalidate();
    });
    el.seek.addEventListener('input', function () {
      if (!S.data) return;
      S.frame = this.value / 1000 * (S.data.nfr - 1);
      S.warnIdx = 0;
      S.data.warn.forEach(function (w) { if (w.f <= S.frame) S.warnIdx++; });
      invalidate();
    });
    // 键盘(避免与表单控件冲突)
    window.addEventListener('keydown', function (e) {
      var tag = (e.target && e.target.tagName) || '';
      if (tag === 'SELECT' || tag === 'INPUT' || tag === 'BUTTON') return;
      if (e.code === 'Space') { e.preventDefault(); S.playing ? pause(false) : play(); }
      else if (e.code === 'ArrowRight') {
        S.frame = Math.min(S.data ? S.data.nfr - 1 : 0, S.frame + 200); invalidate();
      } else if (e.code === 'ArrowLeft') {
        S.frame = Math.max(0, S.frame - 200); invalidate();
      }
    });
    setInterval(function () {
      var n = new Date();
      el.clock.textContent = [n.getHours(), n.getMinutes(), n.getSeconds()]
        .map(function (x) { return (x < 10 ? '0' : '') + x; }).join(':');
    }, 1000);
    window.addEventListener('resize', function () { invalidate(); });
    // 标签页切回前台: 重置时基并立即重绘(否则动画帧暂停期间无渲染)
    document.addEventListener('visibilitychange', function () {
      S.last = 0;
      if (!document.hidden) invalidate();
    });
  }

  function collectDatasets() {
    var dss = [], k, seen = {};
    var D = window.SHM_DATASETS;
    if (D) {
      for (k in D) {
        if (D[k] && D[k].groups && D[k].groups.length) { dss.push(D[k]); seen[D[k].id] = 1; }
      }
    }
    // 主样本清单向后兼容: export_dashboard.py 写的是 window.SHM_INDEX
    if (window.SHM_INDEX && window.SHM_INDEX.length && !seen.main) {
      dss.unshift({ id: 'main', name: '疲劳机主样本 016-020 / 022-027', unit: 's', path: 'data/',
        groups: window.SHM_INDEX });
    }
    return dss;
  }

  function fillGroupSel(ds) {
    el.gidSel.innerHTML = '';
    ds.groups.forEach(function (g) {
      var o = document.createElement('option');
      o.value = g.gid;
      var extra = (g.spec && g.spec.kind) ? ' · ' + g.spec.kind : '';
      o.textContent = '试件 ' + g.gid + extra;
      el.gidSel.appendChild(o);
    });
  }

  function selectDataset(dsId) {
    var ds = (S.datasets || []).filter(function (x) { return x.id === dsId; })[0];
    if (!ds) return;
    S.ds = ds;
    fillGroupSel(ds);
    if (el.dsSel) el.dsSel.value = dsId;
    selectGroup(ds.groups[0].gid);
  }

  function init() {
    bind();
    S.datasets = collectDatasets();
    if (!S.datasets.length) { el.bootText.textContent = '未找到数据清单 index.js'; return; }
    if (el.dsSel) {
      S.datasets.forEach(function (ds) {
        var o = document.createElement('option');
        o.value = ds.id;
        o.textContent = ds.name;
        el.dsSel.appendChild(o);
      });
    }
    selectDataset(S.datasets[0].id);
    requestAnimationFrame(loop);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
