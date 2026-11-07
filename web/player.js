/* 无缝播放器 v8 —— 弹性变速(WSOLA,变速不变调) + 自适应预缓冲 + 采样级拼接。
 *
 * v8 相对 v7 的根本改变(批判:v7 把「不断流」完全押在预缓冲上,合成速率≈实时就要先等 10 秒以上,实测首声 19.2s):
 *   不断流只要求「消耗速度 ≤ 生产速度」。v8 在缓冲不足时把消耗速度 q 轻微降到 [0.87, 1](音高/音色不变,见 wsola.js),
 *   缓冲充足就回到 1.0(此时逐采样精确重建,无任何处理痕迹)。
 *   所需预缓冲由 B ≥ E_rem·(1/r − 1) 降为 B ≥ E_rem·(q_min/r − 1):r≥0.87 时不再需要预缓冲 → 第一块到达即开播。
 *   r<0.87(真的太慢)才回退到预缓冲;真缓冲见底仍只暂停一次并加大余量(v7 逻辑保留)。
 *
 * v7 说明:
 *
 * 为什么以前会「播放-停止-播放-停止」?
 *   旧版一旦缓冲耗尽就只多等 0.7 秒再播,而合成速度如果低于播放速度(RTF>1),
 *   每次都是「刚攒够一点就播 → 很快又空 → 再攒」,于是出现反复停顿。
 *
 * 新算法(第一性原理:不断流 ⇔ 任意时刻 已缓冲时长 + 之后生产的时长 ≥ 之后播放消耗的时长):
 *   r      = 实测生产速率(已到达音频秒数 / 经过的墙钟秒数)
 *   E_rem  = 预计还没生产出来的音频秒数 = 服务端 plan 事件给的预计总时长 − 已到达时长
 *   若 r ≥ 1:      只需很小的抖动余量。
 *   若 r < 1:      需要预缓冲  B ≥ E_rem · (1/r − 1)   才能保证从开播到结束一次都不断。
 *   开播门槛 = clamp(B + 余量, 最小起播量, 上限);一旦开播,后到的块在「上一块结束的那个采样」精确接上(无缝)。
 *   如果上限不够/估计偏差导致真的缓冲见底:不会「播一点停一点」,而是暂停一次,
 *   按最新速率重新算门槛(带滞回、并永久加大本页的余量),攒够后一口气播完。
 */
(function (root) {
  "use strict";
  const WsolaCls = root.Wsola || (typeof require !== "undefined" ? require("./wsola.js") : null);

  class GaplessPlayer {
    constructor(opts) {
      opts = opts || {};
      this.makeCtx = opts.makeCtx || function () { return new (root.AudioContext || root.webkitAudioContext)({ latencyHint: "playback" }); };
      this.clock = opts.clock || function () { return performance.now() / 1000; };
      this.setTimer = opts.setInterval || function (f, ms) { return root.setInterval(f, ms); };
      this.clearTimer = opts.clearInterval || function (h) { return root.clearInterval(h); };
      this.ctx = null;
      this.smooth = true;          // 「无停顿优先」开关
      this.minStart = 0.45;        // 最小起播缓冲(秒)
      this.margin = 0.3;           // 抖动余量(句间衔接、网络抖动)
      this.stretch = !!WsolaCls;   // 弹性变速开关(缓冲不足时轻微放慢,音高不变)
      this.qMin = 0.87;            // 最慢播放速度(再慢就改用预缓冲)
      this.prior = 0.95;           // 合成速率先验(秒音频/秒墙钟),前 1~2 秒用它与实测加权
      this.cap = 14;               // 预缓冲上限(秒):再多就让用户等太久
      this.maxWait = 16;           // 最长等待(秒)后无论如何开播
      this.boost = 0;              // 发生过真正断流后永久增加的余量
      this.onstate = function () {};
      this.onfirst = function () {};
      this.timer = null;
      this.reset();
    }

    ensure() {
      if (!this.ctx) {
        this.ctx = this.makeCtx();
        this.gain = this.ctx.createGain();
        this.an = this.ctx.createAnalyser();
        this.an.fftSize = 128;
        this.gain.connect(this.an);
        this.an.connect(this.ctx.destination);
      }
      if (this.ctx.state !== "running" && this.ctx.resume) this.ctx.resume();
      if (!this.timer) this.timer = this.setTimer(() => this.tick(), 40);
    }

    reset() {
      this.turn = -1;
      this.pending = new Map();
      this.expect = 0;
      this.total = null;
      this.srcs = new Set();
      this.running = false;
      this.nextTime = 0;
      this.arrivedS = 0;
      this.firstArrive = 0;
      this.firstDur = 0;
      this.waitSince = 0;
      this.estTotal = 0;
      this.planFinal = false;
      this.lastArrive = 0;
      this.maxGap = 0;
      this.stallStart = 0;
      this.holdAfterStall = 0;
      this.underruns = 0;
      this.gapTotal = 0;
      this.gaps = [];
      this.firstDone = false;
      this.prebufferS = 0;
      this.state = "idle";
      this.stretcher = null;
      this.q = 1;
      this.flushed = false;
      this.qMinUsed = 1;
      this.firstArriveAt = 0;
    }

    newTurn(turn) { this.stop(); this.turn = turn; }

    stop() {
      for (const s of this.srcs) { try { s.onended = null; s.stop(); } catch (e) { /* 已结束 */ } }
      this.reset();
      this._state("idle");
    }

    _state(s) { if (s !== this.state) { this.state = s; this.onstate(s); } }

    /* ---------- 输入 ---------- */
    plan(estS, final) { if (estS > this.estTotal) this.estTotal = estS; if (final) this.planFinal = true; this.pump(); }
    finish(total) { this.total = total; this.pump(); this.check(); }
    push(turn, idx, sr, f32) {
      if (turn !== this.turn) return;
      const dur = f32.length / sr;
      this.pending.set(idx, { sr: sr, f32: f32, dur: dur });
      const now = this.clock();
      if (!this.firstArrive) { this.firstArrive = now; this.firstDur = dur; this.firstArriveAt = now * 1000; }
      else if (now - this.lastArrive > this.maxGap) this.maxGap = Math.min(4, now - this.lastArrive);
      this.lastArrive = now;
      this.arrivedS += dur;
      this.pump();
    }

    /* ---------- 度量 ---------- */
    contiguousS() { let d = 0, i = this.expect; while (this.pending.has(i)) { d += this.pending.get(i).dur; i++; } return d; }
    rate() {
      // 生产速率:不含第一块(它到达的瞬间还没有「经过的时间」);前期与先验加权,避免「刚到第一块 → 速率=1.0」的乐观误判
      if (!this.firstArrive) return this.prior;
      const el = Math.max(0, this.clock() - this.firstArrive), w = 1.2;
      return Math.max(0.15, (this.prior * w + (this.arrivedS - this.firstDur)) / (w + el));
    }
    stretchOn() { return this.stretch && this.smooth && !!WsolaCls; }
    pendingS() { let d = 0; for (const c of this.pending.values()) d += c.dur; return d; }
    eRemS() {   // 预计还没生产出来的音频秒数
      const known = this.total !== null || this.planFinal;
      let e = Math.max(0, this.estTotal - this.arrivedS) * 1.15;
      if (!known) e += Math.min(20, Math.max(2, 0.3 * this.estTotal));
      return e;
    }
    allArrived() { return this.total !== null && this.expect + this.pending.size >= this.total; }
    needS() {
      if (!this.smooth) return 0.2;
      if (this.allArrived()) return 0;
      const r = this.rate() * 0.9;    // 10% 安全系数:句间衔接/调度抖动会让「后面」的实际速率比「已观测」的低
      const qf = this.stretchOn() ? this.qMin : 1;   // 能放慢到 qMin,所以只需补足 qMin 与 r 之间的差
      const eRem = this.eRemS();
      let need = this.minStart + this.boost;
      if (r < qf && eRem > 0) need = Math.max(need, eRem * (qf / r - 1) + this.margin + this.boost);
      need += Math.min(2.5, this.maxGap * 1.1);          // 已观测到的最长「断料」间隔
      if (this.holdAfterStall) need = Math.max(need, this.holdAfterStall);  // 断流后的滞回:必须比上次更充裕,杜绝反复起停
      return Math.min(this.cap, need);
    }
    /* 本块的播放速度 q:使 (B+E_rem)/q ≥ E_rem/r(剩余音频在生产完之前不会被播完) */
    pickQ() {
      if (!this.stretchOn() || this.allArrived()) return 1;
      const eRem = this.eRemS();
      if (eRem < 0.3) return 1;
      const r = this.rate() * 0.9;
      const lead = this.running ? Math.max(0, this.nextTime - this.ctx.currentTime) : 0;
      const B = lead * this.q + this.pendingS();
      let target = r * (B + eRem) / eRem;
      target = Math.max(this.qMin, Math.min(1, target));
      // 限幅变化率:下降快(防断流)、回升慢(听感平滑)
      const q = target < this.q ? Math.max(target, this.q - 0.05) : Math.min(target, this.q + 0.02);
      return q >= 0.995 ? 1 : q;
    }
    gateOpen() {
      const b = this.contiguousS();
      if (b <= 0) return false;
      if (this.allArrived()) return true;
      if (b >= this.needS()) return true;
      if (this.waitSince && this.clock() - this.waitSince > this.maxWait) return true;
      return false;
    }

    /* ---------- 调度 ---------- */
    pump() {
      if (!this.ctx) return;
      if (this.running) { this.flush(); return; }
      if (!this.pending.has(this.expect)) return;
      if (!this.waitSince) this.waitSince = this.clock();
      if (this.gateOpen()) {
        this.prebufferS = this.contiguousS();
        if (this.stallStart) { const d = this.clock() - this.stallStart; this.gapTotal += d; this.gaps.push(d); this.stallStart = 0; }
        this.running = true;
        this.waitSince = 0;
        this.nextTime = this.ctx.currentTime + 0.04;
        this.flush();
      } else {
        this._state("buffering");
      }
    }
    flush() {
      while (this.pending.has(this.expect)) {
        const c = this.pending.get(this.expect);
        this.pending.delete(this.expect);
        this.expect++;
        this.sched(c);
      }
      this.flushTail();
    }
    sched(c) {
      let f32 = c.f32;
      if (this.stretchOn()) {
        if (!this.stretcher) this.stretcher = new WsolaCls(c.sr);
        const q = this.pickQ();
        this.q = q;
        if (q < this.qMinUsed) this.qMinUsed = q;
        f32 = this.stretcher.process(c.f32, q);
      }
      this.emit(f32, c.sr);
    }
    emit(f32, sr) {
      if (!f32.length) return;
      const ctx = this.ctx;
      const buf = ctx.createBuffer(1, f32.length, sr);
      buf.copyToChannel(f32, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.gain);
      const earliest = ctx.currentTime + 0.01;
      const at = Math.max(this.nextTime, earliest);
      if (this.firstDone && at - this.nextTime > 0.002) { this.gapTotal += at - this.nextTime; this.gaps.push(at - this.nextTime); }
      src.start(at);
      this.nextTime = at + buf.duration;
      this.srcs.add(src);
      src.onended = () => { this.srcs.delete(src); this.check(); };
      this._state("playing");
      if (!this.firstDone) { this.firstDone = true; this.onfirst(this.clock() * 1000, at - ctx.currentTime); }
    }
    flushTail() {   // 一轮的最后一块已全部排入 → 把变速器里残留的尾巴也播出来
      if (this.flushed || !this.stretcher || this.total === null || this.expect < this.total) return;
      this.flushed = true;
      this.emit(this.stretcher.flush(), this.stretcher.sr);
    }

    tick() {
      if (!this.ctx) return;
      if (this.running) {
        this.flush();
        const lead = this.nextTime - this.ctx.currentTime;
        const finished = this.total !== null && this.expect >= this.total;
        if (lead < 0.006 && !finished) {
          // 缓冲真的见底了:不再「来一块播一块」,暂停一次,按最新速率重新攒够再播
          this.running = false;
          this.waitSince = this.clock();
          this.stallStart = this.clock();
          this.holdAfterStall = Math.min(this.cap, Math.max(2.0, this.prebufferS * 1.6 + 0.8));
          this.underruns++;
          this.boost = Math.min(5, this.boost + 0.8);
          this._state("buffering");
        }
        this.check();
      } else {
        this.pump();
      }
    }

    check() {
      if (this.total !== null && this.expect >= this.total && this.srcs.size === 0 && (this.flushed || !this.stretcher)) {
        this.running = false;
        this._state("idle");
        if (this.underruns === 0) this.boost = Math.max(0, this.boost - 0.2);
      }
    }
  }

  root.GaplessPlayer = GaplessPlayer;
  if (typeof module !== "undefined" && module.exports) module.exports = GaplessPlayer;
})(typeof window !== "undefined" ? window : globalThis);
