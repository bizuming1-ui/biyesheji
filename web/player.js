/* 无缝播放器 v7 —— 自适应预缓冲 + 采样级拼接。
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

  class GaplessPlayer {
    constructor(opts) {
      opts = opts || {};
      this.makeCtx = opts.makeCtx || function () { return new (root.AudioContext || root.webkitAudioContext)({ latencyHint: "playback" }); };
      this.clock = opts.clock || function () { return performance.now() / 1000; };
      this.setTimer = opts.setInterval || function (f, ms) { return root.setInterval(f, ms); };
      this.clearTimer = opts.clearInterval || function (h) { return root.clearInterval(h); };
      this.ctx = null;
      this.smooth = true;          // 「无停顿优先」开关
      this.minStart = 0.5;         // 最小起播缓冲(秒)
      this.margin = 0.35;          // 抖动余量(句间衔接、网络长轮询的批量到达)
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
      if (!this.firstArrive) { this.firstArrive = now; this.firstDur = dur; }
      else if (now - this.lastArrive > this.maxGap) this.maxGap = Math.min(4, now - this.lastArrive);
      this.lastArrive = now;
      this.arrivedS += dur;
      this.pump();
    }

    /* ---------- 度量 ---------- */
    contiguousS() { let d = 0, i = this.expect; while (this.pending.has(i)) { d += this.pending.get(i).dur; i++; } return d; }
    rate() {
      // 生产速率:不含第一块(它到达的瞬间还没有「经过的时间」)
      const el = this.clock() - this.firstArrive;
      if (!this.firstArrive || el < 0.3) return 1.0;
      return Math.max(0.15, (this.arrivedS - this.firstDur) / el);
    }
    allArrived() { return this.total !== null && this.expect + this.pending.size >= this.total; }
    needS() {
      if (!this.smooth) return 0.2;
      if (this.allArrived()) return 0;
      const r = this.rate() * 0.85;   // 15% 安全系数:句间衔接/调度抖动会让「后面」的实际速率比「已观测」的低
      // 还没生产出的音频:服务端预计总时长 − 已到达;文本还没流完(没有 final)时再加一段未知尾巴
      const known = this.total !== null || this.planFinal;
      let eRem = Math.max(0, this.estTotal - this.arrivedS) * 1.15;
      if (!known) eRem += Math.min(20, Math.max(2, 0.3 * this.estTotal));
      let need = this.minStart + this.boost;
      if (r < 1.0 && eRem > 0) need = Math.max(need, eRem * (1 / r - 1) + this.margin + this.boost);
      else need = Math.max(need, this.margin + this.boost);
      need += Math.min(2.5, this.maxGap * 1.1);          // 已观测到的最长「断料」间隔
      if (this.holdAfterStall) need = Math.max(need, this.holdAfterStall);  // 断流后的滞回:必须比上次更充裕,杜绝反复起停
      return Math.min(this.cap, need);
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
    }
    sched(c) {
      const ctx = this.ctx;
      const buf = ctx.createBuffer(1, c.f32.length, c.sr);
      buf.copyToChannel(c.f32, 0);
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
      if (this.total !== null && this.expect >= this.total && this.srcs.size === 0) {
        this.running = false;
        this._state("idle");
        if (this.underruns === 0) this.boost = Math.max(0, this.boost - 0.2);
      }
    }
  }

  root.GaplessPlayer = GaplessPlayer;
  if (typeof module !== "undefined" && module.exports) module.exports = GaplessPlayer;
})(typeof window !== "undefined" ? window : globalThis);
