/* 流式 WSOLA 变速不变调(v8)—— 「首声最快」与「绝不断流」同时成立的关键。
 *
 * 第一性原理:播放不断流 ⇔ 已缓冲 + 之后生产的 ≥ 之后消耗的。若合成速率 r 略低于 1,
 *   旧做法 = 先攒 B ≥ E_rem·(1/r − 1) 秒再开播(几秒到十几秒的首声延迟);
 *   新做法 = 马上开播,并把「消耗速度」q 轻微降到 r 附近(q∈[0.87,1],音高/音色不变),缓冲不够时才慢,够了立刻回到 1.0。
 *   q=1 且无状态漂移时 WSOLA 是「逐采样精确重建」(周期 Hann 50% 重叠求和为 1,搜索会选中自然延续位置),
 *   所以只在确实需要时才有任何处理痕迹,且克隆音色(谱包络)不受影响。
 *
 * 用法: const s = new Wsola(sr);  out = s.process(f32, q);  ... ;  tail = s.flush();
 */
(function (root) {
  "use strict";

  class Wsola {
    constructor(sr) {
      this.sr = sr;
      this.N = 2 * Math.round(0.016 * sr);          // 帧长 ≈32ms(偶数)
      this.Hs = this.N / 2;                          // 合成跳距
      this.tol = Math.round(0.006 * sr);             // 搜索范围 ±6ms(> 男声基频周期的一半,足够对齐相位)
      this.win = new Float32Array(this.N);
      for (let n = 0; n < this.N; n++) this.win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / this.N);  // 周期 Hann
      this.reset();
    }
    reset() {
      this.buf = new Float32Array(0);   // 输入缓冲(绝对下标 = base + 局部下标)
      this.base = 0;
      this.nominal = 0;                 // 下一帧的名义输入位置(绝对)
      this.prevPos = -1;                // 上一帧实际输入位置(绝对)
      this.acc = new Float32Array(this.N);  // 重叠相加累加器
      this.frames = 0;
      this.inTotal = 0;
      this.outTotal = 0;
    }
    _append(x) {
      const nb = new Float32Array(this.buf.length + x.length);
      nb.set(this.buf, 0); nb.set(x, this.buf.length);
      this.buf = nb; this.inTotal += x.length;
    }
    _trim() {   // 丢弃不再需要的旧输入
      const keepFrom = Math.max(0, Math.floor(Math.min(this.nominal - this.tol, this.prevPos + this.Hs)) - 2);
      const drop = keepFrom - this.base;
      if (drop > 4096) { this.buf = this.buf.slice(drop); this.base += drop; }
    }
    _bestPos(nominal) {
      const Hs = this.Hs, tol = this.tol, buf = this.buf, base = this.base;
      const refStart = this.prevPos + Hs - base;      // 上一帧后半段「自然延续」的输入
      let lo = Math.max(base, Math.round(nominal) - tol), hi = Math.round(nominal) + tol;
      if (lo > hi) lo = hi;
      let best = Math.round(nominal), bestScore = -Infinity;
      // 先检查「自然延续」位置本身(q=1 时相关为 1,保证精确重建)
      const cands = [];
      const nat = this.prevPos + Hs;
      if (nat >= lo && nat <= hi) cands.push(nat);
      for (let p = lo; p <= hi; p++) cands.push(p);
      const step = 1;
      let refE = 0;
      for (let i = 0; i < Hs; i++) { const v = buf[refStart + i]; refE += v * v; }
      for (let ci = 0; ci < cands.length; ci += step) {
        const p = cands[ci], o = p - base;
        let c = 0, e = 1e-9;
        for (let i = 0; i < Hs; i++) { const v = buf[o + i]; c += v * buf[refStart + i]; e += v * v; }
        const score = refE > 1e-9 ? c / Math.sqrt(e * refE + 1e-12) : (p === Math.round(nominal) ? 1 : 0);
        if (score > bestScore + 1e-7) { bestScore = score; best = p; }   // 并列时保留靠前者(自然延续优先)
      }
      return best;
    }
    _frames(q, final) {
      const N = this.N, Hs = this.Hs, out = [];
      for (;;) {
        const lastNeeded = Math.round(this.nominal) + this.tol + N;
        if (this.base + this.buf.length < lastNeeded) break;
        if (final && Math.round(this.nominal) >= this.inTotal) break;
        let pos;
        if (this.frames === 0) pos = 0;
        else pos = this._bestPos(this.nominal);
        const o = pos - this.base;
        for (let n = 0; n < N; n++) this.acc[n] += this.buf[o + n] * this.win[n];
        out.push(this.acc.slice(0, Hs));
        this.acc.copyWithin(0, Hs); this.acc.fill(0, Hs);
        this.prevPos = pos;
        this.nominal += Hs * q;
        this.frames++;
        if (this.frames === 1) this.nominal = Hs * q;  // 第一帧在 0,第二帧名义位置 = Ha
      }
      return out;
    }
    _join(parts) {
      let n = 0; for (const p of parts) n += p.length;
      const o = new Float32Array(n); let k = 0;
      for (const p of parts) { o.set(p, k); k += p.length; }
      this.outTotal += n;
      return o;
    }
    /** 输入一块音频,q = 播放速度(0.5~1.0:1=原速,0.9=慢 10%);返回已能确定的输出(可能为空)。 */
    process(x, q) {
      q = Math.max(0.5, Math.min(1.5, q || 1));
      this._append(x);
      const parts = this._frames(q, false);
      this._trim();
      return this._join(parts);
    }
    /** 句末/对话结束:补零把剩余输入全部输出,并让尾部淡出。 */
    flush() {
      this._append(new Float32Array(this.N + this.tol + 2));
      const parts = this._frames(1, true);
      parts.push(this.acc.slice(0, this.Hs));
      this.acc.fill(0);
      const o = this._join(parts);
      const real = Math.round(this.inTotal - (this.N + this.tol + 2));
      return o;
    }
  }

  root.Wsola = Wsola;
  if (typeof module !== "undefined" && module.exports) module.exports = Wsola;
})(typeof window !== "undefined" ? window : globalThis);
