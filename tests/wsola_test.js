/* WSOLA 自检:  node tests/wsola_test.js
 * 1) q=1 逐采样精确重建  2) q=0.9 时输出时长≈输入/0.9  3) 音高不变(自相关周期)  4) 无不连续(最大相邻采样差)  5) 分块输入 == 整块输入 */
const Wsola = require("../web/wsola.js");
const sr = 24000;
function voice(sec, f0) {   // 类语音:基频 f0 + 谐波,缓慢颤音和幅度调制
  const n = Math.round(sec * sr), x = new Float32Array(n); let ph = 0;
  for (let i = 0; i < n; i++) {
    const t = i / sr, f = f0 * (1 + 0.03 * Math.sin(2 * Math.PI * 5 * t)); ph += (2 * Math.PI * f) / sr;
    let v = 0; for (let h = 1; h <= 8; h++) v += Math.sin(h * ph) / h;
    x[i] = 0.25 * v * (0.6 + 0.4 * Math.sin(2 * Math.PI * 3 * t));
  }
  return x;
}
function run(x, q, chunk) {
  const w = new Wsola(sr), parts = [];
  for (let i = 0; i < x.length; i += chunk) parts.push(w.process(x.subarray(i, Math.min(x.length, i + chunk)), q));
  parts.push(w.flush());
  let n = 0; for (const p of parts) n += p.length;
  const o = new Float32Array(n); let k = 0; for (const p of parts) { o.set(p, k); k += p.length; }
  return o;
}
function f0est(x, from, len) {   // 自相关估计基频
  let bestLag = 0, best = -1;
  for (let lag = Math.round(sr / 400); lag < Math.round(sr / 80); lag++) {
    let c = 0, e1 = 1e-9, e2 = 1e-9;
    for (let i = 0; i < len; i++) { const a = x[from + i], b = x[from + i + lag]; c += a * b; e1 += a * a; e2 += b * b; }
    const s = c / Math.sqrt(e1 * e2); if (s > best) { best = s; bestLag = lag; }
  }
  return sr / bestLag;
}
const fails = [];
const ck = (name, ok, extra) => { console.log((ok ? "PASS " : "FAIL ") + name + (extra ? " -> " + extra : "")); if (!ok) fails.push(name); };

const x = voice(6, 140);
// 1) q=1 精确重建(跳过首个 Hs 淡入)
const y1 = run(x, 1.0, 8000);
let maxErr = 0; for (let i = 400; i < x.length - 400; i++) maxErr = Math.max(maxErr, Math.abs(y1[i] - x[i]));
ck("q=1 精确重建", maxErr < 1e-4, "maxErr=" + maxErr.toExponential(2) + " len " + y1.length + "/" + x.length);
// 2/3/4) q=0.9, 0.87
for (const q of [0.95, 0.9, 0.87]) {
  const y = run(x, q, 8000), expect = x.length / q;
  ck("q=" + q + " 时长", Math.abs(y.length - expect) / expect < 0.02, y.length + " vs " + Math.round(expect));
  const a = f0est(x, 30000, 2400), b = f0est(y, Math.round(30000 / q), 2400);
  ck("q=" + q + " 音高不变", Math.abs(a - b) / a < 0.06, a.toFixed(1) + "Hz → " + b.toFixed(1) + "Hz");
  let md = 0, mx = 0; for (let i = 1; i < y.length; i++) { md = Math.max(md, Math.abs(y[i] - y[i - 1])); }
  for (let i = 1; i < x.length; i++) mx = Math.max(mx, Math.abs(x[i] - x[i - 1]));
  ck("q=" + q + " 无不连续(相邻采样差不超过原信号 1.3 倍)", md <= mx * 1.3, md.toFixed(4) + " vs " + mx.toFixed(4));
}
// 5) 分块无关性
const ya = run(x, 0.9, 300), yb = run(x, 0.9, 20000);
let diff = 0; const L = Math.min(ya.length, yb.length); for (let i = 0; i < L; i++) diff = Math.max(diff, Math.abs(ya[i] - yb[i]));
ck("分块无关(300 vs 20000 采样一块)", diff < 1e-5 && Math.abs(ya.length - yb.length) <= 2, "diff=" + diff.toExponential(2) + " len " + ya.length + "/" + yb.length);
// 6) 变速中途切回 1.0 的连续性
{
  const w = new Wsola(sr), parts = []; const qs = [0.88, 0.88, 0.92, 1, 1, 1, 0.9, 1];
  for (let i = 0; i < qs.length; i++) parts.push(w.process(x.subarray(i * 12000, (i + 1) * 12000), qs[i]));
  parts.push(w.flush());
  let n = 0; for (const p of parts) n += p.length; const o = new Float32Array(n); let k = 0; for (const p of parts) { o.set(p, k); k += p.length; }
  let md = 0, mx = 0; for (let i = 1; i < o.length; i++) md = Math.max(md, Math.abs(o[i] - o[i - 1]));
  for (let i = 1; i < x.length; i++) mx = Math.max(mx, Math.abs(x[i] - x[i - 1]));
  ck("变速切换无爆音", md <= mx * 1.3, md.toFixed(4) + " vs " + mx.toFixed(4));
}
console.log(fails.length ? "FAILED: " + fails.join(" | ") : "ALL PASS");
process.exit(fails.length ? 1 : 0);
