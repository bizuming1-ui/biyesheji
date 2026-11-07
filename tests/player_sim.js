/* 无缝播放器离线仿真(不需要浏览器/GPU):  node tests/player_sim.js
 * 用假 AudioContext + 假时钟,按不同的「合成速度 / 句间停顿 / 估计误差」喂数据,
 * 统计:首声时间、开播后的断流次数、总断流时长。要求:服务端速度 ≥ 实时的场景必须 0 次断流。 */
const GaplessPlayer = require("../web/player.js");

function makeEnv() {
  const env = { t: 0, srcs: [] };
  env.ctx = {
    get currentTime() { return env.t; },
    state: "running",
    destination: {},
    resume() {},
    createGain() { return { connect() {} }; },
    createAnalyser() { return { fftSize: 0, connect() {} }; },
    createBuffer(ch, len, sr) { return { duration: len / sr, copyToChannel() {} }; },
    createBufferSource() {
      const s = { buffer: null, connect() {}, start(at) { s.at = at; env.srcs.push(s); }, stop() {}, onended: null };
      return s;
    },
  };
  return env;
}

/* 场景: chunks = [{t: 到达时间, dur: 秒}] */
function run(name, chunks, estTotal, opts) {
  opts = opts || {};
  const env = makeEnv();
  const p = new GaplessPlayer({ makeCtx: () => env.ctx, clock: () => env.t, setInterval: () => 1, clearInterval() {} });
  if (opts.boost) p.boost = opts.boost;
  p.ensure(); p.newTurn(1);
  const sr = 24000;
  let i = 0, firstSound = null, planSent = false;
  const END = chunks[chunks.length - 1].t + chunks.reduce((a, c) => a + c.dur, 0) + 60;
  const total = chunks.length;
  p.onfirst = () => { if (firstSound === null) firstSound = env.t; };
  for (env.t = 0; env.t < END; env.t = Math.round((env.t + 0.01) * 1000) / 1000) {
    while (i < chunks.length && chunks[i].t <= env.t) {
      if (!planSent && estTotal) { p.plan(estTotal, true); planSent = true; }
      p.push(1, i, sr, new Float32Array(Math.round(chunks[i].dur * sr)));
      i++;
      if (i === chunks.length) p.finish(total);
    }
    for (const s of env.srcs) if (!s.done && s.at + s.buffer.duration <= env.t) { s.done = true; if (s.onended) s.onended(); }
    p.tick();
    if (p.state === "idle" && i === chunks.length && p.expect >= total && firstSound !== null) break;
  }
  const r = { name, first: firstSound === null ? null : +firstSound.toFixed(2), prebuf: +p.prebufferS.toFixed(2), underruns: p.underruns, gapTotal: +p.gapTotal.toFixed(2), played: p.expect === total };
  return r;
}

/* 生成「按速率 rate(秒音频/秒墙钟)生产」的块序列;每 sentenceLen 秒的音频后有一次 stall 秒的停顿(模拟句间/双卡调度) */
function gen(totalAudio, rate, t0, chunkDur, stall, sentenceLen) {
  const out = []; let t = t0, made = 0, sinceStall = 0;
  while (made < totalAudio - 1e-6) {
    const d = Math.min(chunkDur, totalAudio - made);
    t += d / rate; out.push({ t: +t.toFixed(3), dur: d }); made += d; sinceStall += d;
    if (stall && sinceStall >= sentenceLen) { t += stall; sinceStall = 0; }
  }
  return out;
}

const results = [];
const fails = [];
function expectNoGap(r) { results.push(r); console.log(JSON.stringify(r)); if (r.underruns !== 0 || r.gapTotal > 0.005 || !r.played) fails.push(r.name); }
function expectOk(r, maxUnder) { results.push(r); console.log(JSON.stringify(r)); if (r.underruns > maxUnder || !r.played) fails.push(r.name); }

// 1) 合成比实时快 3 倍(RTF≈0.33,Mock 的情况),长文本 60 秒
expectNoGap(run("快速合成 3x / 60s", gen(60, 3, 1.0, 0.33, 0, 99), 60));
// 2) 刚好实时(RTF=1.0)+ 每 4 秒音频有 0.4 秒停顿(句间衔接)
expectNoGap(run("1.0x + 句间停顿0.4s / 60s", gen(60, 1.0, 1.0, 0.33, 0.4, 4), 60));
// 3) 比实时慢(RTF=1.1 → 速率 0.91)+ 句间停顿:必须靠预缓冲保证不断
expectNoGap(run("0.91x(RTF 1.1)+ 停顿 / 60s", gen(60, 0.91, 1.0, 0.33, 0.3, 5), 60));
// 4) RTF 1.3(速率 0.77),估计时长准确
expectNoGap(run("0.77x(RTF 1.3)/ 40s", gen(40, 0.77, 1.0, 0.33, 0, 99), 40));
// 5) 估计偏低 25%(服务端语速估计不准):允许至多 1 次「暂停重缓冲」,但不能反复起停
expectOk(run("0.8x + 估计偏低25% / 60s", gen(60, 0.8, 1.0, 0.33, 0.2, 5), 45), 1);
// 6) 双卡交替:速率 2.0,每 3 秒音频有 0.8 秒的调度停顿
expectNoGap(run("2.0x + 停顿0.8s / 90s", gen(90, 2.0, 1.2, 0.33, 0.8, 3), 90));
// 7) 长文本(180 秒)速率 0.95 —— 预缓冲上限内可能无法 100% 避免,但断流次数必须 ≤ 2 且不能反复起停
expectOk(run("0.95x / 180s 长文本", gen(180, 0.95, 1.0, 0.33, 0, 99), 180), 2);
// 8) 短句(2 秒),首声应在第一块到达后很快开始
const s8 = run("短句首声", gen(2, 3, 0.8, 0.33, 0, 99), 2);
results.push(s8); console.log(JSON.stringify(s8));
if (s8.first === null || s8.first > 1.6 || s8.underruns) fails.push("短句首声");

console.log(fails.length ? "FAILED: " + fails.join(" | ") : "ALL PASS (" + results.length + " scenarios)");
process.exit(fails.length ? 1 : 0);
