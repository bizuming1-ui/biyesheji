/* v10 连接层离线仿真:  node tests/link_sim.js
   用假的 window/fetch 验证 web/link.js:
   ① 指示灯分级(25s/60s);② 单次失败不切线;③ 连续 ≥4 次失败且备用线路通 → 自动切换并给 fetch 加前缀;
   ④ 备用线路也不通 → 不切;⑤ 切换防抖;⑥ 轮询退避/降频;⑦ 相对路径以外的 URL 不被改写。 */
const assert = require("assert");
let alive = { "https://main.example": true, "https://alt1.example": false, "https://alt2.example": true };
const calls = [];
global.window = global;
global.location = { origin: "https://main.example" };
global.localStorage = { _d: {}, getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = v; } };
global.fetch = async (u) => {
  calls.push(u);
  const origin = u.replace(/(https:\/\/[^/]+).*/, "$1");
  if (!alive[origin]) throw new Error("net down");
  return { ok: true };
};
const notes = [];
global.__linkNotify = (m) => notes.push(m);
const L = require("../web/link.js");

(async () => {
  let pass = 0; const ok = (n, c) => { assert(c, n); console.log("PASS " + n); pass++; };
  L.remember({ alts: ["https://alt1.example/", "https://alt2.example"] });
  ok("remember strips trailing slash & persists", L.alts[0] === "https://alt1.example" && JSON.parse(localStorage.getItem("vc_alts")).length === 2);
  L.ok(300); ok("fresh = ok", L.level() === "ok" && L.rtt === 300);
  L.lastOk = Date.now() - 30000; ok("30s idle = warn", L.level() === "warn");
  L.lastOk = Date.now() - 70000; ok("70s idle = err", L.level() === "err");
  L.ok(); L.fails = 3;
  ok("3 failures: no failover", (await L.maybeFailover()) === false && L.base === "");
  L.fails = 4; alive["https://main.example"] = false;
  ok("4 failures + alt2 alive: switches (skips dead alt1)", (await L.maybeFailover()) === true && L.base === "https://alt2.example" && L.switched === 1);
  ok("notify called", notes.length === 1 && /alt2/.test(notes[0]));
  calls.length = 0; await window.fetch("/api/ping");
  ok("relative url gets alt prefix", calls[0] === "https://alt2.example/api/ping");
  calls.length = 0; await window.fetch("https://alt2.example/api/x");
  ok("absolute url untouched", calls[0] === "https://alt2.example/api/x");
  L.fails = 9;
  ok("debounce: no second switch within 20s", (await L.maybeFailover()) === false);
  L.lastSwitchAt = 0; alive["https://alt2.example"] = false;
  ok("all lines dead: stays", (await L.maybeFailover()) === false && L.base === "https://alt2.example");
  alive["https://main.example"] = true; L.fails = 5;
  ok("main recovered: switches back to origin", (await L.maybeFailover()) === true && L.base === "");
  ok("backoff grows and is capped", L.nextGap(1) === 1500 && L.nextGap(3) > L.nextGap(2) && L.nextGap(20) === 8000);
  ok("idle gaps: visible 3s / hidden 10s / chat 8s", L.nextGap(0, false, 0) === 3000 && L.nextGap(0, true, 0) === 10000 && L.nextGap(0, false, 1) === 8000);
  console.log("\nALL PASSED (" + pass + ")");
})().catch((e) => { console.error("FAIL", e.message); process.exit(1); });
