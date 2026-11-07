/* v10 连接层(Link):让「连接指示灯」只反映真实的连通性,并在主线路卡死时自动切到备用线路。

第一性原理(来自 logs/run-log-20261107-143559.txt 的真实数据):
  /api/ping 失败 36/65、/api/status 失败 56/139,几乎全是前端自己 6~8 秒的超时(signal is aborted)。
  旧版同时有 4 个互不等待的定时器(ping 5s / status 3s / job 0.8s / diag 3s),上一个请求还没回来就发下一个,
  在 RTT≈400ms 的公网隧道上形成「请求风暴」→ 互相拖慢 → 超时 → 误判「重连中」。
  现在:
    1. 状态请求合并成一个 /api/state,串行(上一个结束才发下一个),超时 20s,失败指数退避;
    2. 「已连接」= 最近 25 秒内有任何一次成功(含对话流里的数据),单次超时不再改变指示灯;
    3. 连续失败 ≥4 次 → 探测备用线路(服务端通过 /api/state 下发 alts),通即无缝切换;
    4. 所有相对路径的 fetch('/api/...') 自动加上当前线路前缀(服务端已开放 CORS)。
*/
(function (root, factory) {
  var L = factory(root);
  if (typeof module === "object" && module.exports) module.exports = L;
})(typeof window !== "undefined" ? window : globalThis, function (W) {
  var nativeFetch = W.fetch ? W.fetch.bind(W) : null;
  var L = (W.__LINK = {
    base: "",            // "" = 当前页面所在线路;否则是备用线路的 origin
    origin0: (W.location && W.location.origin) || "",
    alts: [],
    fails: 0,
    lastOk: Date.now(),
    rtt: 0,
    switched: 0,
    probing: false,
    lastSwitchAt: 0,
  });
  try { L.alts = JSON.parse(W.localStorage.getItem("vc_alts") || "[]"); } catch (e) { L.alts = []; }

  function say(msg) { try { if (typeof W.__linkNotify === "function") W.__linkNotify(msg); } catch (e) {} }

  W.fetch = function (u, o) {
    if (typeof u === "string" && u.charAt(0) === "/" && u.charAt(1) !== "/" && L.base) u = L.base + u;
    return nativeFetch(u, o);
  };

  L.remember = function (j) {
    if (!j || !Array.isArray(j.alts)) return;
    var list = j.alts.filter(function (x) { return typeof x === "string" && /^https?:\/\//.test(x); }).map(function (x) { return x.replace(/\/$/, ""); });
    L.alts = list;
    try { W.localStorage.setItem("vc_alts", JSON.stringify(list)); } catch (e) {}
  };

  L.ok = function (ms) {
    L.lastOk = Date.now(); L.fails = 0;
    if (ms != null) L.rtt = L.rtt ? Math.round(L.rtt * 0.7 + ms * 0.3) : Math.round(ms);
  };

  L.idleMs = function () { return Date.now() - L.lastOk; };
  /* ok:25 秒内有成功 / warn:25~60 秒 / err:超过 60 秒没有任何成功 */
  L.level = function () { var i = L.idleMs(); return i < 25000 ? "ok" : i < 60000 ? "warn" : "err"; };

  async function probe(origin) {
    var ac = typeof AbortController !== "undefined" ? new AbortController() : null;
    var t = setTimeout(function () { if (ac) ac.abort(); }, 8000);
    try {
      var r = await nativeFetch(origin + "/api/ping", { cache: "no-store", signal: ac ? ac.signal : undefined });
      return r.ok;
    } catch (e) { return false; } finally { clearTimeout(t); }
  }

  /* 连续失败 ≥4 次才探测;切换后 20 秒内不再切(防抖) */
  L.maybeFailover = async function () {
    if (L.probing || L.fails < 4 || Date.now() - L.lastSwitchAt < 20000) return false;
    L.probing = true;
    try {
      var cur = L.base || L.origin0, seen = {}, cands = [];
      [L.origin0].concat(L.alts).forEach(function (c) { if (c && c !== cur && !seen[c]) { seen[c] = 1; cands.push(c); } });
      for (var i = 0; i < cands.length; i++) {
        if (await probe(cands[i])) {
          L.base = cands[i] === L.origin0 ? "" : cands[i];
          L.fails = 0; L.switched++; L.lastSwitchAt = Date.now();
          say("线路不稳,已自动切换到备用线路 " + cands[i].replace(/^https?:\/\//, ""));
          return true;
        }
      }
      return false;
    } finally { L.probing = false; }
  };

  /* 轮询间隔:失败退避 / 页面隐藏或正在对话时降频(对话流本身就是心跳,不再和音频抢带宽) */
  L.nextGap = function (pollFails, hidden, chatBusy) {
    if (pollFails > 0) return Math.min(8000, 1500 * Math.pow(1.6, pollFails - 1));
    return hidden ? 10000 : chatBusy ? 8000 : 3000;
  };

  return L;
});
