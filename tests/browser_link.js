// 真实浏览器(Chromium)连接稳定性自检:  CLONE_MOCK=1 node tests/browser_link.js
// 需要 npm i playwright && npx playwright install chromium
// 场景 A:主线路偶发超时/断开(前 3 次 /api/state 失败)→ 指示灯不能变红(单次失败不算断连),恢复后继续正常。
// 场景 B:主线路彻底卡死 → 页面自动切到备用线路(第二个服务实例),指示灯仍是「已连接」,训练面板/状态照常刷新。
// 场景 C:页面全程没有 JS 报错,且没有再出现独立的 /api/ping 定时器(请求数远低于旧版的 4 个定时器)。
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs"), os = require("os"), path = require("path");
const ROOT = path.resolve(__dirname, ".."), P1 = process.env.TEST_PORT || "7881", P2 = String(+P1 + 1);
const B1 = "http://127.0.0.1:" + P1, B2 = "http://127.0.0.1:" + P2;
let fails = 0;
const check = (n, c, x) => { console.log((c ? "PASS " : "FAIL ") + n + (x ? " -> " + String(x).slice(0, 200) : "")); if (!c) fails++; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const tun = path.join(os.tmpdir(), "tun_browser_" + P1 + ".json");
fs.writeFileSync(tun, JSON.stringify({ primary: B1, urls: [B1, B2] }));
function start(port) {
  return spawn(process.env.PYTHON || "python3", ["-m", "uvicorn", "server.app:app", "--port", port, "--no-access-log"], { cwd: ROOT, env: Object.assign({}, process.env, { CLONE_MOCK: "1", CLONE_TUNNEL_FILE: tun, PORT: port }), stdio: "ignore" });
}
async function waitUp(base) { for (let i = 0; i < 80; i++) { try { if ((await fetch(base + "/api/health")).ok) return; } catch (e) {} await sleep(500); } throw new Error("server not up " + base); }

(async () => {
  const s1 = start(P1), s2 = start(P2);
  try {
    await waitUp(B1); await waitUp(B2);
    const browser = await chromium.launch();
    const ctx = await browser.newContext(); const page = await ctx.newPage();
    const jsErrors = []; page.on("pageerror", (e) => jsErrors.push(e.message));
    let reqs = { state: 0, ping: 0, status: 0 }, failState = 0, hardDown = false;
    await page.route("**/api/**", async (route) => {
      const u = route.request().url();
      if (u.startsWith(B1)) {
        if (/\/api\/state/.test(u)) reqs.state++;
        if (/\/api\/ping/.test(u)) reqs.ping++;
        if (/\/api\/status/.test(u)) reqs.status++;
        if (hardDown) return route.abort("connectionreset");
        if (failState > 0 && /\/api\/state/.test(u)) { failState--; return route.abort("timedout"); }
      }
      return route.continue();
    });
    failState = 3;
    await page.goto(B1 + "/");
    await page.waitForFunction(() => window.__LINK && window.__LINK.alts.length > 0 && /已连接/.test(document.querySelector("#t-conn").textContent), null, { timeout: 40000 });   // alts 非空 = 至少成功轮询过一次
    check("A: 前 3 次状态请求失败后自动恢复为「已连接」", true, await page.textContent("#t-conn"));
    const lvlDuring = await page.evaluate(() => window.__LINK.level());
    check("A: 单次/少量失败期间指示灯没有变红(level=ok)", lvlDuring === "ok", lvlDuring);
    const alts = await page.evaluate(() => window.__LINK.alts);
    check("服务端下发了备用线路", alts.includes(B2), JSON.stringify(alts));

    hardDown = true;   // 主线路彻底卡死
    await page.waitForFunction(() => window.__LINK.base !== "", null, { timeout: 60000 });
    const base = await page.evaluate(() => window.__LINK.base);
    check("B: 主线路卡死 → 自动切到备用线路", base === B2, base);
    await sleep(4500);
    const txt = await page.textContent("#t-conn");
    check("B: 切换后指示灯回到「已连接 · 备用线」", /已连接/.test(txt) && /备用线/.test(txt), txt);
    const tts = await page.textContent("#t-tts");
    check("B: 备用线路上状态数据照常刷新(语音卡数)", /\d\/\d/.test(tts), tts);
    check("C: 全程无 JS 报错", jsErrors.length === 0, jsErrors.join(" | "));
    check("C: 没有独立的 /api/ping 与 /api/status 轮询(只有 /api/state)", reqs.ping === 0 && reqs.status === 0, JSON.stringify(reqs));
    await browser.close();
  } catch (e) { check("browser_link 运行", false, e.stack || e.message); }
  finally { s1.kill(); s2.kill(); try { fs.unlinkSync(tun); } catch (e) {} }
  console.log(fails ? "\nFAILED " + fails : "\nALL PASSED");
  process.exit(fails ? 1 : 0);
})();
