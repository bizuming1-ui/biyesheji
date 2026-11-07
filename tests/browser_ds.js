// 真实浏览器(Chromium)端到端:假 DeepSeek + Mock TTS + 真实 player.js + 真实 AudioContext,验证「播放零断流」。
//   NODE_PATH=/path/to/node_modules node tests/browser_ds.js           (需要 playwright + chromium)
//   MOCK_CHUNK_SLEEP=0.3 node tests/browser_ds.js                       (模拟单卡 RTF≈0.9 的慢合成)
const { spawn } = require("child_process");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..");
const PORT = process.env.TEST_PORT || "7863", DSP = "7872", GOOD = "sk-test-good-key-1234567890";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fails = [];
const check = (n, c, x) => { console.log((c ? "PASS " : "FAIL ") + n + (x !== undefined ? " -> " + JSON.stringify(x).slice(0, 300) : "")); if (!c) fails.push(n); };

function wav(sec = 8, sr = 24000) {
  const n = sec * sr, b = Buffer.alloc(44 + n * 2);
  b.write("RIFF", 0); b.writeUInt32LE(36 + n * 2, 4); b.write("WAVEfmt ", 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write("data", 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(9000 * Math.sin((2 * Math.PI * 180 * i) / sr) * (0.6 + 0.4 * Math.sin((2 * Math.PI * 2 * i) / sr))), 44 + i * 2);
  return b;
}

(async () => {
  const env = { ...process.env, CLONE_MOCK: "1", PORT, CLONE_DATA: "/tmp/clone_browser_ds", DEEPSEEK_BASE_URL: "http://127.0.0.1:" + DSP, FAKE_DS_PORT: DSP };
  const fake = spawn("python3", ["tests/fake_deepseek.py"], { cwd: ROOT, env, stdio: "ignore" });
  const srv = spawn("python3", ["-m", "uvicorn", "server.app:app", "--port", PORT, "--no-access-log"], { cwd: ROOT, env, stdio: "ignore" });
  let browser;
  try {
    const base = "http://127.0.0.1:" + PORT;
    for (let i = 0; i < 80; i++) { try { if ((await fetch(base + "/api/health")).ok) break; } catch (e) { /* 等待启动 */ } await sleep(500); }
    await sleep(2500);
    const fd = new FormData(); fd.append("file", new Blob([wav()], { type: "audio/wav" }), "ref.wav"); fd.append("text", ""); fd.append("lang", "zh");
    check("train accepted", (await fetch(base + "/api/voice/train", { method: "POST", body: fd })).ok);
    for (let i = 0; i < 80; i++) { const j = await (await fetch(base + "/api/voice/job")).json(); if (j.state === "done" || j.state === "error") { check("train done", j.state === "done"); break; } await sleep(300); }

    browser = await chromium.launch({ args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] });
    const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
    await ctx.addInitScript((k) => localStorage.setItem("vc_ds", JSON.stringify({ key: k, model: "deepseek-v4-pro", thinking: false })), GOOD);
    const page = await ctx.newPage();
    const errs = []; page.on("pageerror", (e) => errs.push(String(e)));
    await page.goto(base + "/");
    await page.click("#enter");
    await page.waitForFunction(() => document.getElementById("t-ds").textContent.includes("deepseek"), null, { timeout: 15000 });
    check("page validated saved key & shows model", true, await page.textContent("#t-ds"));
    check("llm chip says DeepSeek ready", (await page.textContent("#t-llm")).includes("DeepSeek"), await page.textContent("#t-llm"));

    // 记录播放器状态时间线
    await page.evaluate(() => { window.__tl = []; const o = player.onstate; player.onstate = (s) => { window.__tl.push([Math.round(performance.now()), s]); o(s); }; });
    await page.fill("#input", "你好,请介绍一下你自己");
    const t0 = Date.now();
    await page.click("#send");
    await page.waitForFunction(() => window.__tl.some((x) => x[1] === "playing"), null, { timeout: 30000 });
    const tFirst = Date.now() - t0;
    await page.waitForFunction(() => player.state === "idle" && player.total !== null && player.expect >= player.total && player.expect > 5, null, { timeout: 90000 });
    const r = await page.evaluate(() => ({ tl: window.__tl.map((x) => x[1]), underruns: player.underruns, gapTotal: player.gapTotal, prebuf: player.prebufferS, expect: player.expect, text: document.querySelector("#msgs .msg.a")?.textContent.slice(0, 40) }));
    console.log("timeline:", r.tl.join(" > "), "| first sound(ms):", tFirst, "| prebuffer(s):", r.prebuf.toFixed(2));
    check("assistant text rendered from DeepSeek", /DeepSeek/.test((await page.textContent("#msgs")) || ""), r.text);
    check("all audio chunks played", r.expect > 5, r.expect);
    check("ZERO underruns (no play-stop-play)", r.underruns === 0 && r.gapTotal === 0, { u: r.underruns, gap: r.gapTotal });
    const playingAfterStart = r.tl.slice(r.tl.indexOf("playing"));
    check("timeline has no buffering after playback started", !playingAfterStart.slice(1).includes("buffering"), playingAfterStart);
    check("first sound within 6s (mock)", tFirst < 6000, tFirst);
    check("no JS errors", errs.length === 0, errs);
    await page.screenshot({ path: "/tmp/site.png" });
  } catch (e) {
    check("test crashed", false, String(e));
  } finally {
    if (browser) await browser.close();
    srv.kill(); fake.kill();
  }
  console.log(fails.length ? "FAILED: " + fails.join(" | ") : "ALL PASS");
  process.exit(fails.length ? 1 : 0);
})();
