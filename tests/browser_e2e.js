// 真实浏览器(Chromium)自检:  CLONE_MOCK=1 node tests/browser_e2e.js   (需要 npm i playwright && npx playwright install chromium)
// 模拟:上传 70 秒立体声 44.1kHz 参考音频 + 隧道丢包(前 4 个分块请求被掐断)→ 压缩/分块/重试/训练成功 → 打开「运行日志」。
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs"), path = require("path");
const ROOT = path.resolve(__dirname, ".."), PORT = process.env.TEST_PORT || "7864", BASE = "http://127.0.0.1:" + PORT;
let fails = 0;
const check = (n, c, x) => { console.log((c ? "PASS " : "FAIL ") + n + (x ? " -> " + String(x).slice(0, 220) : "")); if (!c) fails++; };
function wav(sec, sr, ch) {
  const n = sec * sr, b = Buffer.alloc(44 + n * ch * 2);
  b.write("RIFF", 0); b.writeUInt32LE(36 + n * ch * 2, 4); b.write("WAVEfmt ", 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(ch, 22);
  b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * ch * 2, 28); b.writeUInt16LE(ch * 2, 32); b.writeUInt16LE(16, 34); b.write("data", 36); b.writeUInt32LE(n * ch * 2, 40);
  for (let i = 0; i < n; i++) { const t = i / sr, v = Math.sin(2 * Math.PI * 170 * t) * (0.5 + 0.4 * Math.sin(2 * Math.PI * 2.5 * t)) * 0.5 + (Math.random() - 0.5) * 0.01; for (let c = 0; c < ch; c++) b.writeInt16LE(Math.round(v * 32000), 44 + (i * ch + c) * 2); }
  return b;
}
(async () => {
  const srv = spawn("python3", ["-m", "uvicorn", "server.app:app", "--port", PORT, "--no-access-log"], { cwd: ROOT, env: { ...process.env, CLONE_MOCK: "1", PORT, CLONE_DATA: path.join(ROOT, "data_test_browser") }, stdio: "ignore" });
  const f = path.join("/tmp", "ref70.wav"); fs.writeFileSync(f, wav(70, 44100, 2));
  check("test wav is big (like user's 4.6MB)", fs.statSync(f).size > 10e6, fs.statSync(f).size);
  try {
    for (let i = 0; i < 80; i++) { try { if ((await fetch(BASE + "/api/health")).ok) break; } catch (e) {} await new Promise((r) => setTimeout(r, 500)); }
    await new Promise((r) => setTimeout(r, 2500));
    const br = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
    const pg = await (await br.newContext({ viewport: { width: 1400, height: 800 } })).newPage();
    const errs = []; pg.on("pageerror", (e) => errs.push(e.message));
    let chunkReq = 0, dropped = 0, sizes = [];
    await pg.route("**/api/voice/chunk?**", async (route) => {
      chunkReq++; sizes.push((route.request().postDataBuffer() || Buffer.alloc(0)).length);
      if (dropped < 4) { dropped++; return route.abort("connectionreset"); }
      route.continue();
    });
    await pg.goto(BASE + "/"); await pg.click("#enter");
    check("log button in header", await pg.isVisible("#logbtn"));
    await pg.setInputFiles("#file", f);
    await pg.waitForFunction(() => !document.getElementById("train").disabled);
    await pg.click("#train");
    await pg.waitForFunction(() => /训练已启动|❌|完成/.test(document.getElementById("gl").textContent) || document.getElementById("result").style.display === "flex", null, { timeout: 60000 });
    await pg.waitForFunction(() => document.getElementById("result").style.display === "flex", null, { timeout: 60000 });
    check("training finished in browser", true, await pg.textContent("#rmode"));
    const total = sizes.slice(4).reduce((a, b) => a + b, 0);  // 前 4 个是被掐断的请求
    check("upload compressed far below 12MB original (<1.6MB)", total < 1.6e6 && total > 100e3, total);
    check("chunks <= 192KB", Math.max(...sizes) <= 192 * 1024 + 1, Math.max(...sizes));
    check("4 dropped requests were retried", dropped === 4 && chunkReq > 4, chunkReq);
    // 运行日志
    await pg.click("#logbtn");
    await pg.waitForFunction(() => /瓶颈分析/.test(document.getElementById("logpre").textContent), null, { timeout: 20000 });
    const txt = await pg.textContent("#logpre"), an = await pg.textContent("#logan");
    check("log has analysis section", /瓶颈分析/.test(txt) && an.length > 10, an);
    check("log has upload retry info", /分块重试|上传正常|上传速度/.test(an), an);
    check("log has perf table", /各环节耗时/.test(txt) && /cpu\/preprocess_reference/.test(txt));
    check("log has training timeline", /训练任务/.test(txt) && /用时/.test(txt));
    check("log has browser section", /浏览器端/.test(txt));
    check("log has server log", /服务端日志/.test(txt));
    check("no JS page errors", errs.length === 0, errs.join("|"));
    await pg.screenshot({ path: "/tmp/log_modal.png" });
    await br.close();
  } catch (e) { console.log("EXC", e); fails++; }
  srv.kill(); fs.rmSync(path.join(ROOT, "data_test_browser"), { recursive: true, force: true });
  console.log("\n结果:", fails ? "失败 " + fails : "全部通过"); process.exit(fails ? 1 : 0);
})();
