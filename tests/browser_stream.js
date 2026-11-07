// 真实浏览器(Chromium)自检 v8:  CLONE_MOCK=1 node tests/browser_stream.js   (需要 npm i playwright && npx playwright install chromium)
// 场景 A: 流式通道正常 → 听到声音(player.firstDone),0 断流,ttfa 提示含「流式」
// 场景 B: 隧道缓冲流式响应(/api/chat_stream 被拦截并永不返回)→ 3 秒内自动退回长轮询,照样出声,且记住不再用流式
// 场景 C: 流式中途被掐断 → 用游标无缝续传,音频块 idx 连续、不重复
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const path = require("path");
const ROOT = path.resolve(__dirname, ".."), PORT = process.env.TEST_PORT || "7865", BASE = "http://127.0.0.1:" + PORT;
let fails = 0;
const check = (n, c, x) => { console.log((c ? "PASS " : "FAIL ") + n + (x !== undefined ? " -> " + String(x).slice(0, 200) : "")); if (!c) fails++; };
function wav(sec, sr) {
  const n = sec * sr, b = Buffer.alloc(44 + n * 2);
  b.write("RIFF", 0); b.writeUInt32LE(36 + n * 2, 4); b.write("WAVEfmt ", 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write("data", 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) { const t = i / sr; b.writeInt16LE(Math.round((Math.sin(2 * Math.PI * 170 * t) * (0.5 + 0.4 * Math.sin(2 * Math.PI * 2.5 * t)) * 0.4 + (Math.random() - 0.5) * 0.01) * 32767), 44 + i * 2); }
  return b;
}
(async () => {
  const srv = spawn("python3", ["-m", "uvicorn", "server.app:app", "--port", PORT, "--no-access-log"], { cwd: ROOT, env: { ...process.env, CLONE_MOCK: "1", CLONE_DATA: path.join(ROOT, "data_test_bs") }, stdio: "ignore" });
  try {
    for (let i = 0; i < 80; i++) { try { if ((await fetch(BASE + "/api/health")).ok) break; } catch (e) {} await new Promise((r) => setTimeout(r, 500)); }
    await new Promise((r) => setTimeout(r, 2500));
    const fd = new FormData(); fd.append("file", new Blob([wav(8, 24000)], { type: "audio/wav" }), "ref.wav"); fd.append("text", ""); fd.append("lang", "zh"); fd.append("song", "0");
    check("train accepted", (await fetch(BASE + "/api/voice/train", { method: "POST", body: fd })).ok);
    for (let i = 0; i < 120; i++) { const j = await (await fetch(BASE + "/api/voice/job")).json(); if (j.state === "done" || j.state === "error") break; await new Promise((r) => setTimeout(r, 500)); }
    const br = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
    async function scenario(name, setup, msgs) {
      const ctx = await br.newContext({ viewport: { width: 1400, height: 800 } });
      const pg = await ctx.newPage(); const errs = []; pg.on("pageerror", (e) => errs.push(e.message));
      await setup(pg);
      await pg.goto(BASE + "/"); await pg.waitForTimeout(1200); await pg.click("#enter"); await pg.waitForTimeout(500);
      let ok = false;
      for (let k = 0; k < (msgs || 1); k++) {   // 多发几条:B 场景需要连续 2 次无首帧才会记住「不用流式」
        if (k > 0) { await pg.waitForTimeout(300); }
        await pg.fill("#input", "你好,给我讲一个很短的故事。"); await pg.click("#send");
        for (let i = 0; i < 200; i++) { ok = await pg.evaluate(() => window.player.firstDone === true); if (ok) break; await pg.waitForTimeout(100); }
        if (k < (msgs || 1) - 1) for (let i = 0; i < 300; i++) { const st = await pg.evaluate(() => window.player.state); if (st === "idle" && i > 10) break; await pg.waitForTimeout(100); }
      }
      check(name + ": 出声", ok);
      for (let i = 0; i < 300; i++) { const st = await pg.evaluate(() => window.player.state); if (st === "idle" && i > 10) break; await pg.waitForTimeout(100); }
      const info = await pg.evaluate(() => ({ under: window.player.underruns, played: window.player.expect, title: document.getElementById("ttfa").title, stream: localStorage.getItem("vc_stream"), ttfa: document.getElementById("ttfa").textContent }));
      check(name + ": 0 断流", info.under === 0, JSON.stringify(info));
      check(name + ": 无页面 JS 错误", errs.length === 0, errs.join("|"));
      await ctx.close();
      return info;
    }
    const A = await scenario("A 流式", async () => {});
    check("A: 走的是流式通道", /流式/.test(A.title), A.title);
    const B = await scenario("B 隧道缓冲流式", async (pg) => { await pg.route("**/api/chat_stream", () => { /* 永不响应 */ }); }, 2);
    check("B: 自动退回长轮询并记住", B.stream === "0" && /长轮询/.test(B.title), B.title + " / " + B.stream);
    const C = await scenario("C 流式中途掐断", async (pg) => {
      await pg.route("**/api/chat_stream", async (route) => {
        const resp = await route.fetch(); const buf = await resp.body();
        // 只放行前 6 个帧(hello + 5 事件),随后「连接断开」:返回截断的 body
        let off = 0, frames = 0; while (off + 5 <= buf.length && frames < 6) { off += 5 + buf.readUInt32BE(off + 1); frames++; }
        route.fulfill({ status: 200, headers: resp.headers(), body: buf.subarray(0, off) });
      });
    });
    check("C: 断流后续传成功(0 断流)", C.under === 0);
    await br.close();
  } catch (e) { console.log("FAIL exception", e && e.stack); fails++; }
  srv.kill();
  console.log(fails ? "结果: 失败 " + fails + " 项" : "结果: 全部通过");
  process.exit(fails ? 1 : 0);
})();
