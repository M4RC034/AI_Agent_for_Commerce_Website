/* Agentic customer-service demo UI.
 *
 * Reads the SSE stream from /api/chat and renders each event as it lands:
 * tool_start puts a live chip in the transcript before the executor has done
 * any work, token appends to the answer, tool_end marks the chip done. The
 * trace panel pulls the structured record the eval harness scores from, so what
 * you see in the UI and what the evals assert are the same thing. */

const $ = (s) => document.querySelector(s);
const messages = $("#messages");
const traceBody = $("#trace-body");
const traceMeta = $("#trace-meta");

let session = null;
let mode = "rag";
let busy = false;
let pendingFile = null;

// ---------------------------------------------------------------- session

async function startSession() {
  traceBody.innerHTML = '<p class="trace-empty">建立工作階段中…</p>';
  try {
    const res = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, show_thinking: $("#thinking").checked }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    session = await res.json();
    traceMeta.textContent = `${session.mode.toUpperCase()} · ${session.model}`;
    traceBody.innerHTML =
      '<p class="trace-empty">模型每一步的決策會顯示在這裡：檢索查詢、充分性判斷、工具呼叫與回傳值。</p>';
  } catch (err) {
    session = null;
    traceMeta.textContent = "無法連線";
    traceBody.innerHTML = `<p class="trace-empty">建立工作階段失敗：${escapeHtml(err.message)}</p>`;
  }
}

function resetChat() {
  if (session) {
    // Best-effort server-side cleanup; the new session is what matters.
    fetch(`/api/session/${session.session_id}`, { method: "DELETE" }).catch(() => {});
  }
  messages.innerHTML = "";
  messages.appendChild(emptyState());
  startSession();
}

// ---------------------------------------------------------------- render

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal markdown: **bold** and bullet lines. The agent writes prose with the
// occasional bold figure; a full parser would be more machinery than that earns.
function lightMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- /gm, "・");
}

function emptyState() {
  const div = document.createElement("div");
  div.className = "empty";
  div.innerHTML = $("#empty")?.innerHTML ?? "";
  wireChips(div);
  return div;
}

function wireChips(root) {
  root.querySelectorAll(".chip").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.hint === "image") { $("#file").click(); return; }
      $("#input").value = b.textContent;
      $("#composer").requestSubmit();
    }));
}

function clearEmpty() {
  messages.querySelector(".empty")?.remove();
}

function addUser(text, file) {
  clearEmpty();
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text || "（附上商品照片）";
  if (file) {
    const img = document.createElement("img");
    img.className = "sent";
    img.src = URL.createObjectURL(file);
    img.alt = "已送出的商品照片";
    el.appendChild(img);
  }
  messages.appendChild(el);
  scroll();
}

function addBot() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = '<div class="who">小電</div>';
  messages.appendChild(el);
  return el;
}

function scroll() {
  messages.scrollTop = messages.scrollHeight;
}

// ---------------------------------------------------------------- streaming

async function send(text) {
  const file = pendingFile;
  if (busy || (!text.trim() && !file)) return;
  if (!session) { await startSession(); if (!session) return; }

  busy = true;
  $("#send").disabled = true;
  addUser(text, file);
  clearPending();

  const bot = addBot();
  let answer = null;   // created lazily, so tool chips can precede the text
  let think = null;
  let openTool = null;

  try {
    // multipart, so an image can ride along with the turn
    const form = new FormData();
    form.append("session_id", session.session_id);
    form.append("message", text);
    if (file) form.append("image", file);

    const res = await fetch("/api/chat", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      const frames = buf.split("\n\n");
      buf = frames.pop();

      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const ev = JSON.parse(line.slice(5).trim());

        if (ev.type === "tool_start") {
          openTool = document.createElement("div");
          openTool.className = "tool";
          openTool.innerHTML = `<span class="spin"></span><span>${escapeHtml(ev.label)}</span>`;
          bot.appendChild(openTool);
          answer = null;   // any further text starts a fresh block below the chip

        } else if (ev.type === "tool_end") {
          if (openTool) {
            openTool.classList.add(ev.ok ? "done" : "err");
            openTool.querySelector("span:last-child").textContent = ev.summary;
            openTool = null;
          }

        } else if (ev.type === "token") {
          if (!answer) {
            answer = document.createElement("div");
            answer.className = "body";
            answer.dataset.raw = "";
            bot.appendChild(answer);
          }
          answer.dataset.raw += ev.text;
          answer.innerHTML = lightMarkdown(answer.dataset.raw);

        } else if (ev.type === "thinking") {
          if (!think) {
            think = document.createElement("div");
            think.className = "think";
            bot.appendChild(think);
          }
          think.textContent += ev.text;

        } else if (ev.type === "notice") {
          const n = document.createElement("div");
          n.className = "notice";
          n.textContent = ev.text;
          bot.appendChild(n);

        } else if (ev.type === "error") {
          const e = document.createElement("div");
          e.className = "err-box";
          e.textContent = ev.text;
          bot.appendChild(e);
        }
        scroll();
      }
    }
  } catch (err) {
    const e = document.createElement("div");
    e.className = "err-box";
    e.textContent = `連線中斷：${err.message}`;
    bot.appendChild(e);
  } finally {
    busy = false;
    $("#send").disabled = false;
    $("#input").focus();
    refreshTrace();
  }
}

// ---------------------------------------------------------------- attachment

function clearPending() {
  pendingFile = null;
  $("#file").value = "";
  $("#preview").hidden = true;
}

$("#file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  pendingFile = f;
  $("#preview-img").src = URL.createObjectURL(f);
  $("#preview").hidden = false;
  $("#input").focus();
});

$("#preview-clear").addEventListener("click", clearPending);

// ---------------------------------------------------------------- trace

async function refreshTrace() {
  if (!session) return;
  try {
    const res = await fetch(`/api/trace/${session.session_id}`);
    const { records } = await res.json();
    if (!records.length) return;

    traceBody.innerHTML = "";
    let steps = 0;
    for (const r of records) {
      if (r.event === "model_response") { steps = Math.max(steps, r.step || 0); continue; }

      const div = document.createElement("div");
      if (r.event === "user_turn") {
        div.className = "tr";
        div.innerHTML = `<div class="k">第 ${r.turn} 輪 · 顧客</div><div class="v">${escapeHtml(r.text)}</div>`;
      } else if (r.event === "tool_call") {
        div.className = "tr tool";
        div.innerHTML = `<div class="k">工具呼叫 · 第 ${r.step} 步</div>
          <div class="v">${escapeHtml(r.tool)}(${escapeHtml(JSON.stringify(r.input))})</div>`;
      } else if (r.event === "tool_result") {
        div.className = "tr";
        div.innerHTML = `<div class="k">回傳 ${r.ok ? "" : "· 錯誤"}</div>
          <div class="v">${escapeHtml(r.payload.slice(0, 260))}</div>`;
      } else if (r.event === "retrieval") {
        div.className = "tr retr";
        const reform = r.reformulation_of
          ? `<div class="v">↻ 改寫自「${escapeHtml(r.reformulation_of)}」</div>` : "";
        div.innerHTML = `<div class="k">檢索 · 第 ${r.attempt} 次
            <span class="badge ${r.sufficiency}">${r.sufficiency} ${r.top_score ?? ""}</span></div>
          <div class="v">${escapeHtml(r.query)}</div>${reform}`;
      } else if (r.event === "image_search") {
        div.className = "tr img";
        const verdict = r.in_domain
          ? `<span class="badge sufficient">in-domain</span>`
          : `<span class="badge insufficient">rejected</span>`;
        div.innerHTML = `<div class="k">以圖搜圖 ${verdict}</div>
          <div class="v">${escapeHtml(r.image_ref)} → 判定「${escapeHtml(r.category)}」，`
          + `候選 ${r.candidates} 項</div>`;
      } else if (r.event === "guard_tripped") {
        div.className = "tr retr";
        div.innerHTML = `<div class="k">保護機制觸發</div><div class="v">${escapeHtml(r.reason)}</div>`;
      }
      traceBody.appendChild(div);
    }
    traceMeta.textContent = `${session.mode.toUpperCase()} · ${steps} 步`;
    traceBody.scrollTop = traceBody.scrollHeight;
  } catch { /* the panel is diagnostic; never break the chat over it */ }
}

// ---------------------------------------------------------------- wiring

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#input").value;
  $("#input").value = "";
  send(text);
});

$("#mode-toggle").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.mode === mode) return;
  mode = btn.dataset.mode;
  document.querySelectorAll("#mode-toggle button")
    .forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  resetChat();   // mode is fixed per session — a switch starts a new one
});

$("#thinking").addEventListener("change", resetChat);
$("#reset").addEventListener("click", resetChat);
wireChips(document);
startSession();
