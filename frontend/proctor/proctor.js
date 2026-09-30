/*
 * proctor.js: drop-in webcam proctoring client for the CAT exam page.
 *
 *   <link rel="stylesheet" href="/proctor-static/proctor.css">
 *   <script src="/proctor-static/proctor.js"></script>
 *   const proctor = new ProctorClient({ apiBase: "", examRef: catSessionId,
 *                                       onAlert: a => ..., onDeclined: () => ... });
 *   const ok = await proctor.start();   // consent -> camera -> calibration; resolves true when monitoring
 *   ...
 *   await proctor.stop();                // on exam submit
 *
 * All detection runs on the server (the browser only streams frames and 16 kHz
 * microphone audio), so a candidate can't disable the checks from DevTools.
 */
(function (global) {
  "use strict";

  // AudioWorklet: downsample the mic to 16 kHz mono int16, 250 ms per message.
  const WORKLET_SRC = `
  class PcmDown extends AudioWorkletProcessor {
    constructor(opts) {
      super();
      this.ratio = sampleRate / 16000; this.pos = 0; this.acc = 0; this.n = 0;
      this.out = new Int16Array(4000); this.k = 0;
    }
    process(inputs) {
      const ch = inputs[0] && inputs[0][0];
      if (!ch) return true;
      for (let i = 0; i < ch.length; i++) {
        this.acc += ch[i]; this.n++; this.pos += 1;
        if (this.pos >= this.ratio) {             // box-filter decimation (anti-aliasing low-pass)
          this.pos -= this.ratio;
          const v = Math.max(-1, Math.min(1, this.acc / this.n));
          this.out[this.k++] = v * 32767; this.acc = 0; this.n = 0;
          if (this.k === this.out.length) {
            // capture time of this chunk's last sample, on the AudioContext clock (sample-accurate,
            // unaffected by main-thread stalls)
            const tEnd = currentTime + (i + 1) / sampleRate;
            this.port.postMessage({ pcm: this.out.buffer, t: tEnd }, [this.out.buffer]);
            this.out = new Int16Array(4000); this.k = 0;
          }
        }
      }
      return true;
    }
  }
  registerProcessor("pcm-down", PcmDown);`;

  const TARGET_POS = { center: [50, 50], top_left: [5, 5], top_right: [95, 5],
                       bottom_right: [95, 95], bottom_left: [5, 95] };

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  class ProctorClient {
    constructor(opts = {}) {
      this.apiBase = opts.apiBase || "";
      this.examRef = opts.examRef || null;
      this.onAlert = opts.onAlert || (() => {});
      this.onStatus = opts.onStatus || (() => {});
      this.onDeclined = opts.onDeclined || (() => {});
      this.onEnded = opts.onEnded || (() => {});
      this.debug = !!opts.debug;
      this.sessionId = null;
      this.ws = null;
      this.stream = null;
      this.timer = null;
      this.mode = "idle";
      this._waiters = {};
    }

    // ---------- lifecycle ----------
    async start() {
      const policy = await this._json("GET", "/proctor/policy");
      const accepted = await this._consent(policy);
      if (!accepted) { this.onDeclined(); return false; }

      const s = await this._json("POST", "/proctor/sessions",
        { consent: true, policy_version: policy.version, exam_ref: this.examRef });
      this.sessionId = s.session_id;
      this.cfg = s;

      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          // no noise suppression (it can hide a second, quieter voice); auto-gain on
          audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false,
                 autoGainControl: true } });   // AGC lifts quiet laptop mics to normal speech level
      } catch (e) {
        this._modal("Camera and microphone required",
          "<p>This exam requires your webcam and microphone. Please allow access to both and reload the page.</p>", ["OK"]);
        throw e;
      }
      this._buildPreview();
      await this._connect();
      this._t0 = performance.now();             // one clock for video frames and audio chunks
      this._startStreaming();
      await this._startAudio();

      let cal;
      for (;;) {
        cal = await this._calibrate();
        if (cal.ok || !cal.retry) break;
        await this._modal("Let's try that again",
          "<p>We couldn't see your face clearly. Sit facing the screen with your face well lit and inside the camera view, then look at the dot.</p>",
          ["Retry"]);
      }
      this.mode = cal.mode;
      this._setBadge(cal.mode === "monitoring" ? "ok" : "warn",
        cal.mode === "monitoring" ? "Proctoring active" : "Proctoring active (presence only)");
      return true;
    }

    async stop() {
      if (!this.ws) return;
      try { this.ws.send(JSON.stringify({ type: "end" })); } catch (e) {}
      this._setBadge("warn", "Finishing proctoring review…");
      await Promise.race([this._wait("ended"), sleep(60000)]);   // AI verdicts still in flight
      this._teardown();
    }

    _teardown() {
      clearInterval(this.timer);
      this.timer = null;
      if (this.audioCtx) { try { this.audioCtx.close(); } catch (e) {} this.audioCtx = null; }
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      if (this.ws && this.ws.readyState <= 1) this.ws.close();
      this.ws = null;
      if (this.preview) this.preview.remove();
      this.mode = "ended";
    }

    // ---------- network ----------
    async _json(method, path, body) {
      const r = await fetch(this.apiBase + path, {
        method, headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined });
      if (!r.ok) throw new Error(`${method} ${path} -> ${r.status}`);
      return r.json();
    }

    _connect() {
      const base = this.apiBase || window.location.origin;
      const url = base.replace(/^http/, "ws") + `/proctor/ws/${this.sessionId}`;
      return new Promise((resolve, reject) => {
        const ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";
        ws.onmessage = (m) => this._onMessage(JSON.parse(m.data));
        ws.onerror = reject;
        ws.addEventListener("close", (e) => reject(new Error("proctoring socket closed: " + e.code)), { once: true });
        ws.onclose = () => { if (this.mode !== "ended") this._setBadge("bad", "Proctoring disconnected"); };
        this.ws = ws;
        this._wait("ready").then(resolve);
      });
    }

    _wait(type) {
      return new Promise((res) => { (this._waiters[type] = this._waiters[type] || []).push(res); });
    }

    _onMessage(msg) {
      (this._waiters[msg.type] || []).splice(0).forEach((f) => f(msg));
      if (msg.type === "status") {
        this.onStatus(msg);
        if (this.mode === "monitoring" || this.mode === "presence_only") {
          if (msg.objects && msg.objects.length) this._setBadge("bad", `${msg.objects.join(", ")} detected`);
          else if (msg.face_count > 1 || msg.persons > 1) this._setBadge("bad", "Another person detected");
          else if (msg.foreign_hands > 0) this._setBadge("bad", "Another person's hand detected");
          else if (msg.face_count === 0) this._setBadge("bad", "Face not visible");
          else if (msg.state === "off_screen") this._setBadge("warn", "Please look at the screen");
          else if (msg.speaking) this._setBadge("warn", "Please stay silent");
          else this._setBadge("ok", this.mode === "monitoring" ? "Proctoring active" : "Proctoring active (presence only)");
        }
      } else if (msg.type === "flag") {
        this._toast(msg.title, msg.message);         // instant heads-up while the AI judge reviews
      } else if (msg.type === "alert") {
        this.onAlert(msg);
        const title = msg.title ? `⚠ ${esc(msg.title)}` : "⚠ Proctoring alert";
        this._modal(title, `<p>${esc(msg.message)}</p>`, ["I understand"]).then(() => {
          try { this.ws.send(JSON.stringify({ type: "ack", event_id: msg.event_id })); } catch (e) {}
        });
      } else if (msg.type === "event" && this.debug) {
        console.info("[proctor] event", msg);
      } else if (msg.type === "ended") {
        this.onEnded(msg.summary);
      }
    }

    // ---------- capture ----------
    _startStreaming() {
      const { fps, width, height, jpeg_quality } = this.cfg.stream;
      const canvas = document.createElement("canvas");
      canvas.width = width; canvas.height = height;
      const ctx = canvas.getContext("2d");
      const video = this.video;
      const t0 = this._t0 || performance.now();
      this.timer = setInterval(() => {
        const ws = this.ws;
        if (!ws || ws.readyState !== 1 || video.readyState < 2) return;
        if (ws.bufferedAmount > 256 * 1024) return;          // slow link: skip rather than queue
        ctx.drawImage(video, 0, 0, width, height);
        const t = performance.now() - t0;
        canvas.toBlob((blob) => {
          if (!blob || !this.ws || this.ws.readyState !== 1) return;
          blob.arrayBuffer().then((buf) => {
            const out = new Uint8Array(8 + buf.byteLength);
            new DataView(out.buffer).setFloat64(0, t, true);
            out.set(new Uint8Array(buf), 8);
            this.ws.send(out.buffer);
          });
        }, "image/jpeg", jpeg_quality);
      }, 1000 / fps);
    }

    async _startAudio() {
      const track = this.stream.getAudioTracks()[0];
      if (!track) return;
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      this.audioCtx = ctx;
      const src = ctx.createMediaStreamSource(new MediaStream([track]));
      // Map the AudioContext clock onto the performance.now() clock the video frames use.
      // The offset is smoothed so audio-device vs system clock drift is followed slowly.
      let offset = null;
      const ctxToPerf = (ctxSec) => {
        let inst = performance.now() - ctx.currentTime * 1000;
        if (ctx.getOutputTimestamp) {
          const ts = ctx.getOutputTimestamp();
          if (ts && ts.performanceTime) inst = ts.performanceTime - ts.contextTime * 1000;
        }
        offset = offset === null ? inst : offset + 0.02 * (inst - offset);
        return ctxSec * 1000 + offset;
      };
      const send = (buf, ctxEnd) => {
        const ws = this.ws;
        if (!ws || ws.readyState !== 1 || ws.bufferedAmount > 512 * 1024) return;
        const t = ctxToPerf(ctxEnd) - this._t0;              // capture time of the chunk's last sample
        const out = new Uint8Array(12 + buf.byteLength);
        out.set([65, 85, 68, 49], 0);                        // "AUD1"
        new DataView(out.buffer).setFloat64(4, t, true);
        out.set(new Uint8Array(buf), 12);
        ws.send(out.buffer);
      };
      try {
        const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
        await ctx.audioWorklet.addModule(url);
        const node = new AudioWorkletNode(ctx, "pcm-down");
        node.port.onmessage = (m) => send(m.data.pcm, m.data.t);
        src.connect(node);
        // keep the graph pulling without making any sound
        const mute = ctx.createGain(); mute.gain.value = 0; node.connect(mute).connect(ctx.destination);
      } catch (e) {                                          // older browsers: ScriptProcessor fallback
        const proc = ctx.createScriptProcessor(4096, 1, 1);
        const ratio = ctx.sampleRate / 16000;
        let pend = [];
        proc.onaudioprocess = (ev) => {
          const x = ev.inputBuffer.getChannelData(0);
          for (let i = 0; i + ratio <= x.length; i += ratio) {
            let a = 0, n = 0;
            for (let j = Math.floor(i); j < Math.floor(i + ratio); j++) { a += x[j]; n++; }
            pend.push(Math.max(-1, Math.min(1, a / Math.max(n, 1))) * 32767);
          }
          if (pend.length >= 4000) {
            send(Int16Array.from(pend.splice(0, 4000)).buffer, ctx.currentTime - pend.length / 16000);
          }
        };
        src.connect(proc); proc.connect(ctx.destination);
      }
      if (ctx.state === "suspended") { try { await ctx.resume(); } catch (e) {} }
    }

    _buildPreview() {
      const p = el("div", "proctor-preview");
      const v = el("video");
      v.autoplay = true; v.muted = true; v.playsInline = true;
      v.srcObject = this.stream;
      const badge = el("div", "proctor-badge", "<span class='dot'></span><span class='txt'>Starting…</span>");
      p.append(v, badge);
      document.body.append(p);
      this.preview = p; this.video = v; this.badge = badge;
    }

    _setBadge(level, text) {
      if (!this.badge) return;
      this.badge.dataset.level = level;
      this.badge.querySelector(".txt").textContent = text;
    }

    // ---------- calibration ----------
    async _calibrate() {
      const overlay = el("div", "proctor-calib");
      const msg = el("div", "proctor-calib-msg",
        "<h2>Quick calibration</h2><p>Sit the way you will during the exam, <b>look at the dot</b> and <b>stay quiet</b> until it disappears (3 seconds). This also measures your room's background noise.</p>");
      const dot = el("div", "proctor-dot");
      overlay.append(msg, dot);
      document.body.append(overlay);
      this._setBadge("warn", "Calibrating…");
      await sleep(2500);
      msg.style.opacity = "0";
      const secs = this.cfg.calibration.seconds_per_target;
      for (const tgt of this.cfg.calibration.targets) {
        const [x, y] = TARGET_POS[tgt.name];
        dot.style.left = x + "%"; dot.style.top = y + "%";
        dot.classList.remove("pulse"); void dot.offsetWidth; dot.classList.add("pulse");
        await sleep(350);                                 // let the eyes arrive before the server starts sampling
        this.ws.send(JSON.stringify({ type: "calib_target", name: tgt.name }));
        await sleep(secs * 1000);
        this.ws.send(JSON.stringify({ type: "calib_target_end" }));
      }
      const resultP = this._wait("calibration");
      this.ws.send(JSON.stringify({ type: "calib_finish" }));
      const result = await resultP;
      overlay.remove();
      if (this.debug) console.info("[proctor] calibration", result);
      return result;
    }

    // ---------- UI ----------
    _consent(policy) {
      const items = policy.points.map((p) => `<li>${esc(p)}</li>`).join("");
      return this._modal(esc(policy.title),
        `<ul class="proctor-policy">${items}</ul><p class="proctor-small">Policy version ${esc(policy.version)}</p>`,
        ["Decline", "I agree, start proctoring"]).then((i) => i === 1);
    }

    _toast(title, message) {
      if (!this._toastBox) {
        this._toastBox = el("div", "proctor-toasts");
        document.body.append(this._toastBox);
      }
      const t = el("div", "proctor-toast", `<b>${esc(title || "Notice")}</b><span>${esc(message || "")}</span>`);
      this._toastBox.append(t);
      setTimeout(() => t.classList.add("leaving"), 5000);
      setTimeout(() => t.remove(), 5600);
    }

    _modal(title, html, buttons) {
      return new Promise((resolve) => {
        const back = el("div", "proctor-modal-back");
        const box = el("div", "proctor-modal", `<h2>${title}</h2>${html}`);
        const row = el("div", "proctor-actions");
        buttons.forEach((b, i) => {
          const btn = el("button", i === buttons.length - 1 ? "primary" : "", esc(b));
          btn.onclick = () => { back.remove(); resolve(i); };
          row.append(btn);
        });
        box.append(row); back.append(box); document.body.append(back);
        row.lastChild.focus();
      });
    }
  }

  global.ProctorClient = ProctorClient;
})(window);
