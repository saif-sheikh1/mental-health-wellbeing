import time
from datetime import datetime
from typing import Optional
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import core server logic, readers and models from main
from main import (
    app,
    live_sensor_snapshot,
    eeg_reader,
    arduino_reader,
    mgr,
    SensorInput,
    _sensor_to_vals,
)

@app.get("/api/sensors/live", summary="Get merged live EEG, GSR, and PPG readings")
def api_sensors_live():
    data = live_sensor_snapshot()
    if data.get("timestamp"):
        age = time.time() - datetime.fromisoformat(data["timestamp"]).timestamp()
        data["age_seconds"] = round(age, 2)
    return data


@app.get("/api/eeg/live", summary="Backward-compatible live sensor alias")
def api_eeg_live():
    return api_sensors_live()


@app.post("/api/sensors/stop", summary="Stop live sensor readers")
def api_sensors_stop():
    eeg_reader.stop()
    arduino_reader.stop()
    return {"online": False, "running": {"eeg": False, "arduino": False}}


@app.post("/api/eeg/stop", summary="Backward-compatible stop alias")
def api_eeg_stop():
    return api_sensors_stop()


@app.get("/api/arduino/live", summary="Live GSR + PPG from Arduino serial port")
def api_arduino_live():
    """Start the Arduino reader and return the latest GSR/PPG snapshot."""
    arduino_reader.start()
    snap = arduino_reader.snapshot()
    running = arduino_reader.is_running()
    if snap:
        age = round(
            time.time() - datetime.fromisoformat(snap["timestamp"]).timestamp(), 2
        )
        return {
            "online": True,
            "running": running,
            "port": arduino_reader.port,
            "baud": arduino_reader.baud,
            "gsr":  snap.get("gsr"),
            "ppg":  snap.get("ppg"),
            "raw_line": snap.get("raw_line"),
            "timestamp": snap.get("timestamp"),
            "age_seconds": age,
            "error": None,
        }
    return {
        "online": False,
        "running": running,
        "port": arduino_reader.port,
        "baud": arduino_reader.baud,
        "gsr":  None,
        "ppg":  None,
        "raw_line": None,
        "timestamp": None,
        "age_seconds": None,
        "error": arduino_reader.error,
    }


@app.post("/api/arduino/start", summary="Explicitly start the Arduino serial reader")
def api_arduino_start():
    ok = arduino_reader.start()
    return {
        "started": ok,
        "running": arduino_reader.is_running(),
        "port":    arduino_reader.port,
        "baud":    arduino_reader.baud,
        "error":   arduino_reader.error,
    }


@app.post("/api/arduino/stop", summary="Stop the Arduino serial reader")
def api_arduino_stop():
    arduino_reader.stop()
    return {"running": False, "port": arduino_reader.port}


@app.post("/api/predict/facial", summary="Facial emotion classification (Residual CNN)")
async def api_facial(file: UploadFile = File(...)):
    return mgr.predict_facial(await file.read())


@app.post("/api/predict/sensor", summary="Mental state from 10 sensor channels (RNN)")
def api_sensor(s: SensorInput):
    return mgr.predict_sensor(_sensor_to_vals(s))


@app.post("/api/predict/future", summary="5-step sensor forecast (Seq2Seq LSTM)")
def api_future(s: SensorInput):
    return {"forecast": mgr.predict_future(_sensor_to_vals(s))}


@app.post("/api/explain/shap/sensor", summary="SHAP feature importance for RNN sensor model")
def api_shap_sensor(s: SensorInput):
    return {"shap": mgr.explain_shap_sensor(_sensor_to_vals(s))}


@app.post("/api/explain/shap/future", summary="SHAP feature importance for future predictor")
def api_shap_future(s: SensorInput):
    return mgr.explain_shap_future(_sensor_to_vals(s))


@app.post("/api/explain/gradcam", summary="GradCAM attention heatmap for facial CNN")
async def api_gradcam(file: UploadFile = File(...)):
    return mgr.gradcam(await file.read())


@app.post("/api/predict/complete", summary="Full pipeline: sensor + facial + future + SHAP + recs")
async def api_complete(
    gsr:        float = Form(...),
    ppg:        float = Form(...),
    delta:      float = Form(...),
    theta:      float = Form(...),
    low_alpha:  float = Form(...),
    high_alpha: float = Form(...),
    low_beta:   float = Form(...),
    high_beta:  float = Form(...),
    low_gamma:  float = Form(...),
    mid_gamma:  float = Form(...),
    file: Optional[UploadFile] = File(None),
):
    vals = [gsr, ppg, delta, theta, low_alpha, high_alpha, low_beta, high_beta, low_gamma, mid_gamma]

    sensor_res    = mgr.predict_sensor(vals)
    future_res    = mgr.predict_future(vals)
    future_states = [f["state"] for f in future_res]

    # Facial is optional — skip gracefully if no image supplied
    facial_res  = None
    gradcam_res = None
    if file and file.filename:
        content = await file.read()
        if content:
            try:
                facial_res  = mgr.predict_facial(content)
                gradcam_res = mgr.gradcam(content)
            except Exception as exc:
                print(f"  [facial] skipped: {exc}")

    emotion = facial_res["emotion"] if facial_res else "Neutral"
    recs    = mgr.recommendations(sensor_res["state"], emotion, future_states)

    # SHAP explanations — best-effort; never crash the full pipeline
    shap_sensor_res = None
    shap_future_res = None
    try:
        shap_sensor_res = mgr.explain_shap_sensor(vals)
    except Exception as exc:
        print(f"  [shap_sensor] skipped: {exc}")
    try:
        shap_future_res = mgr.explain_shap_future(vals)
    except Exception as exc:
        print(f"  [shap_future] skipped: {exc}")

    return {
        "sensor":          sensor_res,
        "facial":          facial_res,
        "future":          future_res,
        "shap_sensor":     shap_sensor_res,
        "shap_future":     shap_future_res,
        "gradcam":         gradcam_res,
        "recommendations": recs,
        "timestamp":       datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  HTML UI  (identical to v3.0 — fully compatible with updated API)
# ══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MindSense AI — Mental Health Assessment</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#070b14;--surface:#0d1220;--surface2:#131929;--surface3:#1a2235;
  --border:#1e2d47;--border2:#263550;
  --teal:#00e5cc;--teal-dim:#00b4a2;--teal-glow:rgba(0,229,204,.12);
  --coral:#ff5e7d;--coral-dim:#cc3d5c;--coral-glow:rgba(255,94,125,.10);
  --amber:#ffb547;--violet:#9f6ef5;--violet-glow:rgba(159,110,245,.10);
  --blue:#4a9eff;--green:#2ecc8a;--green-glow:rgba(46,204,138,.10);
  --text:#e8edf5;--text2:#8a95a8;--text3:#4a5568;
  --font-body:'Space Grotesk',sans-serif;
  --font-serif:'Instrument Serif',serif;
  --font-mono:'JetBrains Mono',monospace;
  --r:14px;--r-sm:8px;
  --shadow:0 8px 40px rgba(0,0,0,.5);
  --glow-teal:0 0 30px rgba(0,229,204,.15);
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:var(--font-body);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;pointer-events:none;
  background:radial-gradient(ellipse 80% 50% at 10% 0%,rgba(0,229,204,.05) 0%,transparent 60%),
             radial-gradient(ellipse 60% 40% at 90% 100%,rgba(159,110,245,.05) 0%,transparent 60%);z-index:0;}
header{position:sticky;top:0;z-index:100;background:rgba(7,11,20,.85);backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border);padding:0 40px;height:64px;display:flex;align-items:center;justify-content:space-between;}
.logo{display:flex;align-items:center;gap:10px;font-size:1.15rem;font-weight:700;letter-spacing:-.02em;}
.logo-mark{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--teal),var(--violet));display:flex;align-items:center;justify-content:center;font-size:.9rem;}
.logo-name em{color:var(--teal);font-style:normal}
.status-bar{display:flex;gap:8px;flex-wrap:wrap}
.chip{display:flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-size:.72rem;font-weight:500;background:var(--surface2);border:1px solid var(--border);color:var(--text2);}
.chip-dot{width:6px;height:6px;border-radius:50%;background:#ef4444;flex-shrink:0}
.chip-dot.ok{background:var(--teal);box-shadow:0 0 6px var(--teal)}
.chip-dot.pulse{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
main{max-width:1400px;margin:0 auto;padding:40px;position:relative;z-index:1}
.sec-head{display:flex;align-items:center;gap:16px;margin-bottom:24px;}
.sec-head h2{font-family:var(--font-serif);font-size:1.6rem;font-weight:400;font-style:italic;color:var(--text);}
.sec-head-line{flex:1;height:1px;background:linear-gradient(to right,var(--border),transparent)}
.sec-badge{padding:3px 10px;border-radius:999px;font-size:.7rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;}
.badge-teal{background:var(--teal-glow);border:1px solid rgba(0,229,204,.3);color:var(--teal)}
.badge-coral{background:var(--coral-glow);border:1px solid rgba(255,94,125,.3);color:var(--coral)}
.badge-violet{background:var(--violet-glow);border:1px solid rgba(159,110,245,.3);color:var(--violet)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;transition:border-color .3s,box-shadow .3s;}
.card:hover{border-color:var(--border2)}
.card-head{padding:18px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;}
.card-head h3{font-size:.9rem;font-weight:600;color:var(--text)}
.card-icon{width:34px;height:34px;border-radius:9px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:1rem;}
.ci-teal{background:var(--teal-glow);border:1px solid rgba(0,229,204,.2)}
.ci-coral{background:var(--coral-glow);border:1px solid rgba(255,94,125,.2)}
.ci-amber{background:rgba(255,181,71,.08);border:1px solid rgba(255,181,71,.2)}
.ci-violet{background:var(--violet-glow);border:1px solid rgba(159,110,245,.2)}
.ci-blue{background:rgba(74,158,255,.08);border:1px solid rgba(74,158,255,.2)}
.ci-green{background:var(--green-glow);border:1px solid rgba(46,204,138,.2)}
.card-body{padding:22px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media(max-width:1100px){.grid-2{grid-template-columns:1fr}}
@media(max-width:780px){.grid-3{grid-template-columns:1fr 1fr}}
@media(max-width:560px){.grid-3{grid-template-columns:1fr}}
#video{width:100%;border-radius:10px;background:#000;display:none;aspect-ratio:4/3;object-fit:cover}
#canvas{display:none}
#snap-preview{width:100%;border-radius:10px;display:none;border:1px solid var(--teal)}
.cam-ph{aspect-ratio:4/3;background:var(--surface2);border:1px dashed var(--border2);border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;cursor:pointer;transition:border-color .2s,background .2s;}
.cam-ph:hover{border-color:var(--teal);background:var(--teal-glow)}
.cam-ph svg{width:44px;opacity:.3}
.cam-ph span{font-size:.85rem;color:var(--text2)}
.field{margin-bottom:16px}
.field-label{display:flex;justify-content:space-between;align-items:center;font-size:.75rem;font-weight:500;color:var(--text2);margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em;}
.field-label .val{font-family:var(--font-mono);font-size:.8rem;color:var(--teal);font-weight:500;text-transform:none;letter-spacing:0;background:var(--teal-glow);padding:1px 8px;border-radius:4px;}
.field-group{margin-bottom:22px}
.field-group-title{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--text3);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border);}
input[type=range]{width:100%;-webkit-appearance:none;height:4px;border-radius:2px;outline:none;cursor:pointer;background:linear-gradient(to right,var(--teal) 0%,var(--teal) var(--pct,50%),var(--border2) var(--pct,50%));}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--text);border:2.5px solid var(--teal);cursor:pointer;box-shadow:0 0 8px var(--teal);transition:box-shadow .2s;}
input[type=range]::-webkit-slider-thumb:hover{box-shadow:0 0 14px var(--teal)}
.btn{padding:10px 20px;border:none;border-radius:var(--r-sm);font-family:var(--font-body);font-size:.85rem;font-weight:600;cursor:pointer;transition:all .2s;display:inline-flex;align-items:center;gap:8px;}
.btn-primary{background:linear-gradient(135deg,var(--teal-dim),var(--teal));color:#070b14;letter-spacing:.01em;}
.btn-primary:hover{filter:brightness(1.1);transform:translateY(-1px);box-shadow:var(--glow-teal)}
.btn-ghost{background:var(--surface2);color:var(--text2);border:1px solid var(--border)}
.btn-ghost:hover{border-color:var(--teal);color:var(--teal)}
.btn:disabled{opacity:.35;pointer-events:none}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.btn-full{width:100%;justify-content:center;padding:13px 20px;font-size:.9rem;margin-top:6px}
.spinner-wrap{padding:80px 0;display:flex;flex-direction:column;align-items:center;gap:16px;}
.spinner{width:44px;height:44px;border:3px solid var(--border2);border-top-color:var(--teal);border-radius:50%;animation:spin 1s linear infinite;}
.spinner-txt{font-size:.85rem;color:var(--text2);font-family:var(--font-mono)}
@keyframes spin{to{transform:rotate(360deg)}}
.hero-result{display:flex;align-items:center;gap:20px;padding:22px;background:var(--surface2);border-radius:12px;border:1px solid var(--border2);margin-bottom:18px;}
.hero-icon{font-size:2.4rem;flex-shrink:0;filter:drop-shadow(0 0 10px currentColor)}
.hero-label{font-family:var(--font-serif);font-size:1.8rem;font-style:italic;color:var(--text);line-height:1.1;}
.hero-sub{font-size:.78rem;color:var(--text2);margin-top:4px}
.hero-conf{margin-left:auto;text-align:right;flex-shrink:0}
.hero-conf .big{font-family:var(--font-mono);font-size:2rem;font-weight:500;color:var(--teal);display:block;line-height:1;}
.hero-conf small{font-size:.68rem;color:var(--text3);margin-top:2px;display:block}
.prob-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:.8rem}
.prob-name{width:120px;flex-shrink:0;color:var(--text2);font-weight:500;font-size:.78rem}
.prob-track{flex:1;height:6px;background:var(--surface3);border-radius:3px;overflow:hidden}
.prob-fill{height:100%;border-radius:3px;transition:width .9s cubic-bezier(.22,1,.36,1)}
.pf-teal{background:linear-gradient(to right,var(--teal-dim),var(--teal))}
.pf-coral{background:linear-gradient(to right,var(--coral-dim),var(--coral))}
.pf-amber{background:var(--amber)}.pf-violet{background:var(--violet)}.pf-blue{background:var(--blue)}.pf-green{background:var(--green)}
.prob-pct{width:38px;text-align:right;font-family:var(--font-mono);font-size:.75rem;color:var(--text3)}
.sensor-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:16px}
@media(max-width:780px){.sensor-grid{grid-template-columns:repeat(3,1fr)}}
.stile{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px 10px;text-align:center;transition:border-color .2s;}
.stile:hover{border-color:var(--border2)}
.stile .sval{font-family:var(--font-mono);font-size:1.1rem;font-weight:500;color:var(--teal)}
.stile .slbl{font-size:.65rem;color:var(--text3);margin-top:3px;font-weight:500}
.timeline{display:flex;overflow-x:auto;padding-bottom:4px}
.tl-step{flex:1;min-width:140px;padding:18px 14px;text-align:center;border-right:1px solid var(--border);position:relative;transition:background .2s;}
.tl-step:hover{background:var(--surface2)}.tl-step:last-child{border-right:none}
.tl-num{font-size:.65rem;color:var(--text3);font-family:var(--font-mono);font-weight:500;letter-spacing:.06em;margin-bottom:8px;text-transform:uppercase;}
.tl-dot{width:10px;height:10px;border-radius:50%;margin:0 auto 8px}
.dot-normal{background:var(--green);box-shadow:0 0 8px var(--green)}.dot-low{background:var(--blue);box-shadow:0 0 8px var(--blue)}
.dot-stress{background:var(--amber);box-shadow:0 0 8px var(--amber)}.dot-anxiety{background:var(--coral);box-shadow:0 0 8px var(--coral)}
.dot-panic{background:#ff2d55;box-shadow:0 0 8px #ff2d55;animation:pulse 1.5s infinite}.dot-depression{background:var(--violet);box-shadow:0 0 8px var(--violet)}
.tl-state{font-size:.78rem;font-weight:600;margin-bottom:8px}.tl-vals{font-size:.66rem;color:var(--text3);font-family:var(--font-mono);line-height:1.8}
.shap-block{margin-bottom:22px}
.shap-class-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:10px;display:flex;align-items:center;gap:8px;}
.shap-class-label::after{content:'';flex:1;height:1px;background:var(--border)}
.shap-row{display:flex;align-items:center;gap:10px;margin-bottom:7px;font-size:.78rem}
.shap-name{width:80px;flex-shrink:0;color:var(--text2)}.shap-track{flex:1;height:5px;background:var(--surface3);border-radius:2px;overflow:hidden}
.shap-fill{height:100%;background:linear-gradient(to right,#7c3aed,#a78bfa);border-radius:2px}
.shap-val{width:56px;text-align:right;font-family:var(--font-mono);font-size:.72rem;color:var(--text3)}
.fshap-step{margin-bottom:14px;background:var(--surface2);border-radius:10px;border:1px solid var(--border);overflow:hidden;}
.fshap-head{padding:10px 14px;display:flex;align-items:center;gap:10px;cursor:pointer;border-bottom:1px solid transparent;transition:border-color .2s;}
.fshap-head:hover{border-bottom-color:var(--border)}.fshap-step-label{font-family:var(--font-mono);font-size:.75rem;font-weight:500;color:var(--teal)}
.fshap-state{font-size:.78rem;font-weight:600;margin-left:4px}.fshap-arrow{margin-left:auto;font-size:.7rem;color:var(--text3);transition:transform .2s}
.fshap-body{padding:14px;display:none}.fshap-body.open{display:block}
.fshap-head.active .fshap-arrow{transform:rotate(180deg)}
.fshap-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:.75rem}
.fshap-name{width:80px;flex-shrink:0;color:var(--text2);font-size:.72rem}.fshap-track{flex:1;height:4px;background:var(--surface3);border-radius:2px;overflow:hidden}
.fshap-fill{height:100%;background:linear-gradient(to right,var(--teal-dim),var(--teal));border-radius:2px}
.fshap-val{width:56px;text-align:right;font-family:var(--font-mono);font-size:.7rem;color:var(--text3)}
.change-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:16px}
@media(max-width:780px){.change-grid{grid-template-columns:repeat(3,1fr)}}
.change-tile{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center;}
.change-tile .cv{font-family:var(--font-mono);font-size:.85rem;font-weight:500}.change-tile .cl{font-size:.62rem;color:var(--text3);margin-top:2px}
.cv-up{color:var(--coral)}.cv-dn{color:var(--green)}.cv-fl{color:var(--text2)}
.trend-badge{display:flex;align-items:center;gap:10px;padding:12px 16px;border-radius:10px;margin-bottom:16px;font-size:.85rem;font-weight:500;}
.trend-worsen{background:rgba(255,94,125,.08);border:1px solid rgba(255,94,125,.25);color:var(--coral)}
.trend-improve{background:rgba(46,204,138,.08);border:1px solid rgba(46,204,138,.25);color:var(--green)}
.trend-stable{background:var(--surface2);border:1px solid var(--border);color:var(--text2)}
.emotion-tip{padding:10px 14px;border-radius:8px;font-size:.83rem;margin-bottom:16px;background:var(--violet-glow);border-left:3px solid var(--violet);color:var(--text);}
.rec-section{margin-bottom:18px}
.rec-label{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:8px;display:flex;align-items:center;gap:8px;}
.rec-tag{padding:2px 7px;border-radius:999px;font-size:.65rem;font-weight:600;margin-left:6px;}
.tag-red{background:rgba(255,94,125,.12);border:1px solid rgba(255,94,125,.3);color:var(--coral)}
.tag-green{background:rgba(46,204,138,.1);border:1px solid rgba(46,204,138,.3);color:var(--green)}
.tag-blue{background:rgba(74,158,255,.1);border:1px solid rgba(74,158,255,.3);color:var(--blue)}
.rec-list{list-style:none;display:flex;flex-direction:column;gap:6px}
.rec-list li{padding:9px 12px;border-radius:8px;font-size:.83rem;line-height:1.55;background:var(--surface2);border-left:2px solid var(--border2);transition:border-color .2s;}
.rec-first-aid li{border-left-color:var(--coral)}.rec-lifestyle li{border-left-color:var(--teal)}.rec-professional li{border-left-color:var(--violet)}
.rec-list li:hover{background:var(--surface3)}
#gradcam-canvas{border-radius:8px;width:100%;margin-top:10px;display:none}
.notice{padding:10px 14px;border-radius:8px;font-size:.82rem;background:rgba(255,181,71,.07);border:1px solid rgba(255,181,71,.2);color:var(--amber);margin-bottom:14px;}
.hidden{display:none!important}
.fade-in{animation:fadeIn .5s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
section{margin-bottom:40px}
kbd{background:var(--surface2);border:1px solid var(--border2);border-radius:4px;padding:1px 6px;font-family:var(--font-mono);font-size:.75rem;color:var(--text2);}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-mark">🧠</div>
    <div class="logo-name">Mind<em>Sense</em> AI</div>
  </div>
  <div class="status-bar" id="status-bar">
    <div class="chip" id="chip-eeg" style="cursor:pointer;" onclick="toggleEEG()"><div class="chip-dot" id="d-eeg"></div>EEG LIVE</div>
    <div class="chip" id="chip-ard" style="cursor:pointer;" onclick="toggleArduino()"><div class="chip-dot" id="d-ard"></div>ARDUINO LIVE</div>
    <div class="chip"><div class="chip-dot pulse" id="d-cnn"></div>Facial CNN</div>
    <div class="chip"><div class="chip-dot pulse" id="d-rnn"></div>RNN Sensor</div>
    <div class="chip"><div class="chip-dot pulse" id="d-pred"></div>Predictor</div>
    <div class="chip"><div class="chip-dot pulse" id="d-scaler"></div>Scaler</div>
    <div class="chip"><div class="chip-dot pulse" id="d-shap"></div>SHAP</div>
  </div>
</header>
<main>
<section>
  <div class="sec-head"><h2>Patient Input</h2><div class="sec-head-line"></div><span class="sec-badge badge-teal">Step 1</span></div>
  <div class="grid-2">
    <div class="card">
      <div class="card-head"><div class="card-icon ci-teal">📸</div><h3>Facial Expression <span style="font-weight:400;color:var(--text3);font-size:.78rem">(optional)</span></h3></div>
      <div class="card-body">
        <div id="cam-ph" class="cam-ph" onclick="startCam()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
          <span>Click to enable webcam</span>
        </div>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
        <img id="snap-preview" alt="Captured frame">
        <div class="btn-row">
          <button class="btn btn-ghost hidden" id="btn-cap" onclick="capture()">📸 Capture <kbd>C</kbd></button>
          <button class="btn btn-ghost hidden" id="btn-ret" onclick="retake()">↺ Retake</button>
          <button class="btn btn-ghost hidden" id="btn-stop" onclick="stopCam()">✕ Stop</button>
        </div>
        <p style="font-size:.73rem;color:var(--text3);margin-top:10px">Webcam is optional. Facial emotion will be skipped if not captured.</p>
      </div>
    </div>
    <div class="card">
      <div class="card-head"><div class="card-icon ci-coral">📡</div><h3>Physiological Sensors</h3></div>
      <div class="card-body">
        <form id="sform" onsubmit="analyze(event)">
          <div class="field-group">
            <div class="field-group-title">Biometric Signals</div>
            <div class="field">
              <div class="field-label">GSR — Galvanic Skin Response <span class="val" id="lv-gsr">2000</span></div>
              <input type="range" id="gsr" min="0" max="40952" step="1" value="2000" oninput="syncSlider('gsr','lv-gsr',this.value,'')">
            </div>
            <div class="field">
              <div class="field-label">PPG — Heart Rate <span class="val" id="lv-ppg">72 bpm</span></div>
              <input type="range" id="ppg" min="40" max="200" step="1" value="72" oninput="syncSlider('ppg','lv-ppg',this.value,' bpm')">
            </div>
          </div>
          <div class="field-group">
            <div class="field-group-title">EEG Band Power (normalised 0–1)</div>
            <div class="grid-2" style="gap:10px">
              <div class="field"><div class="field-label">Delta <span class="val" id="lv-delta">0.30</span></div><input type="range" id="delta" min="0" max="1" step="0.01" value="0.30" oninput="syncSlider('delta','lv-delta',parseFloat(this.value).toFixed(2),'')"></div>
              <div class="field"><div class="field-label">Theta <span class="val" id="lv-theta">0.25</span></div><input type="range" id="theta" min="0" max="1" step="0.01" value="0.25" oninput="syncSlider('theta','lv-theta',parseFloat(this.value).toFixed(2),'')"></div>
              <div class="field"><div class="field-label">Low Alpha <span class="val" id="lv-la">0.40</span></div><input type="range" id="low_alpha" min="0" max="1" step="0.01" value="0.40" oninput="syncSlider('low_alpha','lv-la',parseFloat(this.value).toFixed(2),'')"></div>
              <div class="field"><div class="field-label">High Alpha <span class="val" id="lv-ha">0.35</span></div><input type="range" id="high_alpha" min="0" max="1" step="0.01" value="0.35" oninput="syncSlider('high_alpha','lv-ha',parseFloat(this.value).toFixed(2),'')"></div>
              <div class="field"><div class="field-label">Low Beta <span class="val" id="lv-lb">0.20</span></div><input type="range" id="low_beta" min="0" max="1" step="0.01" value="0.20" oninput="syncSlider('low_beta','lv-lb',parseFloat(this.value).toFixed(2),'')"></div>
              <div class="field"><div class="field-label">High Beta <span class="val" id="lv-hb">0.18</span></div><input type="range" id="high_beta" min="0" max="1" step="0.01" value="0.18" oninput="syncSlider('high_beta','lv-hb',parseFloat(this.value).toFixed(2),'')"></div>
              <div class="field"><div class="field-label">Low Gamma <span class="val" id="lv-lg">0.12</span></div><input type="range" id="low_gamma" min="0" max="1" step="0.01" value="0.12" oninput="syncSlider('low_gamma','lv-lg',parseFloat(this.value).toFixed(2),'')"></div>
              <div class="field"><div class="field-label">Mid Gamma <span class="val" id="lv-mg">0.10</span></div><input type="range" id="mid_gamma" min="0" max="1" step="0.01" value="0.10" oninput="syncSlider('mid_gamma','lv-mg',parseFloat(this.value).toFixed(2),'')"></div>
            </div>
          </div>
          <button type="submit" class="btn btn-primary btn-full">🔬 Run Full Assessment</button>
        </form>
      </div>
    </div>
  </div>
</section>
<div id="results" class="hidden">
  <div id="spinner" class="spinner-wrap"><div class="spinner"></div><div class="spinner-txt">Analysing biosignals…</div></div>
  <div id="res-content" class="hidden">
    <section>
      <div class="sec-head"><h2>Current Assessment</h2><div class="sec-head-line"></div><span class="sec-badge badge-coral">Live</span></div>
      <div class="grid-2">
        <div class="card fade-in">
          <div class="card-head"><div class="card-icon ci-coral">🧠</div><h3>Mental State</h3></div>
          <div class="card-body">
            <div class="hero-result"><div class="hero-icon" id="state-icon">🧠</div>
              <div><div class="hero-label" id="state-label">—</div><div class="hero-sub">BiLSTM + GRU + Attention · 10 sensor channels</div></div>
              <div class="hero-conf"><span class="big" id="state-conf">—</span><small>confidence</small></div>
            </div>
            <div id="state-probs"></div>
            <div class="sensor-grid" id="sensor-tiles"></div>
          </div>
        </div>
        <div class="card fade-in">
          <div class="card-head"><div class="card-icon ci-teal">😊</div><h3>Facial Emotion</h3></div>
          <div class="card-body">
            <div id="no-facial" class="notice">No image captured — sensor-only mode</div>
            <div id="yes-facial" class="hidden">
              <div class="hero-result"><div class="hero-icon" id="emotion-icon">😊</div>
                <div><div class="hero-label" id="emotion-label">—</div><div class="hero-sub">Residual CNN · GradCAM XAI</div></div>
                <div class="hero-conf"><span class="big" id="emotion-conf">—</span><small>confidence</small></div>
              </div>
              <div id="emotion-probs"></div>
              <div id="gradcam-wrap" class="hidden"><p style="font-size:.72rem;color:var(--text3);margin-bottom:4px">GradCAM attention heatmap</p><canvas id="gradcam-canvas"></canvas></div>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section>
      <div class="sec-head"><h2>5-Step Physiological Forecast</h2><div class="sec-head-line"></div><span class="sec-badge badge-violet">Seq2Seq LSTM</span></div>
      <div class="card fade-in">
        <div class="card-head"><div class="card-icon ci-amber">🔮</div><h3>Future State Timeline</h3></div>
        <div class="card-body"><div id="change-summary" class="change-grid"></div><div class="timeline" id="timeline"></div></div>
      </div>
    </section>
    <section>
      <div class="sec-head"><h2>Explainability (XAI)</h2><div class="sec-head-line"></div><span class="sec-badge badge-violet">SHAP</span></div>
      <div class="grid-2">
        <div class="card fade-in"><div class="card-head"><div class="card-icon ci-violet">🔍</div><h3>Current State — SHAP Feature Importance</h3></div><div class="card-body" id="shap-sensor-content"><p style="color:var(--text3);font-size:.82rem">Loading SHAP…</p></div></div>
        <div class="card fade-in"><div class="card-head"><div class="card-icon ci-blue">📊</div><h3>Future Prediction — SHAP per Step</h3></div><div class="card-body" id="shap-future-content"><p style="color:var(--text3);font-size:.82rem">Loading SHAP…</p></div></div>
      </div>
    </section>
    <section>
      <div class="sec-head"><h2>AI Recommendations</h2><div class="sec-head-line"></div><span class="sec-badge badge-teal">Personalised</span></div>
      <div class="card fade-in"><div class="card-head"><div class="card-icon ci-green">💡</div><h3>Health Guidance</h3></div><div class="card-body" id="rec-body"></div></div>
    </section>
  </div>
</div>
</main>
<script>
let stream=null,blob=null;
let eegInterval=null, ardInterval=null;

async function fetchEEG() {
  try {
    const res = await fetch('/api/eeg/live');
    if (!res.ok) throw new Error('network error');
    const data = await res.json();
    document.getElementById('d-eeg').className = 'chip-dot ' + (data.online ? 'ok' : '');
    if (data.online) {
        ['delta','theta','low_alpha','high_alpha','low_beta','high_beta','low_gamma','mid_gamma'].forEach(k => {
            if (data[k] !== undefined && data[k] !== null) {
                const el = document.getElementById(k);
                if (el) { el.value = data[k]; syncSlider(k, 'lv-'+(k==='low_alpha'?'la':k==='high_alpha'?'ha':k==='low_beta'?'lb':k==='high_beta'?'hb':k==='low_gamma'?'lg':k==='mid_gamma'?'mg':k), data[k].toFixed(2), ''); }
            }
        });
    }
  } catch (err) {
    document.getElementById('d-eeg').className = 'chip-dot';
  }
}

async function fetchArduino() {
  try {
    const res = await fetch('/api/arduino/live');
    if (!res.ok) throw new Error('network error');
    const data = await res.json();
    document.getElementById('d-ard').className = 'chip-dot ' + (data.online ? 'ok' : '');
    if (data.online) {
        if (data.gsr !== undefined && data.gsr !== null) {
            const el = document.getElementById('gsr');
            if (el) { el.value = data.gsr; syncSlider('gsr', 'lv-gsr', Math.round(data.gsr), ''); }
        }
        if (data.ppg !== undefined && data.ppg !== null) {
            const el = document.getElementById('ppg');
            if (el) { el.value = data.ppg; syncSlider('ppg', 'lv-ppg', Math.round(data.ppg), ' bpm'); }
        }
    }
  } catch (err) {
    document.getElementById('d-ard').className = 'chip-dot';
  }
}

function toggleEEG() {
  if (eegInterval) { clearInterval(eegInterval); eegInterval = null; fetch('/api/eeg/stop', {method: 'POST'}); document.getElementById('d-eeg').className = 'chip-dot'; document.getElementById('chip-eeg').style.borderColor='var(--border)'; }
  else { eegInterval = setInterval(fetchEEG, 1000); fetchEEG(); document.getElementById('chip-eeg').style.borderColor='var(--teal)'; }
}

function toggleArduino() {
  if (ardInterval) { clearInterval(ardInterval); ardInterval = null; fetch('/api/arduino/stop', {method: 'POST'}); document.getElementById('d-ard').className = 'chip-dot'; document.getElementById('chip-ard').style.borderColor='var(--border)'; }
  else { fetch('/api/arduino/start', {method: 'POST'}); ardInterval = setInterval(fetchArduino, 1000); fetchArduino(); document.getElementById('chip-ard').style.borderColor='var(--coral)'; }
}

function startCam(){navigator.mediaDevices.getUserMedia({video:{width:640,height:480}}).then(s=>{stream=s;const v=document.getElementById('video');v.srcObject=s;v.style.display='block';hide('cam-ph');show('btn-cap');show('btn-stop');}).catch(e=>alert('Camera: '+e.message));}
function stopCam(){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}hide('video');hide('btn-cap');hide('btn-stop');show('cam-ph');}
function capture(){const v=document.getElementById('video'),c=document.getElementById('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);c.toBlob(b=>{blob=b;const img=document.getElementById('snap-preview');img.src=URL.createObjectURL(b);img.style.display='block';hide('video');hide('btn-cap');hide('btn-stop');show('btn-ret');if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}},'image/jpeg',.95);}
function retake(){blob=null;document.getElementById('snap-preview').style.display='none';hide('btn-ret');show('cam-ph');}
document.addEventListener('keydown',e=>{if((e.key==='c'||e.key==='C')&&!document.getElementById('btn-cap').classList.contains('hidden'))capture();});
function syncSlider(id,lblId,val,unit){document.getElementById(lblId).textContent=val+unit;const el=document.getElementById(id);const min=+el.min,max=+el.max;const pct=((+val-min)/(max-min)*100).toFixed(2)+'%';el.style.setProperty('--pct',pct);}
['gsr','ppg','delta','theta','low_alpha','high_alpha','low_beta','high_beta','low_gamma','mid_gamma'].forEach(id=>{const el=document.getElementById(id);if(el)el.dispatchEvent(new Event('input'));});
function show(id){document.getElementById(id)?.classList.remove('hidden')}
function hide(id){document.getElementById(id)?.classList.add('hidden')}
function pct(v){return(v*100).toFixed(1)}
const FILL_CLS=['pf-teal','pf-coral','pf-amber','pf-violet','pf-blue','pf-green'];
function probBars(el,probs,cls){el.innerHTML='';Object.entries(probs).sort((a,b)=>b[1]-a[1]).forEach(([k,v],i)=>{const c=cls?cls[i%cls.length]:FILL_CLS[i%FILL_CLS.length];el.innerHTML+=`<div class="prob-row"><span class="prob-name">${k}</span><div class="prob-track"><div class="prob-fill ${c}" style="width:${pct(v)}%"></div></div><span class="prob-pct">${pct(v)}%</span></div>`;});}
async function pollStatus(){try{const d=await(await fetch('/api/status')).json();const map={facial_cnn:'d-cnn',rnn_sensor:'d-rnn',predictor:'d-pred',scaler:'d-scaler',shap_bg:'d-shap'};for(const[k,id] of Object.entries(map)){const dot=document.getElementById(id);if(dot){dot.className='chip-dot '+(d.models[k]?'ok':'pulse');}}}catch(e){}}
pollStatus();setInterval(pollStatus,30000);
async function analyze(e){e.preventDefault();show('results');document.getElementById('spinner').style.display='flex';hide('res-content');const fd=new FormData();['gsr','ppg','delta','theta','low_alpha','high_alpha','low_beta','high_beta','low_gamma','mid_gamma'].forEach(id=>fd.append(id,document.getElementById(id).value));if(blob)fd.append('file',blob,'frame.jpg');try{const res=await fetch('/api/predict/complete',{method:'POST',body:fd});if(!res.ok){const err=await res.json();throw new Error(err.detail||'Server error');}renderResults(await res.json());}catch(err){alert('Assessment failed: '+err.message);}finally{document.getElementById('spinner').style.display='none';}}
const STATE_ICONS={NORMAL:'✅',LOW_STRESS:'😌',MODERATE_STRESS:'😟',HIGH_ANXIETY:'😰',PANIC_STATE:'🆘',DEPRESSION:'💙'};
const STATE_COLORS={NORMAL:'var(--green)',LOW_STRESS:'var(--blue)',MODERATE_STRESS:'var(--amber)',HIGH_ANXIETY:'var(--coral)',PANIC_STATE:'#ff2d55',DEPRESSION:'var(--violet)'};
function renderResults(d){
  const s=d.sensor;
  document.getElementById('state-label').textContent=s.state.replace(/_/g,' ');
  document.getElementById('state-conf').textContent=pct(s.confidence)+'%';
  document.getElementById('state-icon').textContent=STATE_ICONS[s.state]||'🧠';
  probBars(document.getElementById('state-probs'),Object.fromEntries(Object.entries(s.probabilities).map(([k,v])=>[k.replace(/_/g,' '),v])),['pf-teal','pf-coral','pf-amber','pf-violet','pf-blue','pf-green']);
  const tk=['gsr','ppg','delta','theta','low_alpha','high_alpha','low_beta','high_beta','low_gamma','mid_gamma'];
  const tn=['GSR','PPG','Delta','Theta','Lo-α','Hi-α','Lo-β','Hi-β','Lo-γ','Mid-γ'];
  const inp={};tk.forEach(k=>{inp[k]=+document.getElementById(k).value;});
  const sg=document.getElementById('sensor-tiles');sg.innerHTML=tk.map((k,i)=>`<div class="stile"><div class="sval">${k==='gsr'?Math.round(inp[k]):parseFloat(inp[k]).toFixed(2)}</div><div class="slbl">${tn[i]}</div></div>`).join('');
  if(d.facial){hide('no-facial');show('yes-facial');document.getElementById('emotion-label').textContent=d.facial.emotion;document.getElementById('emotion-conf').textContent=pct(d.facial.confidence)+'%';document.getElementById('emotion-icon').textContent={Angry:'😠',Disgust:'🤢',Fear:'😨',Happy:'😊',Neutral:'😐',Sad:'😢',Surprise:'😲'}[d.facial.emotion]||'😐';probBars(document.getElementById('emotion-probs'),d.facial.probabilities);if(d.gradcam&&d.gradcam.heatmap){show('gradcam-wrap');renderGradCam(d.gradcam.heatmap,d.gradcam.shape);}}else{show('no-facial');hide('yes-facial');hide('gradcam-wrap');}
  renderTimeline(d.future,inp);
  if(d.shap_sensor)renderShapSensor(d.shap_sensor);else document.getElementById('shap-sensor-content').innerHTML='<p style="color:var(--text3);font-size:.8rem">SHAP unavailable</p>';
  if(d.shap_future)renderShapFuture(d.shap_future,d.future);else document.getElementById('shap-future-content').innerHTML='<p style="color:var(--text3);font-size:.8rem">SHAP unavailable</p>';
  renderRec(d.recommendations);
  show('res-content');document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});
}
function renderTimeline(steps,currentInputs){
  const keys=['gsr','ppg','delta','theta','low_alpha','high_alpha','low_beta','high_beta','low_gamma','mid_gamma'];
  const labels=['GSR','PPG','Delta','Theta','Lo-α','Hi-α','Lo-β','Hi-β','Lo-γ','Mid-γ'];
  const lastStep=steps[steps.length-1];
  const cg=document.getElementById('change-summary');
  cg.innerHTML=keys.map((k,i)=>{const curr=currentInputs[k];const futVal=lastStep[k]!==undefined?lastStep[k]:lastStep[k.replace('_','')];if(futVal===undefined)return '';const delta=futVal-curr;const pctChg=curr!==0?(delta/Math.abs(curr)*100).toFixed(1):0;const cls=Math.abs(pctChg)<2?'cv-fl':delta>0?'cv-up':'cv-dn';const arrow=Math.abs(pctChg)<2?'→':delta>0?'↑':'↓';return `<div class="change-tile"><div class="cv ${cls}">${arrow} ${Math.abs(pctChg)}%</div><div class="cl">${labels[i]}</div></div>`;}).join('');
  const dot_cls={NORMAL:'dot-normal',LOW_STRESS:'dot-low',MODERATE_STRESS:'dot-stress',HIGH_ANXIETY:'dot-anxiety',PANIC_STATE:'dot-panic',DEPRESSION:'dot-depression'};
  document.getElementById('timeline').innerHTML=steps.map(step=>{const dc=dot_cls[step.state]||'dot-normal';const col=STATE_COLORS[step.state]||'var(--text2)';const vals=Object.entries(step).filter(([k])=>!['step','state','state_index'].includes(k)).map(([k,v])=>`${k.toUpperCase().substring(0,3)}: ${typeof v==='number'?v.toFixed(2):v}`).slice(0,5).join('<br>');return `<div class="tl-step"><div class="tl-num">Step ${step.step}</div><div class="tl-dot ${dc}"></div><div class="tl-state" style="color:${col}">${step.state.replace(/_/g,' ')}</div><div class="tl-vals">${vals}</div></div>`;}).join('');
}
function renderShapSensor(shap){const el=document.getElementById('shap-sensor-content');el.innerHTML='';for(const[cls,feats] of Object.entries(shap)){const maxV=Math.max(...Object.values(feats));let rows='';Object.entries(feats).sort((a,b)=>b[1]-a[1]).forEach(([f,v])=>{const p=maxV>0?(v/maxV*100).toFixed(1):0;rows+=`<div class="shap-row"><span class="shap-name">${f}</span><div class="shap-track"><div class="shap-fill" style="width:${p}%"></div></div><span class="shap-val">${v.toFixed(4)}</span></div>`;});el.innerHTML+=`<div class="shap-block"><div class="shap-class-label">${cls.replace(/_/g,' ')}</div>${rows}</div>`;}}
function renderShapFuture(shapData,futureSteps){const el=document.getElementById('shap-future-content');el.innerHTML='';const agg=shapData.aggregate_importance;const maxA=Math.max(...Object.values(agg));let aggHtml='<div class="shap-block"><div class="shap-class-label">Aggregate (all 5 steps)</div>';Object.entries(agg).sort((a,b)=>b[1]-a[1]).forEach(([f,v])=>{const p=maxA>0?(v/maxA*100).toFixed(1):0;aggHtml+=`<div class="shap-row"><span class="shap-name">${f}</span><div class="shap-track"><div class="shap-fill" style="width:${p}%;background:linear-gradient(to right,var(--teal-dim),var(--teal))"></div></div><span class="shap-val">${v.toFixed(4)}</span></div>`;});aggHtml+='</div>';el.innerHTML=aggHtml;shapData.per_step.forEach((stepData,i)=>{const state=(futureSteps&&futureSteps[i])?futureSteps[i].state.replace(/_/g,' '):'';const col=futureSteps&&futureSteps[i]?STATE_COLORS[futureSteps[i].state]:'var(--text2)';const maxS=Math.max(...Object.values(stepData.shap));let barHtml='';Object.entries(stepData.shap).sort((a,b)=>b[1]-a[1]).forEach(([f,v])=>{const p=maxS>0?(v/maxS*100).toFixed(1):0;barHtml+=`<div class="fshap-bar-row"><span class="fshap-name">${f}</span><div class="fshap-track"><div class="fshap-fill" style="width:${p}%"></div></div><span class="fshap-val">${v.toFixed(4)}</span></div>`;});el.innerHTML+=`<div class="fshap-step"><div class="fshap-head" onclick="toggleFshap(this)"><span class="fshap-step-label">Step ${stepData.step}</span><span class="fshap-state" style="color:${col}">${state}</span><span class="fshap-arrow">▼</span></div><div class="fshap-body">${barHtml}</div></div>`;});}
function toggleFshap(head){head.classList.toggle('active');head.nextElementSibling.classList.toggle('open');}
function renderGradCam(hm,shape){const canvas=document.getElementById('gradcam-canvas');canvas.style.display='block';const[H,W]=shape;canvas.width=W;canvas.height=H;const ctx=canvas.getContext('2d');const img=ctx.createImageData(W,H);for(let i=0;i<hm.length;i++){const v=hm[i];img.data[i*4]=Math.round(Math.min(255,v*2*255));img.data[i*4+1]=Math.round(Math.min(255,(1-Math.abs(v-.5)*2)*180));img.data[i*4+2]=Math.round(Math.min(255,(1-v)*2*255));img.data[i*4+3]=160;}ctx.putImageData(img,0,0);}
function renderRec(r){const body=document.getElementById('rec-body');const tClass=r.trend==='worsening'?'trend-worsen':r.trend==='improving'?'trend-improve':'trend-stable';const tIcon=r.trend==='worsening'?'📉':r.trend==='improving'?'📈':'📊';let html=`<div class="trend-badge ${tClass}">${tIcon} ${r.trend_message}</div>`;if(r.emotion_tip)html+=`<div class="emotion-tip">${r.emotion_tip}</div>`;html+=`<div class="rec-section"><div class="rec-label">🚨 First Aid <span class="rec-tag tag-red">Immediate</span></div><ul class="rec-list rec-first-aid">${r.first_aid.map(i=>`<li>${i}</li>`).join('')}</ul></div>`;html+=`<div class="rec-section"><div class="rec-label">🌿 Lifestyle <span class="rec-tag tag-green">Daily</span></div><ul class="rec-list rec-lifestyle">${r.lifestyle.map(i=>`<li>${i}</li>`).join('')}</ul></div>`;html+=`<div class="rec-section"><div class="rec-label">🩺 Professional <span class="rec-tag tag-blue">When needed</span></div><ul class="rec-list rec-professional">${r.professional.map(i=>`<li>${i}</li>`).join('')}</ul></div>`;body.innerHTML=html;}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML

if __name__ == "__main__":
    print("\n" + "═" * 64)
    print("  MindSense AI — Mental Health Assessment Server  v4.0")
    print("  Aligned with: unified_training_final_fixed.py")
    print("═" * 64)
    print(f"  UI   →  http://localhost:8000")
    print(f"  Docs →  http://localhost:8000/docs")
    print(f"  API  →  http://localhost:8000/api/status")
    print("═" * 64 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
