from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Optional external providers (keep keys on the backend machine)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_OUTPUT_DIM = int(os.getenv("GEMINI_OUTPUT_DIM", "768"))  # 128..3072 (recommended 768/1536/3072)
GEMINI_TASK_TYPE = os.getenv("GEMINI_TASK_TYPE", "RETRIEVAL_QUERY")  # RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT / ...


# ---------------- Utilities ----------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_issue_dt(s: Optional[str]) -> Optional[np.datetime64]:
    if not s or not isinstance(s, str):
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", s)
    if not m:
        return None
    y, mo, d, hh, mm, ss = map(int, m.groups())
    return np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}")


def _within_last_days(d: np.datetime64, ref: np.datetime64, days: int) -> bool:
    delta = ref - d
    return (delta >= np.timedelta64(0, "s")) and (delta <= np.timedelta64(days, "D"))


def _fmt_date(s: Optional[str]) -> str:
    return (s or "")[:10]


def _parse_issue_id_parts(issueld: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = re.match(r"^([A-Z0-9]{4})-(\d{4})-QT-(\d{6})$", str(issueld or ""))
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def _infer_sorting_action(issue_like: Dict[str, Any]) -> str:
    """
    Heuristic tag used for the UI. Real systems should use explicit fields.
    """
    bi = int(str(issue_like.get("bi") or "0").strip() or "0")
    et = ""
    ev = issue_like.get("issueEvents") or []
    if isinstance(ev, list) and ev and isinstance(ev[0], dict):
        et = str(ev[0].get("eventType") or "")
    text = f"{issue_like.get('title','')} {issue_like.get('description','')} {et}".lower()

    if "sorting initiated" in text or "sorting action" in text or "100%" in text or "quarantine" in text:
        return "Sorting action"
    risky = ["open circuit","short","leak","pressure","not responding","dropout","thread damage","cold solder","insufficient solder"]
    if bi >= 6 or any(k in text for k in risky):
        return "Sorting action"
    return "No sorting action"


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _extract_rca_hints(description: str) -> List[str]:
    if not description:
        return []
    hints = []
    for line in str(description).splitlines():
        if "RCA_HINT:" in line:
            hints.append(line.split("RCA_HINT:", 1)[1].strip())
    return [h for h in hints if h]


def _heuristic_root_causes(title: str, description: str, event_type: str) -> List[str]:
    t = f"{title} {description} {event_type}".lower()
    # Minimal reasonable heuristic set
    if any(k in t for k in ["open circuit", "cold solder", "insufficient solder", "pcb", "no wetting"]):
        return [
            "Insufficient solder wetting due to reflow profile deviation",
            "Contamination on pads / flux residue affecting solderability",
            "Component lead coplanarity or placement issue causing weak joint",
        ]
    if any(k in t for k in ["scratch", "blemish", "handling mark", "surface"]):
        return [
            "Handling damage due to missing protective packaging",
            "Contact with fixture/transport tray during material flow",
            "Inadequate handling procedure adherence at station",
        ]
    if any(k in t for k in ["leak", "seepage", "pressure test", "coolant"]):
        return [
            "Seal seating issue due to clamp position variation",
            "Torque / clamp force out of specification",
            "Surface finish deviation at sealing interface",
        ]
    if any(k in t for k in ["thread", "torque", "cross-thread", "tool resistance"]):
        return [
            "Tool misalignment causing cross-threading",
            "Supplier machining burrs / thread quality deviation",
            "Incorrect torque program or calibration drift",
        ]
    if any(k in t for k in ["dropout", "intermittent", "not responding", "dtc"]):
        return [
            "Connector contact resistance due to crimp quality deviation",
            "Internal sensor damage from handling shock",
            "EMI / shielding deviation affecting signal stability",
        ]
    return [
        "Process parameter drift at supplier",
        "Handling / transport-induced damage",
        "Assembly station parameter mismatch (tooling / calibration)",
    ]




# ---------------- Provider routers (Ollama / Groq / Gemini) ----------------
_GEMINI_EMBED_CACHE: Dict[str, List[float]] = {}
_GROQ_CHAT_CACHE: Dict[str, str] = {}


def _provider_cache_key(model: str, payload: Any) -> str:
    b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = hashlib.sha256(b).hexdigest()
    return f"{model}::{h}"


def _gemini_embed(model: str, text: str, timeout: int = 120) -> np.ndarray:
    """
    model: e.g. 'gemini-embedding-001' (without 'models/' prefix)
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    key = _provider_cache_key(f"gemini:{model}", {"text": text, "dim": GEMINI_OUTPUT_DIM, "taskType": GEMINI_TASK_TYPE})
    if key in _GEMINI_EMBED_CACHE:
        return np.array(_GEMINI_EMBED_CACHE[key], dtype=np.float32)

    url = f"{GEMINI_BASE_URL}/models/{model}:embedContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "taskType": GEMINI_TASK_TYPE,
        "outputDimensionality": GEMINI_OUTPUT_DIM,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json() or {}

    # Typical REST response:
    #   {"embedding": {"values": [...]}}
    # Some SDK/variants:
    #   {"embeddings": [{"values": [...]}]}
    emb = None
    if isinstance(data.get("embedding"), dict):
        emb = data["embedding"].get("values")
    if emb is None and isinstance(data.get("embeddings"), list) and data["embeddings"]:
        emb = (data["embeddings"][0] or {}).get("values")

    if not isinstance(emb, list) or not emb:
        return np.array([], dtype=np.float32)

    _GEMINI_EMBED_CACHE[key] = emb
    return np.array(emb, dtype=np.float32)


def _groq_embed(model: str, text: str, timeout: int = 120) -> np.ndarray:
    """
    model: Groq embedding model ID, e.g. 'text-embedding-3-small'
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    key = _provider_cache_key(f"groq_embed:{model}", {"text": text})
    if key in _GEMINI_EMBED_CACHE:
        return np.array(_GEMINI_EMBED_CACHE[key], dtype=np.float32)

    url = f"{GROQ_BASE_URL.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": text}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json() or {}

    emb = None
    if isinstance(data.get("data"), list) and data["data"]:
        emb = (data["data"][0] or {}).get("embedding")

    if not isinstance(emb, list) or not emb:
        return np.array([], dtype=np.float32)

    _GEMINI_EMBED_CACHE[key] = emb
    return np.array(emb, dtype=np.float32)


def _groq_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 180) -> str:
    """
    model: Groq model ID, e.g. 'llama-3.3-70b-versatile'
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    key = _provider_cache_key(f"groq:{model}", {"messages": messages, "temperature": temperature})
    if key in _GROQ_CHAT_CACHE:
        return _GROQ_CHAT_CACHE[key]

    url = f"{GROQ_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json() or {}
    msg = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    _GROQ_CHAT_CACHE[key] = msg
    return msg


def embed_router(embed_model: str, text: str) -> np.ndarray:
    """
    Supported:
      - 'nomic-embed-text:latest' (Ollama default)
      - 'ollama:nomic-embed-text:latest'
      - 'gemini:gemini-embedding-001'
      - 'groq:text-embedding-3-small'
    """
    if embed_model.startswith("gemini:"):
        return _gemini_embed(embed_model.split(":", 1)[1], text)
    if embed_model.startswith("groq:"):
        return _groq_embed(embed_model.split(":", 1)[1], text)
    if embed_model.startswith("ollama:"):
        return ollama.embed(embed_model.split(":", 1)[1], text)
    return ollama.embed(embed_model, text)


def chat_router(chat_model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """
    Supported:
      - 'llama3:latest' (Ollama default)
      - 'ollama:llama3:latest'
      - 'groq:llama-3.3-70b-versatile'
    """
    if chat_model.startswith("groq:"):
        return _groq_chat(chat_model.split(":", 1)[1], messages, temperature=temperature)
    if chat_model.startswith("ollama:"):
        return ollama.chat(chat_model.split(":", 1)[1], messages, temperature=temperature)
    return ollama.chat(chat_model, messages, temperature=temperature)

# ---------------- Ollama client ----------------
class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._embed_cache: Dict[str, List[float]] = {}
        self._chat_cache: Dict[str, str] = {}

    def _cache_key(self, model: str, payload: Any) -> str:
        b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        h = hashlib.sha256(b).hexdigest()
        return f"{model}::{h}"

    def embed(self, model: str, text: str, timeout: int = 120) -> np.ndarray:
        key = self._cache_key(model, {"text": text})
        if key in self._embed_cache:
            return np.array(self._embed_cache[key], dtype=np.float32)

        url = f"{self.base_url}/api/embeddings"
        r = requests.post(url, json={"model": model, "prompt": text}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            return np.array([], dtype=np.float32)
        self._embed_cache[key] = emb
        return np.array(emb, dtype=np.float32)

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 180) -> str:
        key = self._cache_key(model, {"messages": messages, "temperature": temperature})
        if key in self._chat_cache:
            return self._chat_cache[key]
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        msg = (data.get("message") or {}).get("content") or ""
        self._chat_cache[key] = msg
        return msg


ollama = OllamaClient(OLLAMA_URL)


# ---------------- In-memory indexes ----------------
@dataclass
class PartEntry:
    materialNumber: str
    supplierId: str
    supplierName: str
    drawingNumber: str
    supplierPartNumber: str
    commodityGroup: str
    partDescription: str
    plants: List[str]
    carModels: List[str]
    emb: np.ndarray


@dataclass
class IssueEntry:
    issueld: str
    supplierId: str
    supplierName: str
    materialNumber: str
    PlantId: str
    createdOn: str
    defectDate: str
    title: str
    description: str
    eventType: str
    bi: str
    emb: np.ndarray


STATE: Dict[str, Any] = {
    "plant_names": {},
    "supplier_names": {},
    "parts_by_supplier": {},                # supplierId -> List[PartEntry]
    "parts_usage_by_supplier_material": {}, # "sid||mn" -> raw rows
    "issues_by_supplier": {},               # supplierId -> List[IssueEntry]
    "ingested": False,
}


def _part_text(p: Dict[str, Any]) -> str:
    return (
        f"supplierName: {p.get('supplierName','')}\n"
        f"materialNumber: {p.get('materialNumber','')}\n"
        f"drawingNumber: {p.get('drawingNumber','')}\n"
        f"supplierPartNumber: {p.get('supplierPartNumber','')}\n"
        f"commodityGroup: {p.get('commodityGroup','')}\n"
        f"partDescription: {p.get('partDescription','')}\n"
        f"carModels: {', '.join(sorted(set(p.get('carModels',[]))))}\n"
        f"plants: {', '.join(sorted(set(p.get('plants',[]))))}\n"
    )


def _issue_text(issue: Dict[str, Any]) -> str:
    ev = issue.get("issueEvents") or []
    et = ""
    if isinstance(ev, list) and ev and isinstance(ev[0], dict):
        et = str(ev[0].get("eventType") or "")
    return (
        f"supplierId: {issue.get('supplierId','')}\n"
        f"supplierName: {issue.get('supplierName','')}\n"
        f"materialNumber: {issue.get('materialNumber','')}\n"
        f"plant: {issue.get('PlantId','')}\n"
        f"eventType: {et}\n"
        f"bi: {issue.get('bi','')}\n"
        f"title: {issue.get('title','')}\n"
        f"description: {issue.get('description','')}\n"
    )


def _find_meta(sid: str, mn: str) -> Optional[Dict[str, Any]]:
    key = f"{sid}||{mn}"
    rows = STATE["parts_usage_by_supplier_material"].get(key)
    if not rows:
        return None
    r0 = rows[0]
    return {
        "drawingNumber": str(r0.get("drawingNumber") or ""),
        "supplierPartNumber": str(r0.get("supplierPartNumber") or ""),
        "commodityGroup": str(r0.get("commodityGroup") or ""),
        "partDescription": str(r0.get("partDescription") or ""),
    }


def _ensure_relations(issue: Dict[str, Any], supplier_name: str):
    rel = issue.get("issueRelations")
    if not isinstance(rel, list):
        issue["issueRelations"] = []
        rel = issue["issueRelations"]
    has_lead = any(isinstance(r, dict) and r.get("relationCategoryCode") == "Z01" for r in rel)
    if not has_lead:
        rel.insert(0, {
            "relationAttribute1": supplier_name,
            "relationCategory": "Lead Issue",
            "relationCategoryCode": "Z01",
            "relationMaterialNumber": issue.get("materialNumber"),
            "relationObjectId": issue.get("issueld"),
            "relationPlant": issue.get("PlantId")
        })
    else:
        for r in rel:
            if isinstance(r, dict) and r.get("relationCategoryCode") == "Z01":
                r["relationAttribute1"] = supplier_name


def _candidate_parts(sid: str, query: str, embed_model: str, topn: int) -> List[PartEntry]:
    parts = STATE["parts_by_supplier"].get(sid, [])
    if not parts:
        return []
    qemb = embed_router(embed_model, query)
    scored = [(p, _cosine(qemb, p.emb)) for p in parts]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:topn]]


def _ai_equivalents(chat_model: str, lead_issue: Dict[str, Any], lead_meta: Optional[Dict[str, Any]],
                    candidates: List[PartEntry], threshold: float) -> List[Dict[str, Any]]:
    sid = str(lead_issue.get("supplierId") or "")
    supplier_name = STATE["supplier_names"].get(sid, str(lead_issue.get("supplierName") or ""))

    cand = []
    for c in candidates:
        cand.append({
            "materialNumber": c.materialNumber,
            "drawingNumber": c.drawingNumber,
            "supplierPartNumber": c.supplierPartNumber,
            "commodityGroup": c.commodityGroup,
            "partDescription": c.partDescription,
            "plants": c.plants,
            "carModels": c.carModels
        })

    ev = lead_issue.get("issueEvents") or []
    et = ""
    if isinstance(ev, list) and ev and isinstance(ev[0], dict):
        et = str(ev[0].get("eventType") or "")

    payload = {
        "lead": {
            "supplierId": sid,
            "supplierName": supplier_name,
            "materialNumber": str(lead_issue.get("materialNumber") or ""),
            "leadMetaFromMDBB": lead_meta or {},
            "issueTitle": str(lead_issue.get("title") or ""),
            "issueDescription": str(lead_issue.get("description") or ""),
            "eventType": et
        },
        "candidates": cand,
        "output_schema": {
            "equivalents": [{"materialNumber": "string", "confidence": 0.0, "reason": "string"}]
        },
        "rules": [
            "Select materials that represent the SAME physical part as the lead (within the same supplier).",
            "Use drawingNumber and supplierPartNumber as strongest signals; then commodityGroup and partDescription.",
            "Return STRICT JSON only (no markdown).",
            "Prefer fewer, high-confidence equivalents over many low-confidence matches."
        ]
    }

    messages = [
        {"role": "system", "content": "You are an automotive quality engineer. Output STRICT JSON only."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]
    raw = chat_router(chat_model, messages, temperature=0.1)
    obj = _safe_json_extract(raw)

    out: List[Dict[str, Any]] = []
    if obj and isinstance(obj.get("equivalents"), list):
        for it in obj["equivalents"]:
            try:
                mn = str(it.get("materialNumber") or "")
                conf = float(it.get("confidence") or 0.0)
                reason = str(it.get("reason") or "")
                if mn and conf >= threshold:
                    out.append({"materialNumber": mn, "confidence": conf, "reason": reason})
            except Exception:
                continue

    # Always include lead material as anchor
    lead_mn = str(lead_issue.get("materialNumber") or "")
    if lead_mn and not any(x["materialNumber"] == lead_mn for x in out):
        out.insert(0, {"materialNumber": lead_mn, "confidence": 1.0, "reason": "Lead material number."})

    # Dedup keep max confidence
    best: Dict[str, Dict[str, Any]] = {}
    for x in out:
        mn = x["materialNumber"]
        if mn not in best or float(x["confidence"]) > float(best[mn]["confidence"]):
            best[mn] = x
    return list(best.values())


def _pattern_hits(sid: str, lead: Dict[str, Any], embed_model: str, max_hits: int, sim_thresh: float) -> Tuple[int, List[Tuple[IssueEntry, float]]]:
    issues = STATE["issues_by_supplier"].get(sid, [])
    if not issues:
        return 0, []
    qemb = embed_router(embed_model, _issue_text(lead))
    lead_dt = _parse_issue_dt(str(lead.get("defectDate") or "")) or _parse_issue_dt(str(lead.get("createdOn") or ""))

    matches: List[Tuple[IssueEntry, float]] = []
    for it in issues:
        if lead_dt is not None:
            d = _parse_issue_dt(it.defectDate) or _parse_issue_dt(it.createdOn)
            if d is None or not _within_last_days(d, lead_dt, 365):
                continue
        sc = _cosine(qemb, it.emb)
        if sc >= sim_thresh:
            matches.append((it, sc))

    matches.sort(key=lambda x: x[1], reverse=True)
    return len(matches), matches[:max_hits]


def _trend_and_plant_counts(sid: str, lead: Dict[str, Any]) -> Dict[str, Any]:
    issues = STATE["issues_by_supplier"].get(sid, [])
    if not issues:
        return {"trendPct": 0.0, "countsLastMonthByPlant": []}

    lead_dt = _parse_issue_dt(str(lead.get("defectDate") or "")) or _parse_issue_dt(str(lead.get("createdOn") or ""))
    if lead_dt is None:
        return {"trendPct": 0.0, "countsLastMonthByPlant": []}

    last30_start = lead_dt - np.timedelta64(30, "D")
    prev30_start = lead_dt - np.timedelta64(60, "D")

    last30 = []
    prev30 = []
    for it in issues:
        d = _parse_issue_dt(it.defectDate) or _parse_issue_dt(it.createdOn)
        if d is None:
            continue
        if prev30_start <= d < last30_start:
            prev30.append(it)
        if last30_start <= d <= lead_dt:
            last30.append(it)

    prev_n = len(prev30)
    last_n = len(last30)
    trend = ((last_n - prev_n) / float(max(prev_n, 1))) * 100.0

    # counts per plant in last 30 days
    cnt: Dict[str, int] = {}
    for it in last30:
        cnt[it.PlantId] = cnt.get(it.PlantId, 0) + 1

    out = [{"plantId": k, "count": v} for k, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))]
    return {"trendPct": round(trend, 1), "countsLastMonthByPlant": out}


def _risk_score(bi: int, pattern_count: int) -> Dict[str, Any]:
    # Simple composite score: severity (BI) + recurrence (pattern_count)
    sev = min(1.0, max(0.0, bi / 8.0))
    rec = min(1.0, np.log1p(pattern_count) / np.log1p(50))
    score = int(round(sev * 50 + rec * 50))
    bars = 10
    filled = int(round((score / 100.0) * bars))
    return {"scorePct": score, "bars": bars, "filled": filled}


def _agentic_texts_and_rootcauses(chat_model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single LLM call: produce row texts + a short root-cause list grounded in payload facts.
    """
    messages = [
        {"role": "system", "content": "Output STRICT JSON only. Do not invent issue IDs, plants, or numbers. Use only payload."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]
    raw = chat_router(chat_model, messages, temperature=0.2)
    obj = _safe_json_extract(raw) or {}
    # Provide safe defaults
    out = {
        "row2_error_and_sorting": str(obj.get("row2_error_and_sorting") or ""),
        "row3_plants_info": str(obj.get("row3_plants_info") or ""),
        "row4_error_pattern_intro": str(obj.get("row4_error_pattern_intro") or ""),
        "row5_root_causes_intro": str(obj.get("row5_root_causes_intro") or ""),
        "rootCauses": obj.get("rootCauses") if isinstance(obj.get("rootCauses"), list) else [],
    }
    return out


# ---------------- API ----------------
class IngestRequest(BaseModel):
    parts: List[Dict[str, Any]]
    issues_train: List[Dict[str, Any]]
    embed_model: str = Field(default="nomic-embed-text:latest")


class ProcessRequest(BaseModel):
    leads: List[Dict[str, Any]]
    embed_model: str = Field(default="nomic-embed-text:latest")
    chat_model: str = Field(default="llama3:latest")
    max_part_candidates: int = Field(default=18, ge=8, le=80)
    part_equivalence_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    pattern_top_hits: int = Field(default=5, ge=3, le=10)
    pattern_similarity_threshold: float = Field(default=0.74, ge=0.0, le=1.0)


app = FastAPI(title="Child-Issue AI (Ollama) - React Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "ollama_url": OLLAMA_URL, "ingested": STATE["ingested"], "providers": {"groq_configured": bool(GROQ_API_KEY), "gemini_configured": bool(GEMINI_API_KEY)}}


@app.post("/ingest")
def ingest(req: IngestRequest):
    parts = req.parts
    issues_train = req.issues_train
    embed_model = req.embed_model

    plant_names: Dict[str, str] = {}
    supplier_names: Dict[str, str] = {}
    usage_group: Dict[str, List[Dict[str, Any]]] = {}

    for r in parts:
        pid = str(r.get("plantId") or "")
        pname = str(r.get("plantName") or "")
        if pid and pname:
            plant_names[pid] = pname

        sid = str(r.get("supplierId") or "")
        sname = str(r.get("supplierName") or "")
        if sid and sname:
            supplier_names[sid] = sname

        mn = str(r.get("materialNumber") or "")
        if sid and mn:
            usage_group.setdefault(f"{sid}||{mn}", []).append(r)

    parts_by_supplier: Dict[str, List[PartEntry]] = {}
    parts_usage_by_supplier_material: Dict[str, List[Dict[str, Any]]] = {}

    for key, rows in usage_group.items():
        sid, mn = key.split("||", 1)
        sname = supplier_names.get(sid, str(rows[0].get("supplierName") or ""))
        plants = sorted(set(str(x.get("plantId") or "") for x in rows if x.get("plantId")))
        carModels = sorted(set(str(x.get("carModel") or "") for x in rows if x.get("carModel")))
        drawing = str(rows[0].get("drawingNumber") or "")
        sppn = str(rows[0].get("supplierPartNumber") or "")
        comm = str(rows[0].get("commodityGroup") or "")
        desc = str(rows[0].get("partDescription") or "")

        part_payload = {
            "supplierId": sid,
            "supplierName": sname,
            "materialNumber": mn,
            "drawingNumber": drawing,
            "supplierPartNumber": sppn,
            "commodityGroup": comm,
            "partDescription": desc,
            "plants": plants,
            "carModels": carModels
        }
        emb = embed_router(embed_model, _part_text(part_payload))
        entry = PartEntry(mn, sid, sname, drawing, sppn, comm, desc, plants, carModels, emb)
        parts_by_supplier.setdefault(sid, []).append(entry)
        parts_usage_by_supplier_material[key] = rows

    issues_by_supplier: Dict[str, List[IssueEntry]] = {}
    for iss in issues_train:
        sid = str(iss.get("supplierId") or "")
        if not sid:
            continue
        sname = str(iss.get("supplierName") or supplier_names.get(sid, ""))
        ev = iss.get("issueEvents") or []
        et = ""
        if isinstance(ev, list) and ev and isinstance(ev[0], dict):
            et = str(ev[0].get("eventType") or "")
        emb = embed_router(embed_model, _issue_text(iss))
        issues_by_supplier.setdefault(sid, []).append(IssueEntry(
            issueld=str(iss.get("issueld") or ""),
            supplierId=sid,
            supplierName=sname,
            materialNumber=str(iss.get("materialNumber") or ""),
            PlantId=str(iss.get("PlantId") or ""),
            createdOn=str(iss.get("createdOn") or ""),
            defectDate=str(iss.get("defectDate") or ""),
            title=str(iss.get("title") or ""),
            description=str(iss.get("description") or ""),
            eventType=et,
            bi=str(iss.get("bi") or ""),
            emb=emb
        ))

    STATE["plant_names"] = plant_names
    STATE["supplier_names"] = supplier_names
    STATE["parts_by_supplier"] = parts_by_supplier
    STATE["parts_usage_by_supplier_material"] = parts_usage_by_supplier_material
    STATE["issues_by_supplier"] = issues_by_supplier
    STATE["ingested"] = True

    return {
        "ok": True,
        "parts_rows": len(parts),
        "unique_supplier_materials": sum(len(v) for v in parts_by_supplier.values()),
        "train_issues": len(issues_train),
        "suppliers_indexed": len(parts_by_supplier),
        "embed_model": embed_model
    }


@app.post("/process")
def process(req: ProcessRequest):
    if not STATE["ingested"]:
        return {"ok": False, "error": "Call /ingest first."}

    embed_model = req.embed_model
    chat_model = req.chat_model

    plant_names = STATE["plant_names"]
    supplier_names = STATE["supplier_names"]

    enriched: List[Dict[str, Any]] = []
    for lead in req.leads:
        issue = json.loads(json.dumps(lead))
        sid = str(issue.get("supplierId") or "")
        if not sid:
            enriched.append(issue)
            continue

        supplier_name = supplier_names.get(sid, str(issue.get("supplierName") or ""))
        issue["supplierName"] = supplier_name
        _ensure_relations(issue, supplier_name)

        lead_mn = str(issue.get("materialNumber") or "")
        lead_plant = str(issue.get("PlantId") or "")
        meta = _find_meta(sid, lead_mn)

        # Candidate generation (embedding-based)
        query = _issue_text(issue)
        if meta:
            query += "\nMDBB_META:\n" + json.dumps(meta, ensure_ascii=False)
        candidates = _candidate_parts(sid, query, embed_model, req.max_part_candidates)

        # LLM equivalence selection
        equivalents = _ai_equivalents(chat_model, issue, meta, candidates, req.part_equivalence_threshold)

        # Build child suggestions from where-used of equivalents
        child_map: Dict[str, Dict[str, Any]] = {}
        for e in equivalents:
            mn = e["materialNumber"]
            conf = float(e["confidence"])
            reason = str(e.get("reason") or "")
            key = f"{sid}||{mn}"
            rows = STATE["parts_usage_by_supplier_material"].get(key) or []
            for r in rows:
                pid = str(r.get("plantId") or "")
                if not pid or pid == lead_plant:
                    continue
                prev = child_map.get(pid)
                if prev is None or conf > float(prev["confidence"]):
                    child_map[pid] = {
                        "plantId": pid,
                        "plantName": plant_names.get(pid, ""),
                        "materialNumber": mn,
                        "confidence": conf,
                        "reason": reason
                    }

        existing_child_plants = set(
            r.get("relationPlant") for r in (issue.get("issueRelations") or [])
            if isinstance(r, dict) and r.get("relationCategoryCode") == "Z02" and r.get("relationPlant")
        )

        _, year, num = _parse_issue_id_parts(str(issue.get("issueld") or ""))

        # add missing children
        added_children = []
        for pid in sorted(child_map.keys()):
            if pid in existing_child_plants:
                continue
            rel_mn = child_map[pid]["materialNumber"]
            rel_obj = f"{pid}-{year or '0000'}-QT-{num or '000000'}"
            issue["issueRelations"].append({
                "relationAttribute1": supplier_name,
                "relationCategory": "Child Issue",
                "relationCategoryCode": "Z02",
                "relationMaterialNumber": rel_mn,
                "relationObjectId": rel_obj,
                "relationPlant": pid
            })
            added_children.append(pid)

        # Pattern hits (merged row: "same error pattern")
        pattern_count, pattern_hits = _pattern_hits(
            sid=sid,
            lead=issue,
            embed_model=embed_model,
            max_hits=req.pattern_top_hits,
            sim_thresh=req.pattern_similarity_threshold
        )

        # Root causes from history
        hints = []
        for it, _sc in pattern_hits:
            hints.extend(_extract_rca_hints(it.description))
        hint_counts: Dict[str, int] = {}
        for h in hints:
            hint_counts[h] = hint_counts.get(h, 0) + 1
        top_hints = [{"hint": k, "count": v} for k, v in sorted(hint_counts.items(), key=lambda x: (-x[1], x[0]))[:6]]

        # Risk + trend + plant counts
        bi = int(str(issue.get("bi") or "0").strip() or "0")
        risk = _risk_score(bi=bi, pattern_count=pattern_count)
        trend = _trend_and_plant_counts(sid=sid, lead=issue)

        # Prepare list for UI tables
        pattern_rows = []
        for it, sc in pattern_hits:
            pattern_rows.append({
                "date": _fmt_date(it.defectDate or it.createdOn),
                "plantId": it.PlantId,
                "plantName": plant_names.get(it.PlantId, ""),
                "issueld": it.issueld,
                "sorting": _infer_sorting_action({"bi": it.bi, "title": it.title, "description": it.description, "issueEvents": [{"eventType": it.eventType}]}),
                "similarity": round(float(sc), 3),
                "materialNumber": it.materialNumber
            })

        # Map markers: lead + children (existing vs AI-added)
        markers = []
        markers.append({
            "type": "lead",
            "plantId": lead_plant,
            "materialNumber": lead_mn,
            "title": str(issue.get("title") or ""),
            "supplierName": supplier_name,
            "bi": str(issue.get("bi") or ""),
            "quantity": int(issue.get("quantity") or 1),
            "confidence": None
        })
        # existing children
        for pid in sorted(existing_child_plants):
            markers.append({
                "type": "child_existing",
                "plantId": pid,
                "materialNumber": None,
                "title": str(issue.get("title") or ""),
                "supplierName": supplier_name,
                "bi": str(issue.get("bi") or ""),
                "quantity": int(issue.get("quantity") or 1),
                "confidence": 1.0
            })
        # added children
        for pid in sorted(added_children):
            markers.append({
                "type": "child_added",
                "plantId": pid,
                "materialNumber": child_map[pid]["materialNumber"],
                "title": str(issue.get("title") or ""),
                "supplierName": supplier_name,
                "bi": str(issue.get("bi") or ""),
                "quantity": int(issue.get("quantity") or 1),
                "confidence": float(child_map[pid]["confidence"])
            })

        # Build payload for one LLM call (texts + root cause rephrase)
        ev = issue.get("issueEvents") or []
        et = ""
        if isinstance(ev, list) and ev and isinstance(ev[0], dict):
            et = str(ev[0].get("eventType") or "")

        heuristic_rca = _heuristic_root_causes(str(issue.get("title") or ""), str(issue.get("description") or ""), et)

        llm_payload = {
            "lead": {
                "issueld": issue.get("issueld"),
                "plantId": lead_plant,
                "plantName": plant_names.get(lead_plant, ""),
                "supplierId": sid,
                "supplierName": supplier_name,
                "materialNumber": lead_mn,
                "bi": bi,
                "eventType": et,
                "title": issue.get("title"),
                "description": issue.get("description"),
            },
            "leadMetaFromMDBB": meta or {},
            "childPlants": [{"plantId": k, "plantName": plant_names.get(k, ""), "confidence": child_map.get(k, {}).get("confidence", 1.0)} for k in sorted(set(existing_child_plants) | set(added_children))],
            "errorPattern": {
                "countLast12Months": pattern_count,
                "top": pattern_rows
            },
            "rootCauseSignals": {
                "topHintsFromHistory": top_hints,
                "heuristicSuggestions": heuristic_rca
            },
            "risk": risk,
            "trend": trend,
            "output_schema": {
                "row2_error_and_sorting": "string",
                "row3_plants_info": "string",
                "row4_error_pattern_intro": "string",
                "row5_root_causes_intro": "string",
                "rootCauses": [{"cause": "string", "confidence": 0.0, "support": "string"}]
            },
            "rules": [
                "Return STRICT JSON only.",
                "Do NOT invent issue IDs or plants outside of provided lists.",
                "Use only provided numbers/counts. If data is missing, write neutral text.",
                "Root causes must be plausible and grounded in provided hints/suggestions."
            ]
        }

        texts = _agentic_texts_and_rootcauses(chat_model, llm_payload)

        # Fallbacks if LLM failed
        if not texts["row2_error_and_sorting"]:
            texts["row2_error_and_sorting"] = f"In {plant_names.get(lead_plant, lead_plant)} ({lead_plant}) a part was identified as faulty. Containment and sorting should be evaluated based on BI={bi} and the observed pattern."
        if not texts["row3_plants_info"]:
            plants_list = ", ".join([f"{plant_names.get(p,p)} ({p})" for p in sorted(set(existing_child_plants)|set(added_children))]) or "no additional plants"
            texts["row3_plants_info"] = f"Current recall notices / related actions are available for: {plants_list}."
        if not texts["row4_error_pattern_intro"]:
            texts["row4_error_pattern_intro"] = "The table lists the most similar historical issues within the last 12 months for the same supplier (same error pattern)."
        if not texts["row5_root_causes_intro"]:
            texts["row5_root_causes_intro"] = "Potential root causes are derived from historical patterns and engineering heuristics."

        # Root causes fallback
        root_causes = []
        if isinstance(texts.get("rootCauses"), list) and texts["rootCauses"]:
            for rc in texts["rootCauses"][:5]:
                try:
                    root_causes.append({
                        "cause": str(rc.get("cause") or ""),
                        "confidence": float(rc.get("confidence") or 0.0),
                        "support": str(rc.get("support") or "")
                    })
                except Exception:
                    continue
        if not root_causes:
            # make from hints
            use = [h["hint"] for h in top_hints[:3]] or heuristic_rca[:3]
            for idx, c in enumerate(use):
                root_causes.append({"cause": c, "confidence": round(0.75 - idx*0.1, 2), "support": "Derived from historical hints / heuristics."})

        enrichment_obj = {
            "childSuggestions": [child_map[k] for k in sorted(child_map.keys())],
            "row2_error_and_sorting": texts["row2_error_and_sorting"],
            "row3_plants_info": texts["row3_plants_info"],
            "errorPattern": {
                "countLast12Months": pattern_count,
                "intro": texts["row4_error_pattern_intro"],
                "top": pattern_rows
            },
            "rootCauses": {
                "intro": texts["row5_root_causes_intro"],
                "items": root_causes
            },
            "risk": risk,
            "trend": trend,
            "map": {
                "markers": markers
            }
        }

        issue["description"] = (
            (issue.get("description") or "")
            + "\n\n---\nAI_ENRICHMENT_JSON_BEGIN\n"
            + json.dumps(enrichment_obj, ensure_ascii=False)
            + "\nAI_ENRICHMENT_JSON_END\n"
        )

        enriched.append(issue)

    return {"ok": True, "enriched": enriched}
