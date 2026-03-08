import React, { useEffect, useMemo, useState } from 'react'
import { MapContainer, TileLayer, Marker, Tooltip } from 'react-leaflet'
import L from 'leaflet'

const PLANT_COORDS = {
  B210: { name: 'Schanghai', lat: 31.2304, lon: 121.4737 },
  FO33: { name: 'Munchen', lat: 48.1351, lon: 11.5820 },
  W520: { name: 'Würzburg', lat: 49.7913, lon: 9.9534 },
  V490: { name: 'Berlin', lat: 52.52, lon: 13.405 },
  JK05: { name: 'Ulm', lat: 48.4011, lon: 9.9876 },
  LO10: { name: 'London', lat: 51.5072, lon: -0.1276 }
}

function fmtPlant(id){
  const p = PLANT_COORDS[id]
  return p ? `${p.name} (${id})` : (id || '—')
}

function safeParse(txt){ try { return JSON.parse(txt) } catch { return null } }

function extractEnrichment(desc){
  if(!desc) return null
  const m = String(desc).match(/AI_ENRICHMENT_JSON_BEGIN\s*([\s\S]*?)\s*AI_ENRICHMENT_JSON_END/)
  if(!m) return null
  return safeParse(m[1])
}

function replaceEnrichment(desc, obj){
  const begin = "AI_ENRICHMENT_JSON_BEGIN"
  const end = "AI_ENRICHMENT_JSON_END"
  const s = String(desc || "")
  const m = s.match(/AI_ENRICHMENT_JSON_BEGIN\s*[\s\S]*?\s*AI_ENRICHMENT_JSON_END/)
  const payload = `${begin}\n${JSON.stringify(obj)}\n${end}`
  if(m){
    return s.replace(m[0], payload)
  }
  return s + `\n\n---\n${payload}`
}

function readFile(file){
  return new Promise((resolve,reject)=>{
    const r = new FileReader()
    r.onload=()=>resolve(String(r.result||""))
    r.onerror=()=>reject(r.error)
    r.readAsText(file)
  })
}

function dotIcon(color, size=16, glow=true){
  const shadow = glow ? `box-shadow: 0 0 0 2px rgba(0,0,0,.35), 0 0 14px ${color};` : `box-shadow: 0 0 0 2px rgba(0,0,0,.35);`
  const html = `<div style="width:${size}px;height:${size}px;border-radius:999px;background:${color};border:2px solid rgba(255,255,255,.9);${shadow}"></div>`
  return L.divIcon({ className: '', html, iconSize: [size+6,size+6], iconAnchor: [(size+6)/2,(size+6)/2] })
}
function symbolIcon(symbol, color="#ff3b3b", size=18){
  const html = `<div style="color:${color};font-weight:900;font-size:${size}px;line-height:${size}px;text-shadow:0 0 10px rgba(255,0,0,.55);">${symbol}</div>`
  return L.divIcon({ className: '', html, iconSize: [size+6,size+6], iconAnchor: [(size+6)/2,(size+6)/2] })
}

const ICONS = {
  lead: dotIcon('#3b82f6', 14, true),               // blue
  child_existing: dotIcon('#ffe27a', 14, true),     // brighter yellow
  child_added: symbolIcon('⚠', '#ff3b3b', 15),      // red warning
  synergy: dotIcon('#5b3cc4', 14, true)             // dark purple
}


function TrashIcon(){
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M9 3h6l1 2h4v2H4V5h4l1-2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M6 9v11a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M10 12v7M14 12v7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  )
}

export default function App(){
  const [baseUrl, setBaseUrl] = useState("http://localhost:8000")
  const [embedModel, setEmbedModel] = useState("nomic-embed-text:latest")
  const [chatModel, setChatModel] = useState("llama3:latest")

  const [showSidebar, setShowSidebar] = useState(true)

  const [parts, setParts] = useState(null)
  const [train, setTrain] = useState(null)
  const [leads, setLeads] = useState(null)

  const [enriched, setEnriched] = useState(null) // working copy (editable)
  const [selectedIdx, setSelectedIdx] = useState(null)
  const [backendStatus, setBackendStatus] = useState("Backend: —")
  const [stats, setStats] = useState("Parts loaded: 0 | Train issues loaded: 0")
  const [linkedJson, setLinkedJson] = useState(null)

  useEffect(()=>{
    setStats(`Parts loaded: ${parts?parts.length:0} | Train issues loaded: ${train?train.length:0}`)
  }, [parts, train])

  async function pingBackend(){
    try{
      const r = await fetch(baseUrl.replace(/\/$/,"") + "/health")
      const j = await r.json()
      setBackendStatus(`Backend: OK (ingested: ${j.ingested})`)
    }catch{
      setBackendStatus("Backend: NOT REACHABLE — start backend/server.py")
    }
  }

  useEffect(()=>{ pingBackend() }, [baseUrl])

  async function postJSON(path, payload){
    const url = baseUrl.replace(/\/$/,"") + path
    const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)})
    if(!r.ok){
      const t = await r.text()
      throw new Error(`HTTP ${r.status}: ${t}`)
    }
    return await r.json()
  }

  async function ingest(){
    try{
      setBackendStatus("Backend: ingesting + embedding… (Ollama)")
      const j = await postJSON("/ingest", {parts, issues_train: train, embed_model: embedModel})
      setBackendStatus(`Backend: Indexed — suppliers: ${j.suppliers_indexed}, unique materials: ${j.unique_supplier_materials}, train issues: ${j.train_issues}`)
    }catch(e){
      setBackendStatus(`Backend: ERROR — ${e.message}`)
    }
  }

  async function processLeads(){
    try{
      setBackendStatus("Backend: processing leads… (AI search + LLM enrichment)")
      const j = await postJSON("/process", {
        leads,
        embed_model: embedModel,
        chat_model: chatModel,
        max_part_candidates: 18,
        part_equivalence_threshold: 0.60,
        pattern_top_hits: 5,
        pattern_similarity_threshold: 0.74
      })
      if(!j.ok) throw new Error(j.error || "Unknown error")
      setEnriched(structuredClone(j.enriched))
      setSelectedIdx(0)
      setLinkedJson(null)
      setBackendStatus(`Backend: Done — enriched issues: ${j.enriched.length}`)
    }catch(e){
      setBackendStatus(`Backend: ERROR — ${e.message}`)
    }
  }

  const selected = useMemo(()=>{
    if(selectedIdx == null || !enriched) return null
    return enriched[selectedIdx]
  }, [selectedIdx, enriched])

  const selectedOrig = useMemo(()=>{
    if(selectedIdx == null || !leads) return null
    return leads[selectedIdx]
  }, [selectedIdx, leads])

  const enrichment = useMemo(()=>{
    if(!selected) return null
    return extractEnrichment(selected.description) || null
  }, [selected])

  // children rows from relations + delete
  function removeChild(plantId){
    if(selectedIdx == null) return
    setEnriched(prev=>{
      const next = structuredClone(prev)
      const issue = next[selectedIdx]
      issue.issueRelations = (issue.issueRelations || []).filter(r => !(r && r.relationCategoryCode==="Z02" && r.relationPlant===plantId))
      // also remove from enrichment childSuggestions/map markers if present
      const enr = extractEnrichment(issue.description)
      if(enr){
        if(Array.isArray(enr.childSuggestions)){
          enr.childSuggestions = enr.childSuggestions.filter(x => x && x.plantId !== plantId)
        }
        if(enr.map && Array.isArray(enr.map.markers)){
          enr.map.markers = enr.map.markers.filter(m => m && m.plantId !== plantId)
        }
        issue.description = replaceEnrichment(issue.description, enr)
      }
      return next
    })
  }

  // clickable issue JSON with MDBB
  function mdbbRows(supplierId, materialNumber, plantId=null){
    if(!parts) return []
    return parts.filter(r =>
      r.supplierId === supplierId &&
      r.materialNumber === materialNumber &&
      (plantId ? r.plantId === plantId : true)
    )
  }
  function findIssueById(issueld){
    if(leads){
      const x = leads.find(i => i && i.issueld === issueld)
      if(x) return x
    }
    if(enriched){
      const x = enriched.find(i => i && i.issueld === issueld)
      if(x) return x
    }
    if(train){
      const x = train.find(i => i && i.issueld === issueld)
      if(x) return x
    }
    return null
  }

  function openLinked(issueld, supplierId, supplierName, materialNumber, plantId){
    const found = findIssueById(issueld)
    const base = found ? structuredClone(found) : {issueld, supplierId, supplierName, materialNumber, PlantId: plantId, issueRelations: []}
    base.mdbbWhereUsed = mdbbRows(supplierId, materialNumber)
    base.mdbbInPlant = plantId ? mdbbRows(supplierId, materialNumber, plantId) : []
    setLinkedJson(base)
  }

  // derived list of children for UI
  const childrenList = useMemo(()=>{
    if(!selected) return []
    const supplierId = selected.supplierId
    const supplierName = selected.supplierName || supplierId

    const confByPlant = new Map()
    const sugg = enrichment?.childSuggestions
    if(Array.isArray(sugg)){
      for(const cs of sugg){
        if(cs && cs.plantId){
          const v = Number(cs.confidence)
          if(Number.isFinite(v)) confByPlant.set(cs.plantId, v)
        }
      }
    }

    return (selected.issueRelations || [])
      .filter(r => r && r.relationCategoryCode==="Z02")
      .map(r => {
        const pid = r.relationPlant
        const conf = confByPlant.has(pid) ? confByPlant.get(pid) : 1.0
        return ({
          issueld: r.relationObjectId,
          plantId: pid,
          materialNumber: r.relationMaterialNumber,
          supplierId,
          supplierName,
          confidence: Number.isFinite(conf) ? conf : 1.0
        })
      })
  }, [selected, enrichment])

  const missingAddedPlants = useMemo(()=>{
    if(!selected || !selectedOrig) return new Set()
    const o = new Set((selectedOrig.issueRelations || []).filter(r=>r && r.relationCategoryCode==="Z02").map(r=>r.relationPlant))
    const n = new Set((selected.issueRelations || []).filter(r=>r && r.relationCategoryCode==="Z02").map(r=>r.relationPlant))
    const added = new Set()
    for(const p of n){
      if(!o.has(p)) added.add(p)
    }
    return added
  }, [selected, selectedOrig])

  // Map markers
  const mapMarkers = useMemo(()=>{
    if(!selected || !enrichment || !enrichment.map || !Array.isArray(enrichment.map.markers)) return []

    const baseMarkers = enrichment.map.markers.map(m => {
      if(!m || !m.plantId) return null
      let type = m.type
      if(type === "child_existing" && missingAddedPlants.has(m.plantId)) type = "child_added"
      return {...m, type}
    }).filter(Boolean)

    const synergyMarkers = Array.isArray(enrichment?.errorPattern?.top)
      ? enrichment.errorPattern.top
          .filter(r => r && r.plantId && r.issueld)
          .map(r => ({
            type: "synergy",
            plantId: r.plantId,
            issueld: r.issueld,
            materialNumber: r.materialNumber,
            title: selected?.title || "Same error pattern",
            supplierName: selected?.supplierName || selected?.supplierId || "—",
            bi: selected?.bi || "—",
            quantity: 1,
            confidence: null,
            similarity: r.similarity,
            sorting: r.sorting
          }))
      : []

    const seen = new Set()
    return [...baseMarkers, ...synergyMarkers].filter(m => {
      const key = `${m.type}|${m.plantId}|${m.issueld || m.materialNumber || ""}`
      if(seen.has(key)) return false
      seen.add(key)
      return true
    })
  }, [selected, enrichment, missingAddedPlants])

  const mapCenter = useMemo(()=>{
    // center on Europe-ish by default, or on lead
    const lead = mapMarkers.find(m => m.type==="lead")
    if(lead && PLANT_COORDS[lead.plantId]) return [PLANT_COORDS[lead.plantId].lat, PLANT_COORDS[lead.plantId].lon]
    return [49.0, 10.0]
  }, [mapMarkers])

  function downloadEnriched(){
    const blob = new Blob([JSON.stringify(enriched, null, 2)], {type:"application/json"})
    const a=document.createElement("a")
    a.href=URL.createObjectURL(blob)
    a.download="lead_issues_enriched.json"
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <>
      <div className="header">
        <h1>Child-Issue AI (Ollama)</h1>
        <span className="pill">React + SCSS + Leaflet</span>
        <span className="pill">Same supplier only</span>
        <button
          className="btn secondary"
          style={{padding:"6px 10px"}}
          onClick={()=>setShowSidebar(s=>!s)}
          title="Show/hide settings panel"
        >
          {showSidebar ? "Hide settings" : "Show settings"}
        </button>
      </div>

      <div className={`container ${showSidebar ? "" : "oneCol"}`}>
        {/* Left */}
        {showSidebar && (
          <section className="card">
          <h2>0) Backend settings</h2>

          <label className="label">Backend base URL</label>
          <input className="input" value={baseUrl} onChange={e=>setBaseUrl(e.target.value)} />

          <label className="label">Ollama embedding model</label>
          <input className="input" value={embedModel} onChange={e=>setEmbedModel(e.target.value)} />

          <label className="label">Ollama chat model</label>
          <input className="input" value={chatModel} onChange={e=>setChatModel(e.target.value)} />

          <div className="small">
            Example pulls:<br/>
            <span className="mono">ollama pull nomic-embed-text:latest</span><br/>
            <span className="mono">ollama pull llama3:latest</span>
          </div>

          <div className="hr" />

          <h2>1) Load datasets</h2>

          <label className="label">MDBB parts usage (JSON array)</label>
          <input className="input" type="file" accept=".json,application/json"
            onChange={async e=>{
              const f=e.target.files?.[0]; if(!f) return
              const obj = safeParse(await readFile(f))
              if(!Array.isArray(obj)) return alert("Parts must be a JSON array")
              setParts(obj); setEnriched(null); setSelectedIdx(null)
            }} />

          <label className="label">Issues knowledge base (TRAIN) (JSON array)</label>
          <input className="input" type="file" accept=".json,application/json"
            onChange={async e=>{
              const f=e.target.files?.[0]; if(!f) return
              const obj = safeParse(await readFile(f))
              if(!Array.isArray(obj)) return alert("Train issues must be a JSON array")
              setTrain(obj); setEnriched(null); setSelectedIdx(null)
            }} />

          <div className="row" style={{marginTop:10}}>
            <button className="btn secondary" disabled={!(parts && train)} onClick={ingest}>Ingest + build AI indexes</button>
            <button className="btn secondary" onClick={()=>{
              setParts(null); setTrain(null); setLeads(null); setEnriched(null); setSelectedIdx(null); setLinkedJson(null)
            }}>Reset</button>
          </div>

          <div className="small">{stats}</div>
          <div className="small">{backendStatus}</div>

          <div className="hr" />

          <h2>2) Upload lead issues</h2>
          <label className="label">Lead issues (JSON array)</label>
          <input className="input" type="file" accept=".json,application/json"
            onChange={async e=>{
              const f=e.target.files?.[0]; if(!f) return
              const obj = safeParse(await readFile(f))
              if(!Array.isArray(obj)) return alert("Lead issues must be a JSON array")
              setLeads(obj); setEnriched(null); setSelectedIdx(null); setLinkedJson(null)
            }} />

          <div className="row" style={{marginTop:10}}>
            <button className="btn" disabled={!(parts && train && leads)} onClick={processLeads}>Find children + enrich (AI)</button>
            <button className="btn secondary" disabled={!enriched} onClick={downloadEnriched}>Download enriched JSON</button>
          </div>

          <div className="small">
            Tip: open an issue → remove a child via trash icon → download updated JSON.
          </div>
        </section>
        )}

        {/* Right */}
        <section className="card">
          <h2>Issues</h2>
          <div className="small">Click a row to view details.</div>

          <div className="tableScroll">
            <table className="table">
              <thead>
                <tr>
                  <th style={{width:"30%"}}>issueld</th>
                  <th style={{width:"20%"}}>Plant</th>
                  <th style={{width:"20%"}}>Material</th>
                  <th style={{width:"20%"}}>Supplier</th>
                  <th style={{width:"10%"}}>Childs</th>
                </tr>
              </thead>
              <tbody>
                {(enriched || []).map((iss, idx)=>{
                  const s = iss.supplierName || iss.supplierId || "—"
                  const childs = (iss.issueRelations||[]).filter(r=>r && r.relationCategoryCode==="Z02").length
                  const active = idx === selectedIdx
                  return (
                    <tr key={idx} onClick={()=>{ setSelectedIdx(idx); setLinkedJson(null) }}
                        style={active ? {outline:`2px solid rgba(106,169,255,.4)`} : undefined}>
                      <td className="mono">{iss.issueld || "—"}</td>
                      <td>{fmtPlant(iss.PlantId)}</td>
                      <td className="mono">{iss.materialNumber || "—"}</td>
                      <td>{s}</td>
                      <td>{childs}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="hr" />

          <h2>Issue details</h2>
          <div className="tableScroll">
            <table className="table">
              <tbody>
                <tr>
                  <th style={{width:220}}>1) Children</th>
                  <td>
                    {childrenList.length === 0 ? <span className="small">No child relations.</span> : (
                      <div className="tableScroll">
                        <table className="table">
                          <thead>
                            <tr>
                              <th style={{width:"28%"}}>issueld</th>
                              <th style={{width:"20%"}}>Plant</th>
                              <th style={{width:"20%"}}>Material</th>
                              <th style={{width:"18%"}}>Supplier</th>
                              <th style={{width:"12%"}}>Conf.</th>
                              <th style={{width:"10%"}}></th>
                            </tr>
                          </thead>
                          <tbody>
                            {childrenList.map((c)=>(
                              <tr key={c.plantId}>
                                <td className="mono">
                                  <a href="#" onClick={(e)=>{e.preventDefault(); openLinked(c.issueld, c.supplierId, c.supplierName, c.materialNumber, c.plantId)}}>
                                    {c.issueld}
                                  </a>
                                </td>
                                <td>{fmtPlant(c.plantId)}</td>
                                <td className="mono">{c.materialNumber}</td>
                                <td>{c.supplierName}</td>
                                <td className="mono">{Number.isFinite(c.confidence) ? c.confidence.toFixed(2) : "1.00"}</td>
                                <td style={{textAlign:"right"}}>
                                  <button className="iconBtn" title="Remove child" onClick={()=>removeChild(c.plantId)}><TrashIcon/></button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </td>
                </tr>

                <tr>
                  <th>2) Error description & sorting information</th>
                  <td>{enrichment?.row2_error_and_sorting || <span className="small">—</span>}</td>
                </tr>

                <tr>
                  <th>3) Info for other plants</th>
                  <td>{enrichment?.row3_plants_info || <span className="small">—</span>}</td>
                </tr>

                <tr>
                  <th>4) Same error pattern</th>
                  <td>
                    <div className="small">
                      {enrichment?.errorPattern?.countLast12Months != null
                        ? `Last 12 months: ${enrichment.errorPattern.countLast12Months} similar cases`
                        : "—"}
                    </div>
                    <div style={{margin:"6px 0 10px 0"}}>{enrichment?.errorPattern?.intro || ""}</div>

                    {Array.isArray(enrichment?.errorPattern?.top) && enrichment.errorPattern.top.length ? (
                      <div className="tableScroll">
                        <table className="table">
                          <thead>
                            <tr>
                              <th style={{width:"16%"}}>Date</th>
                              <th style={{width:"20%"}}>Plant</th>
                              <th style={{width:"26%"}}>issueld</th>
                              <th style={{width:"18%"}}>Sorting</th>
                              <th style={{width:"10%"}}>Sim.</th>
                              <th style={{width:"10%"}}>Mat.</th>
                            </tr>
                          </thead>
                          <tbody>
                            {enrichment.errorPattern.top.map((r)=>(
                              <tr key={r.issueld}>
                                <td>{r.date || "—"}</td>
                                <td>{fmtPlant(r.plantId)}</td>
                                <td className="mono">
                                  <a href="#" onClick={(e)=>{e.preventDefault(); openLinked(r.issueld, selected?.supplierId, selected?.supplierName, r.materialNumber, r.plantId)}}>
                                    {r.issueld}
                                  </a>
                                </td>
                                <td>{r.sorting || "—"}</td>
                                <td>{r.similarity ?? "—"}</td>
                                <td className="mono">{r.materialNumber || "—"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="small">No similar cases found.</div>
                    )}
                  </td>
                </tr>

                <tr>
                  <th>5) Possible root causes</th>
                  <td>
                    <div style={{margin:"0 0 10px 0"}}>{enrichment?.rootCauses?.intro || ""}</div>
                    {Array.isArray(enrichment?.rootCauses?.items) && enrichment.rootCauses.items.length ? (
                      <div className="tableScroll">
                        <table className="table">
                          <thead>
                            <tr>
                              <th style={{width:"60%"}}>Cause</th>
                              <th style={{width:"12%"}}>Conf.</th>
                              <th style={{width:"28%"}}>Support</th>
                            </tr>
                          </thead>
                          <tbody>
                            {enrichment.rootCauses.items.map((rc, idx)=>(
                              <tr key={idx}>
                                <td>{rc.cause}</td>
                                <td>{rc.confidence ?? "—"}</td>
                                <td>{rc.support || "—"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="small">—</div>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Map + widgets */}
          <div className="hr" />
          <h2>Alerted plants map</h2>

          <div className="mapWrap">
            <MapContainer center={mapCenter} zoom={4} scrollWheelZoom={true}>
              <TileLayer
                attribution='&copy; OpenStreetMap contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              {mapMarkers.map((m, idx)=>{
                const p = PLANT_COORDS[m.plantId]
                if(!p) return null
                const icon = ICONS[m.type] || ICONS.child_existing
                const pos = m.type === "synergy" ? [p.lat + 0.12, p.lon + 0.12] : [p.lat, p.lon]
                return (
                  <Marker key={idx} position={pos} icon={icon}>
                    <Tooltip direction="top" opacity={1} className="tooltipBox">
                      <div><b>{fmtPlant(m.plantId)}</b></div>
                      <div>Type: {m.type}</div>
                      {m.issueld && <div>Issue: <span className="mono">{m.issueld}</span></div>}
                      <div>Material: <span className="mono">{m.materialNumber || selected?.materialNumber || "—"}</span></div>
                      <div>Supplier: {m.supplierName}</div>
                      <div>BI: {m.bi}</div>
                      {m.type !== "lead" && m.type !== "synergy" && <div>Child confidence: {m.confidence ?? "—"}</div>}
                      {m.type === "synergy" && <div>Pattern similarity: {m.similarity ?? "—"}</div>}
                      {m.type === "synergy" && <div>Sorting: {m.sorting || "—"}</div>}
                      <div style={{marginTop:6}}><i>{m.title}</i></div>
                    </Tooltip>
                  </Marker>
                )
              })}
            </MapContainer>
          </div>

          <div className="legend">
            <div className="item"><span className="dot" style={{background:"#3b82f6"}}></span> Lead</div>
            <div className="item"><span className="dot" style={{background:"#ffe27a"}}></span> Child (already in issue)</div>
            <div className="item"><span className="dot" style={{background:"#ff3b3b"}}></span> AI-added child (!)</div>
            <div className="item"><span className="dot" style={{background:"#5b3cc4"}}></span> Same error pattern</div>
          </div>

          <div className="hr" />
          <h2>Risk & trends</h2>

          <div className="riskRow">
            <div>
              <div className="small">Risk score</div>
              <div style={{fontWeight:900, fontSize:16}}>
                {enrichment?.risk?.scorePct != null ? `${enrichment.risk.scorePct}%` : "—"}
              </div>
            </div>

            <div>
              <div className="small">Risk bars</div>
              <div className="riskBars">
                {Array.from({length: enrichment?.risk?.bars || 10}).map((_, i)=>{
                  const filled = enrichment?.risk?.filled ?? 0
                  const pct = (i+1) / ((enrichment?.risk?.bars)||10)
                  // gradient green->red
                  const r = Math.round(40 + pct*180)
                  const g = Math.round(180 - pct*120)
                  const color = i < filled ? `rgb(${r},${g},80)` : "#102032"
                  return <div key={i} className="bar" style={{background: color}} />
                })}
              </div>
            </div>

            <div>
              <div className="small">Trend (last 30d vs prev 30d)</div>
              <div style={{fontWeight:900, fontSize:16, color: (() => { const raw = enrichment?.trend?.trendPct; const t = (raw == null) ? 0 : ((Math.abs(raw) < 0.05 || raw <= -99.9) ? 0 : raw); return t > 0 ? "var(--danger)" : "var(--ok)"; })()}}>
                {(() => {
                  const raw = enrichment?.trend?.trendPct
                  if(raw == null) return "—"
                  const t = (Math.abs(raw) < 0.05 || raw <= -99.9) ? 0 : raw
                  const arrow = t > 0 ? "↑" : (t < 0 ? "↓" : "")
                  return `${t}%${arrow ? " " + arrow : ""}`
                })()}
              </div>
            </div>
          </div>

          <div style={{marginTop:10}}>
            <div className="small">Issues per plant (last 30 days, training KB)</div>
            <div style={{display:"grid", gap:6, marginTop:6}}>
              {(() => {
                const rows = enrichment?.trend?.countsLastMonthByPlant || []
                const max = Math.max(1, ...rows.map(r=>r.count))
                const all = Object.keys(PLANT_COORDS).map(pid => {
                  const hit = rows.find(r=>r.plantId===pid)
                  return {plantId: pid, count: hit ? hit.count : 0}
                })
                return all.map(r=>(
                  <div key={r.plantId} style={{display:"grid", gridTemplateColumns:"160px 1fr 40px", gap:8, alignItems:"center"}}>
                    <div className="mono">{fmtPlant(r.plantId)}</div>
                    <div style={{height:18, border:"1px solid var(--line)", borderRadius:6, overflow:"hidden", background:"#0c121c"}}>
                      <div style={{height:"100%", width:`${(r.count/max)*100}%`, background:"rgba(106,169,255,.55)"}} />
                    </div>
                    <div style={{textAlign:"right"}}>{r.count}</div>
                  </div>
                ))
              })()}
            </div>
          </div>

          {/* JSON panes */}
          <div className="hr" />
          <div className="split">
            <div>
              <h2>Selected issue (original)</h2>
              <pre>{selectedOrig ? JSON.stringify(selectedOrig, null, 2) : "—"}</pre>
            </div>
            <div>
              <h2>Selected issue (enriched)</h2>
              <pre>{selected ? JSON.stringify(selected, null, 2) : "—"}</pre>
            </div>
          </div>

          <div className="hr" />
          <h2>Linked issue JSON (with MDBB info)</h2>
          <pre>{linkedJson ? JSON.stringify(linkedJson, null, 2) : "—"}</pre>
        </section>
      </div>
    </>
  )
}
