import { useEffect, useMemo, useRef, useState } from "react";
import { fetchDemos, runDemo } from "./api";

function defaultsFor(demo) {
  const values = {};
  for (const field of demo.fields) values[field.name] = field.default;
  return values;
}

function groupFields(fields) {
  const groups = [];
  for (const field of fields) {
    const name = field.group || "Parameters";
    let group = groups.find((item) => item.name === name);
    if (!group) {
      group = { name, fields: [] };
      groups.push(group);
    }
    group.fields.push(field);
  }
  return groups;
}

function formatElapsed(ms) {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes === 0) return `${seconds}s`;
  return `${minutes}m ${String(rest).padStart(2, "0")}s`;
}

function applySuggestion(field, current, suggestion) {
  if (field.type !== "symbols") return suggestion;
  const parts = String(current || "")
    .split(",")
    .map((part) => part.trim().toUpperCase())
    .filter(Boolean);
  const next = parts.includes(suggestion) ? parts.filter((part) => part !== suggestion) : [...parts, suggestion];
  return next.join(", ");
}

function createTab(demo, values, number) {
  return {
    id: `${demo.id}-${number}-${Date.now()}`,
    name: `Run ${number}`,
    values: { ...values },
    status: "idle",
    error: "",
    data: null,
  };
}

function createSession(demo) {
  const tab = createTab(demo, defaultsFor(demo), 1);
  return { activeTabId: tab.id, nextNumber: 2, compare: false, tabs: [tab] };
}

function metricRows(tabs) {
  const labels = [];
  for (const tab of tabs) {
    for (const metric of tab.data?.metrics || []) {
      if (!labels.includes(metric.label)) labels.push(metric.label);
    }
  }
  return labels.map((label) => ({
    label,
    values: tabs.map((tab) => tab.data?.metrics?.find((metric) => metric.label === label)?.value || "—"),
  }));
}

function FieldControl({ field, value, onChange, inputId }) {
  const id = inputId;
  if (field.type === "checkbox") {
    return (
      <label className="check" htmlFor={id}>
        <input id={id} type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
        <span>
          <strong>{field.label}</strong>
          {field.help ? <small>{field.help}</small> : null}
        </span>
      </label>
    );
  }

  const suggestions = field.suggestions || [];
  const selected = new Set(
    field.type === "symbols"
      ? String(value || "")
          .split(",")
          .map((part) => part.trim().toUpperCase())
          .filter(Boolean)
      : [String(value || "").toUpperCase()]
  );

  const wide = field.type === "symbols" || suggestions.length > 0;
  return (
    <div className={wide ? "field field-wide" : "field"}>
      <label htmlFor={id}>{field.label}</label>
      {field.type === "select" ? (
        <select id={id} value={value} onChange={(event) => onChange(event.target.value)}>
          {field.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type={field.type === "number" ? "number" : field.type === "date" ? "date" : "text"}
          value={value}
          min={field.min}
          max={field.max}
          step={field.step}
          onChange={(event) => {
            const raw = event.target.value;
            if (field.type !== "number") {
              onChange(raw);
              return;
            }
            if (raw === "") {
              onChange("");
              return;
            }
            onChange(field.kind === "int" ? Number.parseInt(raw, 10) : Number.parseFloat(raw));
          }}
        />
      )}
      {field.help ? <small>{field.help}</small> : null}
      {suggestions.length > 0 ? (
        <div className="suggestions">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className={selected.has(suggestion) ? "chip on" : "chip"}
              onClick={() => onChange(applySuggestion(field, value, suggestion))}
            >
              {suggestion}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function Results({ demo, result }) {
  if (!result) return null;
  return (
    <div className="results">
      {result.metrics?.length > 0 ? (
        <section>
          <h2>Measures</h2>
          {result.metrics.some((metric) => metric.help) ? (
            <p className="metric-hint">Hover a measure to see what it is, how it is calculated, and why it matters.</p>
          ) : null}
          <div className="metrics">
            {result.metrics.map((metric) => (
              <article key={metric.label} className={metric.help ? "metric has-help" : "metric"} tabIndex={metric.help ? 0 : undefined}>
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
                {metric.help ? (
                  <div className="metric-help" role="tooltip">
                    <p><b>What it is.</b> {metric.help.what}</p>
                    <p><b>How it is calculated.</b> {metric.help.how}</p>
                    <p><b>Why it matters.</b> {metric.help.why}</p>
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {result.sample?.rows?.length > 0 ? (
        <section className="table-block">
          <h2>{result.sample.title}</h2>
          {result.sample.caption ? <p className="metric-hint">{result.sample.caption}</p> : null}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {result.sample.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.sample.rows.map((row, rowIndex) => (
                  <tr key={`sample-${rowIndex}`}>
                    {row.map((cell, cellIndex) => (
                      <td key={`${result.sample.columns[cellIndex]}-${cellIndex}`}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {result.notes?.length > 0 ? (
        <ul className="notes">
          {result.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}

      {result.plots?.length > 0 ? (
        <section className="plots">
          {result.plots.map((plot, index) => (
            <figure key={plot.title} className="plot">
              <figcaption>
                <span>{plot.title}</span>
                <a href={`data:image/png;base64,${plot.image}`} download={`${demo.id}-${index + 1}.png`}>
                  Download
                </a>
              </figcaption>
              <img src={`data:image/png;base64,${plot.image}`} alt={plot.title} />
            </figure>
          ))}
        </section>
      ) : null}

      {result.tables?.map((table) => (
        <section key={table.title} className="table-block">
          <h2>{table.title}</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {table.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, rowIndex) => (
                  <tr key={`${table.title}-${rowIndex}`}>
                    {row.map((cell, cellIndex) => (
                      <td key={`${table.columns[cellIndex]}-${cellIndex}`}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      {result.report ? (
        <section>
          <h2>Model summary</h2>
          <pre className="report">{result.report}</pre>
        </section>
      ) : null}
    </div>
  );
}

export default function App() {
  const [demos, setDemos] = useState([]);
  const [loadError, setLoadError] = useState("");
  const [activeId, setActiveId] = useState(null);
  const [sessions, setSessions] = useState({});
  const [runningKey, setRunningKey] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [batch, setBatch] = useState(null);
  const sessionsRef = useRef({});
  const batchRef = useRef(false);
  const stopRef = useRef(false);
  sessionsRef.current = sessions;

  useEffect(() => {
    fetchDemos()
      .then((list) => {
        setDemos(list);
        setActiveId(list[0]?.id ?? null);
        const initial = {};
        for (const demo of list) initial[demo.id] = createSession(demo);
        setSessions(initial);
      })
      .catch((error) => setLoadError(error.message));
  }, []);

  useEffect(() => {
    if (!runningKey) return undefined;
    const started = Date.now();
    setElapsed(0);
    const timer = setInterval(() => setElapsed(Date.now() - started), 500);
    return () => clearInterval(timer);
  }, [runningKey]);

  const active = demos.find((demo) => demo.id === activeId) || null;
  const session = active ? sessions[active.id] : null;
  const tab = session?.tabs.find((item) => item.id === session.activeTabId) || null;
  const groups = useMemo(() => (active ? groupFields(active.fields) : []), [active]);
  const finishedCount = session?.tabs.filter((item) => item.data).length || 0;
  const chapters = [];
  for (const demo of demos) {
    const last = chapters[chapters.length - 1];
    if (!last || last.name !== demo.chapter) chapters.push({ name: demo.chapter, items: [demo] });
    else last.items.push(demo);
  }

  function updateSession(demoId, recipe) {
    setSessions((current) => {
      const next = recipe(current[demoId]);
      return { ...current, [demoId]: next };
    });
  }

  function updateField(name, value) {
    const demoId = active.id;
    const tabId = session.activeTabId;
    updateSession(demoId, (current) => ({
      ...current,
      tabs: current.tabs.map((item) =>
        item.id === tabId ? { ...item, values: { ...item.values, [name]: value } } : item
      ),
    }));
  }

  function selectTab(tabId) {
    updateSession(active.id, (current) => ({ ...current, activeTabId: tabId }));
  }

  function addTab() {
    updateSession(active.id, (current) => {
      const source = current.tabs.find((item) => item.id === current.activeTabId);
      const created = createTab(active, source.values, current.nextNumber);
      return {
        ...current,
        nextNumber: current.nextNumber + 1,
        activeTabId: created.id,
        tabs: [...current.tabs, created],
      };
    });
  }

  function closeTab(tabId) {
    if (`${active.id}:${tabId}` === runningKey) return;
    updateSession(active.id, (current) => {
      if (current.tabs.length === 1) return current;
      const tabs = current.tabs.filter((item) => item.id !== tabId);
      const activeTabId = current.activeTabId === tabId ? tabs[tabs.length - 1].id : current.activeTabId;
      return { ...current, tabs, activeTabId };
    });
  }

  function renameTab(tabId, name) {
    updateSession(active.id, (current) => ({
      ...current,
      tabs: current.tabs.map((item) => (item.id === tabId ? { ...item, name } : item)),
    }));
  }

  function stopAll() {
    stopRef.current = true;
    setBatch((current) => (current?.running ? { ...current, stopping: true } : current));
  }

  function clearResults() {
    if (runningKey || batchRef.current) return;
    setSessions((current) => {
      const next = {};
      for (const [demoId, session] of Object.entries(current)) {
        next[demoId] = {
          ...session,
          compare: false,
          tabs: session.tabs.map((item) => ({ ...item, status: "idle", error: "", data: null })),
        };
      }
      return next;
    });
    setBatch({
      running: false,
      stopping: false,
      index: 0,
      total: 0,
      demoTitle: "",
      tabName: "",
      summary: "Cleared measures and graphs.",
    });
  }

  async function runAll() {
    if (batchRef.current || runningKey || demos.length === 0) return;
    const queue = [];
    for (const demo of demos) {
      const current = sessionsRef.current[demo.id];
      if (!current) continue;
      for (const item of current.tabs) {
        queue.push({ demoId: demo.id, title: demo.title, tabId: item.id });
      }
    }
    if (queue.length === 0) return;

    batchRef.current = true;
    stopRef.current = false;
    let failed = 0;
    let completed = 0;
    try {
      for (let index = 0; index < queue.length; index += 1) {
        if (stopRef.current) break;
        const job = queue[index];
        const live = sessionsRef.current[job.demoId]?.tabs.find((item) => item.id === job.tabId);
        if (!live) continue;
        const params = { ...live.values };
        const tabName = live.name || "Untitled";
        const key = `${job.demoId}:${job.tabId}`;
        setActiveId(job.demoId);
        setBatch({
          running: true,
          stopping: false,
          index,
          total: queue.length,
          demoTitle: job.title,
          tabName,
          summary: "",
        });
        setRunningKey(key);
        updateSession(job.demoId, (current) => ({
          ...current,
          activeTabId: job.tabId,
          tabs: current.tabs.map((item) =>
            item.id === job.tabId ? { ...item, status: "running", error: "" } : item
          ),
        }));
        try {
          const data = await runDemo(job.demoId, params);
          updateSession(job.demoId, (current) => ({
            ...current,
            tabs: current.tabs.map((item) =>
              item.id === job.tabId ? { ...item, status: "done", error: "", data } : item
            ),
          }));
        } catch (error) {
          failed += 1;
          updateSession(job.demoId, (current) => ({
            ...current,
            tabs: current.tabs.map((item) =>
              item.id === job.tabId ? { ...item, status: "error", error: error.message, data: item.data } : item
            ),
          }));
        } finally {
          setRunningKey("");
        }
        completed += 1;
      }
    } finally {
      batchRef.current = false;
      const stopped = stopRef.current;
      const problem = failed ? ` ${failed} failed.` : "";
      setBatch({
        running: false,
        stopping: false,
        index: completed,
        total: queue.length,
        demoTitle: "",
        tabName: "",
        summary: stopped
          ? `Stopped after ${completed} of ${queue.length} runs.${problem}`
          : `Finished ${completed} runs.${problem}`,
      });
    }
  }

  async function onRun() {
    if (!active || !tab || runningKey || batchRef.current) return;
    setBatch(null);
    const demoId = active.id;
    const tabId = tab.id;
    const params = tab.values;
    const key = `${demoId}:${tabId}`;
    setRunningKey(key);
    updateSession(demoId, (current) => ({
      ...current,
      tabs: current.tabs.map((item) => (item.id === tabId ? { ...item, status: "running", error: "" } : item)),
    }));
    try {
      const data = await runDemo(demoId, params);
      updateSession(demoId, (current) => ({
        ...current,
        tabs: current.tabs.map((item) =>
          item.id === tabId ? { ...item, status: "done", error: "", data } : item
        ),
      }));
    } catch (error) {
      updateSession(demoId, (current) => ({
        ...current,
        tabs: current.tabs.map((item) =>
          item.id === tabId ? { ...item, status: "error", error: error.message, data: item.data } : item
        ),
      }));
    } finally {
      setRunningKey("");
    }
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <div>
            <strong>Finntech Lab</strong>
            <span>Chapters 7 and 8</span>
          </div>
        </div>
        <nav>
          {chapters.map((chapter) => (
            <div key={chapter.name} className="nav-group">
              <p>{chapter.name}</p>
              {chapter.items.map((demo) => (
                <button
                  key={demo.id}
                  type="button"
                  className={demo.id === activeId ? "nav-item active" : "nav-item"}
                  onClick={() => setActiveId(demo.id)}
                >
                  {demo.title}
                </button>
              ))}
            </div>
          ))}
        </nav>
      </aside>

      <main>
        {demos.length > 0 ? (
          <div className="run-all">
            <div className="run-all-actions">
              {batch?.running ? (
                <button type="button" className="secondary" onClick={stopAll} disabled={batch.stopping}>
                  {batch.stopping ? "Stopping" : "Stop"}
                </button>
              ) : (
                <button type="button" className="primary" onClick={runAll} disabled={Boolean(runningKey)}>
                  Run all
                </button>
              )}
              <button
                type="button"
                className="secondary"
                onClick={clearResults}
                disabled={Boolean(runningKey) || Boolean(batch?.running)}
              >
                Clear results
              </button>
            </div>
            <p>
              {batch?.running
                ? `${batch.stopping ? "Finishing this model, then stopping. " : ""}${batch.demoTitle} · ${batch.tabName} (${batch.index + 1} of ${batch.total}) · ${formatElapsed(elapsed)}`
                : batch?.summary || "Runs every feature and every open tab, one after another."}
            </p>
          </div>
        ) : null}

        {loadError ? (
          <section className="banner error">
            <h1>API offline</h1>
            <p>{loadError}</p>
          </section>
        ) : null}

        {!active && !loadError ? <p className="loading">Loading demos…</p> : null}

        {active && session && tab ? (
          <>
            <header className="demo-head">
              <p className="kicker">
                {active.chapter}
                <code>{active.script}</code>
              </p>
              <h1>{active.title}</h1>
              <p className="blurb">{active.blurb}</p>
            </header>

            <div className="tabbar" role="tablist" aria-label={`${active.title} runs`}>
              {session.tabs.map((item) => {
                const selected = item.id === session.activeTabId;
                const itemRunning = runningKey === `${active.id}:${item.id}`;
                return (
                  <div
                    key={item.id}
                    className={selected ? "tab selected" : "tab"}
                    role="tab"
                    aria-selected={selected}
                    onClick={() => selectTab(item.id)}
                  >
                    {item.status === "running" || item.status === "error" || item.data ? (
                      <span className={`tab-dot ${item.status === "error" ? "error" : ""} ${itemRunning ? "running" : ""}`} />
                    ) : null}
                    <input
                      className="tab-name"
                      aria-label={`${item.name} name`}
                      value={item.name}
                      onFocus={() => selectTab(item.id)}
                      onChange={(event) => renameTab(item.id, event.target.value)}
                    />
                    {session.tabs.length > 1 ? (
                      <button
                        type="button"
                        className="tab-close"
                        aria-label={`Close ${item.name}`}
                        disabled={itemRunning}
                        onClick={(event) => {
                          event.stopPropagation();
                          closeTab(item.id);
                        }}
                      >
                        ×
                      </button>
                    ) : null}
                  </div>
                );
              })}
              <button type="button" className="tab-add" onClick={addTab}>
                New tab
              </button>
              <button
                type="button"
                className={session.compare ? "tab-add on" : "tab-add"}
                disabled={finishedCount < 2}
                onClick={() => updateSession(active.id, (current) => ({ ...current, compare: !current.compare }))}
              >
                Compare
              </button>
            </div>
            <p className="tab-note">Each tab keeps its own inputs and results. Add a tab to try another setup of this model.</p>

            <section className="panel">
              {groups.map((group) => (
                <fieldset key={group.name}>
                  <legend>{group.name}</legend>
                  <div className="form-grid">
                    {group.fields.map((field) => (
                      <FieldControl
                        key={field.name}
                        inputId={`${tab.id}-${field.name}`}
                        field={field}
                        value={tab.values[field.name]}
                        onChange={(value) => updateField(field.name, value)}
                      />
                    ))}
                  </div>
                </fieldset>
              ))}

              <div className="actions">
                <button type="button" className="primary" onClick={onRun} disabled={Boolean(runningKey) || Boolean(batch?.running)}>
                  {runningKey === `${active.id}:${tab.id}` ? `Running ${formatElapsed(elapsed)}` : "Run model"}
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={() =>
                    updateSession(active.id, (current) => ({
                      ...current,
                      tabs: current.tabs.map((item) =>
                        item.id === tab.id ? { ...item, values: defaultsFor(active) } : item
                      ),
                    }))
                  }
                  disabled={Boolean(runningKey)}
                >
                  Reset
                </button>
                {(active.presets || []).map((preset) => (
                  <button
                    key={preset.id}
                    type="button"
                    className="secondary"
                    disabled={Boolean(runningKey)}
                    onClick={() =>
                      updateSession(active.id, (current) => ({
                        ...current,
                        tabs: current.tabs.map((item) =>
                          item.id === tab.id ? { ...item, values: { ...defaultsFor(active), ...preset.values } } : item
                        ),
                      }))
                    }
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <p className="runtime">
                {runningKey && runningKey !== `${active.id}:${tab.id}` ? "Another model is still running. " : ""}
                {active.runtime}
              </p>
            </section>

            {runningKey === `${active.id}:${tab.id}` ? (
              <div className="running" role="status">
                <span className="pulse" />
                Fitting {tab.name} · {formatElapsed(elapsed)}
              </div>
            ) : null}

            {tab.error ? (
              <div className="banner error" role="alert">
                {tab.error}
              </div>
            ) : null}

            {session.compare && finishedCount >= 2 ? (
              <section className="table-block compare">
                <h2>Compared runs</h2>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Measure</th>
                        {session.tabs.map((item) => (
                          <th key={item.id}>{item.name || "Untitled"}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {metricRows(session.tabs).map((row) => (
                        <tr key={row.label}>
                          <td>{row.label}</td>
                          {row.values.map((value, index) => (
                            <td key={session.tabs[index].id}>{value}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            ) : null}

            {tab.data ? (
              <Results demo={active} result={tab.data} />
            ) : runningKey === `${active.id}:${tab.id}` ? null : (
              <section className="empty">
                <h2>What this run produces</h2>
                <ul>
                  {active.outputs.map((output) => (
                    <li key={output}>{output}</li>
                  ))}
                </ul>
              </section>
            )}
          </>
        ) : null}

        <footer>Prices come from Yahoo Finance. Every model runs on this computer.</footer>
      </main>
    </div>
  );
}
