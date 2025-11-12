import React, { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Search, Plus, Send, Loader2, FileText, X, BookOpen, Bookmark,
  Link as LinkIcon, Sparkles, Settings, ChevronLeft, ChevronRight,
  Filter, Sun, Moon
} from "lucide-react";

/**
 * NotebookLM-style Clinical RAG UI (local, files preloaded; no upload)
 * Tailwind v4 compatible. Backend endpoints expected:
 *   GET  /api/files               -> [{ id, title, path, kind, section?, size? }]
 *   GET  /api/files?q=term        -> filtered list
 *   POST /api/chat {query, topK, temperature} -> {answer, sources:[...]}
 */

// ---------- Tiny UI primitives ----------
const Button = ({ className = "", variant = "", size = "", disabled, ...props }) => (
  <button
    disabled={disabled}
    className={[
      "inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 text-sm font-medium shadow-sm transition",
      variant === "ghost" && "bg-transparent hover:bg-zinc-900/5 dark:hover:bg-zinc-100/5",
      variant === "outline" && "border border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-900/40",
      variant === "primary" && "bg-zinc-900 text-white hover:bg-black disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900",
      size === "icon" && "p-2 aspect-square",
      className,
    ].filter(Boolean).join(" ")}
    {...props}
  />
);

const Input = ({ className = "", ...props }) => (
  <input
    className={[
      "w-full rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm outline-none ring-0 focus:border-zinc-300",
      "dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100",
      className,
    ].join(" ")}
    {...props}
  />
);

const Textarea = ({ className = "", ...props }) => (
  <textarea
    className={[
      "w-full resize-none rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm outline-none ring-0 focus:border-zinc-300",
      "dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100",
      className,
    ].join(" ")}
    {...props}
  />
);

const cn = (...a) => a.filter(Boolean).join(" ");

/** @typedef {{ id: string, title: string, path?: string, kind: string, section?: string, size?: number }} FileItem */
/** @typedef {{ role: "user"|"assistant"|"system", content: string, sources?: FileItem[] }} Message */

export default function NotebookRAGApp() {
  const [messages, setMessages] = useState(/** @type {Message[]} */([]));
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  // THEME: init from localStorage or system preference
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return true;
    const saved = localStorage.getItem("theme");
    return saved ? saved === "dark"
                 : window.matchMedia?.("(prefers-color-scheme: dark)")?.matches ?? true;
  });

  const [rightOpen, setRightOpen] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [params, setParams] = useState({ topK: 5, temperature: 0.1 });
  const [files, setFiles] = useState(/** @type {FileItem[]} */([]));
  const [q, setQ] = useState("");

  // Apply theme + persist
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  // Load library files
  async function loadFiles(query = "") {
    try {
      const url = query ? `/api/files?q=${encodeURIComponent(query)}` : "/api/files";
      const res = await fetch(url);
      const data = await res.json();
      setFiles(data || []);
    } catch (e) {
      console.warn("files load error", e);
      setFiles([]);
    }
  }
  useEffect(() => { loadFiles(); }, []);

  // Scroll chat to bottom
  const bottomRef = useRef(null);
  useEffect(() => { bottomRef.current?.scrollIntoView?.({ behavior: "smooth" }); }, [messages, busy]);

  async function handleSend() {
    if (!draft.trim()) return;
    const userMsg = { role: "user", content: draft };
    setMessages(prev => [...prev, userMsg]);
    setDraft("");
    setBusy(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMsg.content,
          topK: Number(params.topK) || 5,
          temperature: Number(params.temperature) || 0.1,
          stream: false,
        }),
      });
      const data = await res.json();
      const assistantMsg = { role: "assistant", content: data.answer || "", sources: data.sources || [] };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (e) {
      setMessages(prev => [...prev, { role: "assistant", content: "(Backend error. Is your local API running?)" }]);
    } finally {
      setBusy(false);
    }
  }

  function clearChat() { setMessages([]); }

  const grouped = groupByFolder(files);

  return (
    <div className="flex h-[100dvh] w-full overflow-hidden text-zinc-900 antialiased bg-white dark:bg-zinc-950 dark:text-zinc-100 transition-colors">
      {/* Sidebar: Retrieval params + search */}
      <AnimatePresence initial={false}>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -24, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -24, opacity: 0 }}
            transition={{ type: "spring", stiffness: 260, damping: 24 }}
            className="hidden w-[300px] shrink-0 border-r border-zinc-200 bg-zinc-50/60 p-4 backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-900/40 md:block"
          >
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                <h2 className="text-sm font-semibold">Notebook</h2>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" onClick={() => setDark(d => !d)} aria-label="Toggle theme">
                  {dark ? <Sun className="h-4 w-4"/> : <Moon className="h-4 w-4"/>}
                </Button>
              </div>
            </div>

            <div className="mb-4">
              <div className="mb-1 text-xs opacity-60">Search files</div>
              <div className="flex items-center gap-2">
                <Input
                  placeholder="e.g. CSF gram stain"
                  value={q}
                  onChange={(e)=>setQ(e.target.value)}
                  onKeyDown={(e)=>{ if (e.key==='Enter') loadFiles(q); }}
                />
                <Button variant="outline" size="icon" onClick={()=>loadFiles(q)} aria-label="Search">
                  <Search className="h-4 w-4"/>
                </Button>
              </div>
            </div>

            <div>
              <div className="mb-2 flex items-center gap-2 text-xs opacity-60">
                <Filter className="h-3.5 w-3.5"/> Retrieval
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs opacity-70 w-16">Top-K</label>
                <Input
                  type="number" min={1} max={50} value={params.topK}
                  onChange={(e)=>setParams(p=>({...p, topK: e.target.value}))}
                  className="h-8 w-24"
                />
              </div>
              <div className="mt-2 flex items-center gap-2">
                <label className="text-xs opacity-70 w-16">Temp</label>
                <Input
                  type="number" step={0.1} min={0} max={2} value={params.temperature}
                  onChange={(e)=>setParams(p=>({...p, temperature: e.target.value}))}
                  className="h-8 w-24"
                />
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <div className="flex h-14 items-center justify-between gap-2 border-b border-zinc-200 bg-white/70 px-3 backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-950/40">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" onClick={()=>setSidebarOpen(v=>!v)} aria-label="Toggle sidebar">
              {sidebarOpen ? <ChevronLeft className="h-4 w-4"/> : <ChevronRight className="h-4 w-4"/>}
            </Button>
            <div className="flex items-center gap-2">
              <Sparkles className="h-5 w-5"/>
              <div className="text-sm">
                <div className="font-semibold leading-tight">Clinical RAG – Notebook view</div>
                <div className="text-[11px] opacity-60">Files pre-indexed locally</div>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" aria-label="Settings"><Settings className="h-4 w-4"/></Button>
          </div>
        </div>

        {/* Content area */}
        <div className="flex min-h-0 flex-1">
          {/* Chat column */}
          <div className="flex min-w-0 flex-1 flex-col">
            {/* Messages */}
            <div className="flex-1 space-y-4 overflow-y-auto p-4">
              {messages.length === 0 && (
                <div className="mx-auto mt-16 max-w-xl rounded-2xl border border-zinc-200 bg-white p-6 text-center text-sm opacity-80 dark:border-zinc-800 dark:bg-zinc-900">
                  <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-xl bg-zinc-100 dark:bg-zinc-800">
                    <Search className="h-5 w-5"/>
                  </div>
                  <div className="text-base font-medium">Ask across your SOPs</div>
                  <div className="mt-1 text-sm opacity-70">Examples: “What’s the hemolysis rejection threshold?” · “Summarize steps for CSF gram stain.”</div>
                </div>
              )}

              {messages.map((m, i) => (
                <div key={i} className={cn("flex w-full gap-3", m.role === "user" ? "justify-end" : "justify-start")}>
                  {m.role === "assistant" && (
                    <div className="mt-1 h-7 w-7 shrink-0 rounded-full bg-gradient-to-br from-zinc-900 to-zinc-700 dark:from-zinc-100 dark:to-zinc-400"/>
                  )}
                  <div
                    className={cn(
                      "max-w-[min(78ch,85%)] rounded-2xl border px-4 py-3 text-sm shadow-sm",
                      m.role === "user"
                        ? "rounded-br-md border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900"
                        : "rounded-bl-md border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900/50"
                    )}
                  >
                    <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
                    {m.sources?.length ? <Citations sources={m.sources}/> : null}
                  </div>
                  {m.role === "user" && (
                    <div className="mt-1 h-7 w-7 shrink-0 rounded-full bg-zinc-200 dark:bg-zinc-700"/>
                  )}
                </div>
              ))}
              {busy && (
                <div className="flex items-center gap-2 text-sm opacity-70">
                  <Loader2 className="h-4 w-4 animate-spin"/>Thinking…
                </div>
              )}
              <div ref={bottomRef}/>
            </div>

            {/* Composer */}
            <div className="border-t border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-end gap-2">
                <Textarea
                  rows={1}
                  value={draft}
                  onChange={(e)=>setDraft(e.target.value)}
                  onKeyDown={(e)=>{ if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                  placeholder="Ask a clinical question or request a summary…"
                />
                <Button variant="primary" onClick={handleSend} disabled={busy || !draft.trim()} aria-label="Send">
                  <Send className="h-4 w-4"/>
                </Button>
                <Button variant="outline" onClick={clearChat}>Clear</Button>
              </div>
            </div>
          </div>

          {/* Right column: Files / Library */}
          <AnimatePresence initial={false}>
            {rightOpen && (
              <motion.aside
                initial={{ x: 24, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: 24, opacity: 0 }}
                transition={{ type: "spring", stiffness: 260, damping: 24 }}
                className="hidden w-[360px] shrink-0 border-l border-zinc-200 bg-zinc-50/60 p-3 backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-900/40 lg:block"
              >
                <div className="mb-2 flex items-center justify-between px-1">
                  <div className="flex items-center gap-2 text-sm font-semibold"><LinkIcon className="h-4 w-4"/>Files</div>
                </div>
                <div className="space-y-3 overflow-y-auto p-1">
                  {Object.keys(grouped).length === 0 && (
                    <div className="rounded-xl border border-dashed border-zinc-300 p-3 text-sm opacity-70 dark:border-zinc-700">
                      No files found. Ensure your backend exposes /api/files
                    </div>
                  )}
                  {Object.entries(grouped).map(([folder, items]) => (
                    <div key={folder} className="rounded-xl border border-zinc-200 bg-white p-2 text-sm shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                      <div className="mb-1 truncate text-xs font-semibold opacity-70" title={folder}>{folder || "/"}</div>
                      <ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
                        {items.map((s) => (
                          <li key={s.id} className="py-1.5">
                            <a
                              href={s.path || "#"} target={s.path ? "_blank" : undefined} rel={s.path ? "noreferrer" : undefined}
                              className="group flex items-center gap-2 truncate rounded-lg px-2 py-1 hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
                            >
                              <KindIcon kind={s.kind}/>
                              <span className="truncate" title={s.title}>{s.title}</span>
                              <span className="ml-auto text-[10px] opacity-60">{s.section || s.kind}</span>
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </motion.aside>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

function Citations({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="mt-3 space-y-1">
      <div className="text-[11px] font-medium uppercase tracking-wide opacity-70">Citations</div>
      <div className="flex flex-wrap gap-2">
        {sources.map((s, idx) => (
          <a
            key={s.id || idx}
            href={s.path || "#"}
            target={s.path ? "_blank" : undefined}
            rel={s.path ? "noreferrer" : undefined}
            className="group inline-flex max-w-[16rem] items-center gap-1 truncate rounded-full border border-zinc-200 bg-white px-2 py-1 text-[11px] shadow-sm hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900"
          >
            <KindIcon kind={s.kind}/>
            <span className="truncate" title={s.title}>{s.title}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

function KindIcon({ kind }) {
  const k = String(kind || "").toLowerCase();
  if (k.includes("pdf")) return <FileText className="h-3.5 w-3.5 opacity-70"/>;
  if (k.includes("doc")) return <FileText className="h-3.5 w-3.5 opacity-70"/>;
  if (k.includes("url") || k.includes("http")) return <LinkIcon className="h-3.5 w-3.5 opacity-70"/>;
  return <FileText className="h-3.5 w-3.5 opacity-70"/>;
}

function groupByFolder(items /**: FileItem[] */) {
  const out = {};
  for (const it of items || []) {
    const folder = (it.path || "").split("/").slice(0, -1).join("/");
    out[folder] = out[folder] || [];
    out[folder].push(it);
  }
  return out;
}
