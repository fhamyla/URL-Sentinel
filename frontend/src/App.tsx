// Copyright (c) 2026 fhamyla
// This file is part of URL Sentinel and is licensed under the MIT License.
// See the LICENSE file in the project root for license information.

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Sparkles,
  Heart,
  Wand2,
  Link2,
  Copy,
  Check,
  ExternalLink,
  ScanLine,
  Loader2,
  Clock,
  Stars,
  Flower2,
  Feather,
  ChevronRight,
  Lock,
  Unlock,
  Info,
  X,
  BookOpen,
  HelpCircle,
  ArrowRight,
  Sun,
  Moon,
} from "lucide-react";
import { cn } from "./utils/cn";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

type Verdict = "safe" | "suspicious" | "phishing";
type Theme = "light" | "dark-sepia";

interface Feature {
  id: string;
  label: string;
  value: string;
  status: "good" | "warn" | "bad" | "neutral";
  description: string;
}

interface AnalysisResult {
  url: string;
  domain: string;
  verdict: Verdict;
  riskScore: number;
  confidence: number;
  features: Feature[];
  summary: string;
  scannedAt: string;
  aura: string;
}


const SAMPLE_URLS = [
  "https://notion.so/login",
  "http://secure-paypal-verify-account.com/login",
  "https://www.pinterest.com/search/pins/?q=aesthetic",
  "https://dribbble.com/shots/popular",
];

function getDomain(url: string) {
  try {
    const u = new URL(url.startsWith("http") ? url : `http://${url}`);
    return u.hostname.replace(/^www\./, "");
  } catch {
    return url.split("/")[0];
  }
}

/* ──────────────────────────────────────────────────────
   Theme hook — persists choice to localStorage
   ────────────────────────────────────────────────────── */
function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window !== "undefined") {
      return (localStorage.getItem("url-sentinel-theme") as Theme) || "light";
    }
    return "light";
  });

  const setTheme = useCallback((t: Theme) => {
    const root = document.documentElement;
    root.classList.add("theme-transitioning");
    setThemeState(t);
    localStorage.setItem("url-sentinel-theme", t);
    root.setAttribute("data-theme", t === "light" ? "" : t);
    // Remove the transitioning class after CSS transitions finish
    setTimeout(() => root.classList.remove("theme-transitioning"), 500);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme === "light" ? "" : theme);
  }, []);

  return { theme, setTheme, toggleTheme: () => setTheme(theme === "light" ? "dark-sepia" : "light") };
}

/* ──────────────────────────────────────────────────────
   Session hook — anonymous persistent browser identity
   ────────────────────────────────────────────────────── */
function generateUUID(): string {
  // crypto.randomUUID is available in all modern browsers
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older browsers
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

const SESSION_STORAGE_KEY = "url-sentinel-session-id";

function useSessionId(): string {
  const [sessionId] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const existing = localStorage.getItem(SESSION_STORAGE_KEY);
      if (existing) return existing;
      const id = generateUUID();
      localStorage.setItem(SESSION_STORAGE_KEY, id);
      return id;
    }
    return generateUUID();
  });
  return sessionId;
}

/* ──────────────────────────────────────────────────────
   Theme Toggle Button Component
   ────────────────────────────────────────────────────── */
function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const isDark = theme === "dark-sepia";

  return (
    <button
      id="theme-toggle"
      onClick={onToggle}
      className="theme-toggle"
      aria-label={isDark ? "Switch to light mode" : "Switch to dark sepia mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark sepia mode"}
    >
      <AnimatePresence mode="wait">
        {isDark ? (
          <motion.div
            key="sun"
            initial={{ rotate: -90, scale: 0, opacity: 0 }}
            animate={{ rotate: 0, scale: 1, opacity: 1 }}
            exit={{ rotate: 90, scale: 0, opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.34, 1.56, 0.64, 1] }}
            className="flex items-center justify-center"
          >
            <Sun className="size-[18px]" />
          </motion.div>
        ) : (
          <motion.div
            key="moon"
            initial={{ rotate: -90, scale: 0, opacity: 0 }}
            animate={{ rotate: 0, scale: 1, opacity: 1 }}
            exit={{ rotate: 90, scale: 0, opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.34, 1.56, 0.64, 1] }}
            className="flex items-center justify-center"
          >
            <Moon className="size-[18px]" />
          </motion.div>
        )}
      </AnimatePresence>
    </button>
  );
}



function VerdictPill({ verdict }: { verdict: Verdict }) {
  const cfg = {
    safe: {
      icon: ShieldCheck,
      label: "Safe & Sweet",
    },
    suspicious: {
      icon: ShieldAlert,
      label: "Hmm… Suspicious",
    },
    phishing: {
      icon: ShieldX,
      label: "Phishing Alert",
    },
  }[verdict];
  const Icon = cfg.icon;

  const verdictStyles = {
    safe: {
      background: `linear-gradient(to right, var(--verdict-safe-from), var(--verdict-safe-to))`,
      color: `var(--verdict-safe-text)`,
      boxShadow: `inset 0 0 0 1px var(--verdict-ring)`,
    },
    suspicious: {
      background: `linear-gradient(to right, var(--verdict-sus-from), var(--verdict-sus-to))`,
      color: `var(--verdict-sus-text)`,
      boxShadow: `inset 0 0 0 1px var(--verdict-ring)`,
    },
    phishing: {
      background: `linear-gradient(to right, var(--verdict-phish-from), var(--verdict-phish-to))`,
      color: `var(--verdict-phish-text)`,
      boxShadow: `inset 0 0 0 1px var(--verdict-ring)`,
    },
  }[verdict];

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium shadow-sm"
      style={verdictStyles}
    >
      <Icon className="size-3.5" />
      {cfg.label}
    </span>
  );
}

function RiskRing({ score }: { score: number }) {
  const radius = 58;
  const stroke = 10;
  const normalizedRadius = radius - stroke / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const progress = score / 100;
  const offset = circumference - progress * circumference;

  const color =
    score < 30 ? "#f472b6" : score < 65 ? "#fb923c" : "#f43f5e";

  return (
    <div className="relative w-[120px] h-[120px] sm:w-[150px] sm:h-[150px]">
      <svg
        viewBox={`0 0 ${radius * 2} ${radius * 2}`}
        className="w-full h-full -rotate-90"
      >
        <defs>
          <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={color} stopOpacity="1" />
            <stop offset="100%" stopColor="#c084fc" stopOpacity="0.6" />
          </linearGradient>
          <filter id="softGlow">
            <feGaussianBlur stdDeviation="4" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <circle
          stroke="var(--color-ring-track)"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          className="drop-shadow-sm"
        />
        <motion.circle
          stroke="url(#g1)"
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.4, ease: "easeOut" }}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          filter="url(#softGlow)"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="font-display text-4xl sm:text-5xl leading-none" style={{ color: 'var(--color-risk-text)' }}>{score}</div>
        <div className="mt-1 text-[10px] uppercase tracking-[0.18em]" style={{ color: 'var(--color-risk-sub)' }}>
          Risk Score
        </div>
      </div>
    </div>
  );
}

function GuidesModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 backdrop-blur-sm"
            style={{ backgroundColor: 'var(--color-backdrop)' }}
            onClick={onClose}
          />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ type: "spring", duration: 0.5 }}
              className="relative max-h-[85vh] w-full max-w-3xl overflow-hidden rounded-[32px] shadow-2xl backdrop-blur-2xl"
              style={{
                backgroundColor: 'var(--color-surface-glass)',
                borderColor: 'var(--color-border)',
                borderWidth: '1px',
                boxShadow: `0 25px 50px -12px var(--shadow-heavy)`,
              }}
            >
              <div className="relative max-h-[85vh] overflow-y-auto">
                {/* Sticky header */}
                <div
                  className="sticky top-0 z-10 flex items-center justify-between px-5 sm:px-8 py-5 sm:py-6 backdrop-blur-xl"
                  style={{
                    background: `linear-gradient(to right, var(--color-modal-header-from), var(--color-modal-header-to))`,
                    borderBottom: `1px solid var(--color-border-light)`,
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="flex size-10 items-center justify-center rounded-2xl shadow-md"
                      style={{ background: `linear-gradient(to bottom right, var(--color-accent-from), var(--color-accent-to))` }}
                    >
                      <BookOpen className="size-5 text-white" />
                    </div>
                    <div>
                      <h2 className="font-display text-2xl" style={{ color: 'var(--color-text-primary)' }}>How to use URL Sentinel</h2>
                      <p className="text-sm" style={{ color: 'var(--color-text-tertiary)' }}>Your gentle guide to safer browsing</p>
                    </div>
                  </div>
                  <button
                    onClick={onClose}
                    className="flex size-9 items-center justify-center rounded-full transition"
                    style={{
                      border: `1px solid var(--color-border)`,
                      backgroundColor: 'var(--color-surface-solid)',
                      color: 'var(--color-text-accent)',
                    }}
                  >
                    <X className="size-4" />
                  </button>
                </div>

                <div className="px-5 sm:px-8 py-6 sm:py-8 space-y-8">
                  {/* Quick start */}
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Sparkles className="size-4" style={{ color: 'var(--color-text-accent)' }} />
                      <h3 className="font-display text-xl" style={{ color: 'var(--color-text-primary)' }}>Quick Start</h3>
                    </div>
                    <div className="space-y-3">
                      {[
                        { step: "1", title: "Paste your link", desc: "Copy any suspicious URL and paste it into the input field. We support http, https, and even bare domains." },
                        { step: "2", title: "Tap Analyze", desc: "Our gentle scanner checks 40+ signals including domain age, keywords, and patterns." },
                        { step: "3", title: "Read your aura", desc: "Get an instant verdict with risk score, confidence, and a pretty breakdown of what we found." },
                      ].map((item) => (
                        <div
                          key={item.step}
                          className="flex gap-4 rounded-2xl p-4"
                          style={{
                            border: `1px solid var(--color-border-light)`,
                            background: `linear-gradient(to right, var(--color-step-from), var(--color-step-to))`,
                          }}
                        >
                          <div
                            className="flex size-8 shrink-0 items-center justify-center rounded-full text-sm font-medium text-white"
                            style={{ background: `linear-gradient(to bottom right, var(--color-accent-from), var(--color-accent-to))` }}
                          >
                            {item.step}
                          </div>
                          <div>
                            <div className="font-medium" style={{ color: 'var(--color-text-primary)' }}>{item.title}</div>
                            <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>{item.desc}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Understanding results */}
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Heart className="size-4" style={{ color: 'var(--color-text-accent)' }} />
                      <h3 className="font-display text-xl" style={{ color: 'var(--color-text-primary)' }}>Understanding results</h3>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-3">
                      {[
                        { verdict: "safe" as Verdict, title: "Safe & Sweet", desc: "Low risk. Standard security observed. Browse with peace.", color: "emerald" },
                        { verdict: "suspicious" as Verdict, title: "Hmm… Suspicious", desc: "Mixed signals. Be cautious with passwords or downloads.", color: "amber" },
                        { verdict: "phishing" as Verdict, title: "Phishing Alert", desc: "High risk. Likely trying to steal info. Avoid!", color: "rose" },
                      ].map((item) => (
                        <div
                          key={item.verdict}
                          className="rounded-2xl p-4"
                          style={{
                            border: `1px solid var(--color-border-light)`,
                            backgroundColor: 'var(--color-surface-solid)',
                          }}
                        >
                          <VerdictPill verdict={item.verdict} />
                          <div className="mt-3 text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>{item.title}</div>
                          <div className="mt-1 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>{item.desc}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Signal garden */}
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Flower2 className="size-4" style={{ color: 'var(--color-text-accent)' }} />
                      <h3 className="font-display text-xl" style={{ color: 'var(--color-text-primary)' }}>Signal garden explained</h3>
                    </div>
                    <div
                      className="rounded-2xl p-5"
                      style={{
                        border: `1px solid var(--color-border-light)`,
                        background: `linear-gradient(to bottom right, var(--gradient-soft-start), var(--gradient-soft-end))`,
                      }}
                    >
                      <div className="grid gap-4 sm:grid-cols-2">
                        {[
                          { name: "Whisper Words", desc: "Phishing loves urgent words like 'verify' or 'secure'." },
                          { name: "TLD Petal", desc: "Some domain endings (.tk, .xyz) are abused more." },
                        ].map((s) => (
                          <div key={s.name} className="flex gap-3">
                            <div className="mt-0.5 size-1.5 rounded-full" style={{ backgroundColor: 'var(--color-text-accent)' }} />
                            <div>
                              <div className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>{s.name}</div>
                              <div className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>{s.desc}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Tips */}
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <HelpCircle className="size-4" style={{ color: 'var(--color-text-accent)' }} />
                      <h3 className="font-display text-xl" style={{ color: 'var(--color-text-primary)' }}>Pro tips</h3>
                    </div>
                    <div className="space-y-2.5">
                      {[
                        "Always check the domain carefully — phishers use lookalikes like 'paypaI' (capital i) instead of 'paypal'",
                        "Legit sites never ask for passwords via email links",
                        "When in doubt, type the URL directly instead of clicking",
                        "Look for the padlock, but remember: HTTPS alone doesn't guarantee safety",
                      ].map((tip, i) => (
                        <div
                          key={i}
                          className="flex gap-3 rounded-xl px-4 py-3"
                          style={{
                            backgroundColor: 'var(--color-surface-solid)',
                            border: `1px solid var(--color-border-light)`,
                          }}
                        >
                          <ArrowRight className="mt-0.5 size-4 shrink-0" style={{ color: 'var(--color-text-accent)' }} />
                          <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>{tip}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Footer */}
                  <div
                    className="rounded-2xl p-5 text-center"
                    style={{
                      background: `linear-gradient(to right, var(--color-tip-from), var(--color-tip-to))`,
                      border: `1px solid var(--color-border)`,
                    }}
                  >
                    <p className="text-sm" style={{ color: 'var(--color-text-primary)' }}>
                      Stay curious, stay safe. URL Sentinel is your companion, not a replacement for critical thinking.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}

export default function App() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [copied, setCopied] = useState(false);
  const [guidesOpen, setGuidesOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  interface ModelInfo {
    model_name: string;
    threshold: number;
    metrics: Record<string, any>;
  }
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const { theme, toggleTheme } = useTheme();
  const sessionId = useSessionId();

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/api/history`, {
        headers: { "X-Session-Id": sessionId },
      });
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      } else {
        throw new Error();
      }
    } catch {
      const saved = localStorage.getItem(`phish-history-${sessionId}`);
      if (saved) {
        try {
          setHistory(JSON.parse(saved));
        } catch {}
      }
    }
  };

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_URL}/api/model-info`);
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch (err) {
      console.error("Failed to fetch model info:", err);
    }
  };

  useEffect(() => {
    fetchHistory();
    fetchModelInfo();
  }, []);

  useEffect(() => {
    if (history.length > 0) {
      localStorage.setItem(`phish-history-${sessionId}`, JSON.stringify(history.slice(0, 10)));
    }
  }, [history, sessionId]);

  const handleAnalyze = async (u = url) => {
    if (!u.trim()) {
      inputRef.current?.focus();
      return;
    }

    const isValidUrl = (str: string) => {
      const pattern = /^(https?:\/\/)?((\\d{1,3}\\.){3}\\d{1,3}|([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})(:\\d+)?([/?].*)?$/;
      return pattern.test(str.trim());
    };

    if (!isValidUrl(u)) {
      setError("Please enter a valid URL.");
      setResult(null);
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-Id": sessionId,
        },
        body: JSON.stringify({ url: u }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server returned ${res.status}`);
      }
      const analysis = await res.json();
      setResult(analysis);
      setHistory((h) => [analysis, ...h.filter((x) => x.url !== analysis.url)].slice(0, 10));
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to scan. Please check if the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const pastelBg = useMemo(() => {
    if (theme === "dark-sepia") {
      if (!result) return "from-[#1a1410] via-[#201a14] to-[#1a1410]";
      return result.verdict === "safe"
        ? "from-[#141e16] via-[#1a1410] to-[#181614]"
        : result.verdict === "suspicious"
        ? "from-[#1e1a10] via-[#1a1410] to-[#1c1610]"
        : "from-[#1e1414] via-[#1a1410] to-[#1c1418]";
    }
    if (!result) return "from-pink-100 via-rose-50 to-violet-100";
    return result.verdict === "safe"
      ? "from-emerald-50 via-pink-50 to-violet-100"
      : result.verdict === "suspicious"
      ? "from-amber-50 via-pink-50 to-orange-100"
      : "from-rose-100 via-pink-100 to-fuchsia-100";
  }, [result, theme]);

  return (
    <div
      className="min-h-screen antialiased"
      style={{ backgroundColor: 'var(--color-page)', color: 'var(--color-text-primary)' }}
    >
      {/* Soft background */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className={cn("absolute inset-0 bg-gradient-to-b", pastelBg)} />
        <img
          src="/images/girly-bg.jpg"
          alt=""
          className="absolute inset-0 h-full w-full object-cover mix-blend-soft-light"
          style={{ opacity: `var(--bg-image-opacity)` }}
        />
        <div
          className="absolute inset-0"
          style={{ background: `radial-gradient(60% 60% at 50% 0%, var(--color-radial-overlay), transparent 70%)` }}
        />
        <div
          className="absolute inset-x-0 top-0 h-px"
          style={{ background: `linear-gradient(to right, transparent, var(--color-top-line), transparent)` }}
        />
      </div>

      <div className="mx-auto max-w-[1100px] px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
        {/* Header */}
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 rounded-[20px] blur-2xl opacity-40" style={{ backgroundColor: 'var(--color-text-accent)' }} />
              <div
                className="relative flex size-11 items-center justify-center rounded-[20px] shadow-lg ring-1 ring-white/70"
                style={{
                  background: `linear-gradient(to bottom right, var(--color-accent-from), var(--color-accent-to))`,
                  boxShadow: `0 10px 15px -3px var(--shadow-heavy)`,
                }}
              >
                <Flower2 className="size-5 text-white" />
              </div>
            </div>
            <div>
              <div className="font-display text-[22px] leading-none tracking-tight" style={{ color: 'var(--color-text-primary)' }}>URL Sentinel</div>
              <div className="flex items-center gap-1 text-[11px] uppercase tracking-widest" style={{ color: 'var(--color-text-secondary)' }}>
                <Heart className="size-3" />
                aesthetic security
              </div>
            </div>
          </div>
          <nav className="flex items-center gap-1.5">
            <button
              onClick={() => setGuidesOpen(true)}
              className="hidden md:inline-flex rounded-full px-3 py-1.5 text-sm transition"
              style={{ color: 'var(--color-text-tertiary)' }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--color-surface-hover)';
                e.currentTarget.style.color = 'var(--color-text-primary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.color = 'var(--color-text-tertiary)';
              }}
            >
              Guides
            </button>
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
          </nav>
        </header>

        {/* Hero */}
        <section className="mx-auto mt-16 max-w-3xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div
              className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs shadow-sm backdrop-blur"
              style={{
                border: `1px solid var(--color-border)`,
                backgroundColor: 'var(--color-surface-alt)',
                color: 'var(--color-text-secondary)',
              }}
            >
              <Wand2 className="size-3.5" style={{ color: 'var(--color-text-accent)' }} />
              Soft security for the modern web
            </div>
            <h1 className="font-display mt-6 text-[48px] leading-[1.05] tracking-[-0.02em] sm:text-[64px]" style={{ color: 'var(--color-text-primary)' }}>
              Check links with
              <span className="relative mx-3 inline-block">
                <span className="relative z-10 bg-gradient-to-r from-rose-600 via-fuchsia-600 to-violet-600 bg-clip-text text-transparent italic px-2 pb-3 pt-1 -mx-2">
                  gentle
                </span>
                <span className="absolute inset-x-0 bottom-1 h-[10px] -rotate-1" style={{ backgroundColor: theme === 'dark-sepia' ? 'rgba(180, 115, 51, 0.35)' : 'rgba(252, 231, 243, 0.7)' }} />
              </span>
              intuition
            </h1>
            <p className="mx-auto mt-4 max-w-xl text-[15px] leading-relaxed" style={{ color: 'var(--color-text-tertiary)' }}>
              We read its aura — whisper words, TLD petals — and give you a pretty, honest verdict.
            </p>
          </motion.div>

          {/* Input */}
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="mt-10"
          >
            <div className="group relative">
              <div
                className="absolute -inset-1 rounded-[28px] opacity-60 blur-2xl transition group-focus-within:opacity-100"
                style={{ background: `linear-gradient(to right, var(--color-input-glow-from), var(--color-input-glow-via), var(--color-input-glow-to))` }}
              />
              <div
                className="relative flex items-center gap-2 rounded-[24px] p-2.5 shadow-xl backdrop-blur-xl"
                style={{
                  border: `1px solid var(--color-border)`,
                  backgroundColor: 'var(--color-surface)',
                  boxShadow: `0 20px 25px -5px var(--shadow-color)`,
                }}
              >
                <div className="pl-3 pr-1" style={{ color: 'var(--color-text-accent)' }}>
                  <Link2 className="size-5" />
                </div>
                <input
                  ref={inputRef}
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
                  placeholder="Paste a link, darling… e.g. https://secure-login-paypal.com"
                  className="h-14 w-full bg-transparent font-body text-[15px] outline-none"
                  style={{
                    color: 'var(--color-text-primary)',
                  }}
                />
                <button
                  onClick={() => handleAnalyze()}
                  disabled={loading}
                  className="inline-flex h-12 items-center gap-2 rounded-[16px] px-5 text-sm font-medium text-white shadow-md transition active:scale-[0.98] disabled:opacity-60 shrink-0"
                  style={{
                    background: `linear-gradient(to right, var(--color-accent-from), var(--color-accent-to))`,
                    boxShadow: `0 4px 6px -1px var(--shadow-heavy)`,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = `linear-gradient(to right, var(--color-accent-from-hover), var(--color-accent-to-hover))`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = `linear-gradient(to right, var(--color-accent-from), var(--color-accent-to))`;
                  }}
                >
                  {loading ? (
                    <>
                      <Loader2 className="size-4 animate-spin" />
                      Scanning
                    </>
                  ) : (
                    <>
                      <ScanLine className="size-4" />
                      Analyze
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
              <span className="inline-flex items-center gap-1">
                <Stars className="size-3.5" /> try:
              </span>
              {SAMPLE_URLS.map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    setUrl(s);
                    handleAnalyze(s);
                  }}
                  className="rounded-full px-3 py-1 transition"
                  style={{
                    border: `1px solid var(--color-sample-border)`,
                    backgroundColor: 'var(--color-sample-bg)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--color-sample-hover)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--color-sample-bg)';
                  }}
                >
                  {getDomain(s)}
                </button>
              ))}
            </div>
            {error && (
              <div
                className="mt-4 rounded-2xl px-4 py-2.5 text-xs shadow-sm backdrop-blur inline-block"
                style={{
                  backgroundColor: 'var(--color-error-bg)',
                  border: `1px solid var(--color-error-border)`,
                  color: 'var(--color-error-text)',
                }}
              >
                ⚠️ {error}
              </div>
            )}
          </motion.div>
        </section>

        {/* Results */}
        <AnimatePresence mode="wait">
          {loading && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mx-auto mt-16 grid max-w-5xl gap-4 sm:grid-cols-3"
            >
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-36 animate-pulse rounded-[28px]"
                  style={{
                    border: `1px solid var(--color-skeleton-border)`,
                    backgroundColor: 'var(--color-skeleton)',
                  }}
                />
              ))}
            </motion.div>
          )}

          {result && !loading && (
            <motion.section
              key="result"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mx-auto mt-16 max-w-6xl"
            >
              {/* Top card */}
              <div
                className="relative overflow-hidden rounded-[32px] p-[1px] shadow-2xl backdrop-blur-xl"
                style={{
                  border: `1px solid var(--color-border)`,
                  backgroundColor: 'var(--color-surface)',
                  boxShadow: `0 25px 50px -12px var(--shadow-color)`,
                }}
              >
                <div
                  className="absolute inset-0"
                  style={{ background: `linear-gradient(to bottom right, var(--gradient-surface-start), transparent, var(--gradient-surface-end))` }}
                />
                <div className="relative grid gap-10 p-5 sm:p-8 lg:grid-cols-[auto_1fr] lg:p-12">
                  <div className="flex flex-col items-center gap-4">
                    <RiskRing score={result.riskScore} />
                    <VerdictPill verdict={result.verdict} />
                    <div className="text-center">
                      <div className="font-display text-lg" style={{ color: 'var(--color-text-primary)' }}>{result.aura}</div>
                      <div className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                        Confidence {result.confidence}% • <Clock className="inline size-3" /> {new Date(result.scannedAt).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>

                  <div className="min-w-0">
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="font-display text-[28px] leading-tight break-words" style={{ color: 'var(--color-text-primary)' }}>
                          {result.domain}
                        </div>
                        <div className="mt-1 truncate text-sm" style={{ color: 'var(--color-text-secondary)' }}>{result.url}</div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(result.url);
                            setCopied(true);
                            setTimeout(() => setCopied(false), 1500);
                          }}
                          className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs transition"
                          style={{
                            border: `1px solid var(--color-border)`,
                            backgroundColor: 'var(--color-surface-solid)',
                            color: 'var(--color-text-primary)',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = 'var(--color-surface-hover)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = 'var(--color-surface-solid)';
                          }}
                        >
                          {copied ? <Check className="size-3.5" style={{ color: 'var(--color-status-good-text)' }} /> : <Copy className="size-3.5" />}
                          {copied ? "Copied" : "Copy"}
                        </button>
                        <a
                          href={result.url.startsWith("http") ? result.url : `http://${result.url}`}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs transition"
                          style={{
                            border: `1px solid var(--color-border)`,
                            backgroundColor: 'var(--color-surface-solid)',
                            color: 'var(--color-text-primary)',
                          }}
                        >
                          Open <ExternalLink className="size-3.5" />
                        </a>
                      </div>
                    </div>

                    <p className="mt-5 max-w-2xl font-body text-[15px] leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                      {result.summary}
                    </p>

                    <div className="mt-6 flex flex-wrap gap-2">
                      {[
                        "Soft scan",
                        "Privacy first"
                      ].map((tag) => (
                        <span
                          key={tag}
                          className="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs ring-1"
                          style={{
                            backgroundColor: 'var(--color-tag-bg)',
                            color: 'var(--color-tag-text)',
                            '--tw-ring-color': 'var(--color-tag-ring)',
                          } as React.CSSProperties}
                        >
                          <Feather className="size-3" />
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Features */}
              <div className="mt-6 grid gap-5 grid-cols-1 lg:grid-cols-3">
                <div
                  className="lg:col-span-2 rounded-[28px] sm:rounded-[32px] p-4 sm:p-5 md:p-7 shadow-xl backdrop-blur"
                  style={{
                    border: `1px solid var(--color-border)`,
                    backgroundColor: 'var(--color-surface)',
                    boxShadow: `0 20px 25px -5px var(--shadow-color)`,
                  }}
                >
                  <div className="flex items-center justify-between">
                    <h3 className="font-display text-2xl" style={{ color: 'var(--color-text-primary)' }}>Signal garden</h3>
                    <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>4 charms inspected</span>
                  </div>
                  <div className="signal-garden-grid mt-4 sm:mt-6 grid gap-3 sm:gap-4 md:grid-cols-2">
                    {result.features.map((f, idx) => {
                      const isLastOdd = idx === result.features.length - 1 && result.features.length % 2 !== 0;
                      return (
                        <div
                          key={f.id}
                          className={cn(
                            "group relative overflow-hidden rounded-[20px] p-4 transition hover:shadow-lg",
                            isLastOdd && "md:col-span-2"
                          )}
                          style={{
                            border: `1px solid var(--color-border)`,
                            background: `linear-gradient(to bottom, var(--color-card-from), var(--color-card-to))`,
                          }}
                        >
                          <div
                            className="absolute right-0 top-0 h-20 w-20 translate-x-6 -translate-y-6 rounded-full blur-2xl"
                            style={{ background: `linear-gradient(to bottom right, var(--color-card-orb-from), var(--color-card-orb-to))` }}
                          />
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="text-[11px] uppercase tracking-widest" style={{ color: 'var(--color-text-muted)' }}>{f.label}</div>
                              <div className="mt-1 font-medium break-words" style={{ color: 'var(--color-text-primary)' }}>{f.value}</div>
                            </div>
                            <span
                              className="inline-flex size-8 shrink-0 items-center justify-center rounded-full ring-1"
                              style={{
                                backgroundColor: f.status === "good" ? 'var(--color-status-good-bg)' :
                                                 f.status === "warn" ? 'var(--color-status-warn-bg)' :
                                                 f.status === "bad" ? 'var(--color-status-bad-bg)' :
                                                 'var(--color-status-neutral-bg)',
                                color: f.status === "good" ? 'var(--color-status-good-text)' :
                                       f.status === "warn" ? 'var(--color-status-warn-text)' :
                                       f.status === "bad" ? 'var(--color-status-bad-text)' :
                                       'var(--color-status-neutral-text)',
                                '--tw-ring-color': f.status === "good" ? 'var(--color-status-good-ring)' :
                                                   f.status === "warn" ? 'var(--color-status-warn-ring)' :
                                                   f.status === "bad" ? 'var(--color-status-bad-ring)' :
                                                   'var(--color-status-neutral-ring)',
                              } as React.CSSProperties}
                            >
                              {f.status === "good" ? <Lock className="size-4" /> : f.status === "bad" ? <Unlock className="size-4" /> : <Info className="size-4" />}
                            </span>
                          </div>
                          <p className="mt-2 text-sm leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>{f.description}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Side */}
                <div className="space-y-4 sm:space-y-5">
                  <div
                    className="rounded-[28px] sm:rounded-[32px] p-4 sm:p-5 md:p-6 shadow-xl backdrop-blur"
                    style={{
                      border: `1px solid var(--color-border)`,
                      backgroundColor: 'var(--color-surface)',
                      boxShadow: `0 20px 25px -5px var(--shadow-color)`,
                    }}
                  >
                    <h3 className="font-display text-xl flex items-center gap-2" style={{ color: 'var(--color-text-primary)' }}>
                      <Heart className="size-4" style={{ color: 'var(--color-text-accent)' }} />
                      Recent scans
                    </h3>
                    <div className="mt-4 space-y-2.5">
                      {history.slice(0, 5).map((h) => (
                        <button
                          key={h.scannedAt + h.url}
                          onClick={() => setResult(h)}
                          className="history-item group flex w-full items-center justify-between rounded-2xl px-3 sm:px-3.5 py-2.5 sm:py-3 text-left transition hover:shadow-md"
                          style={{
                            border: `1px solid var(--color-border-light)`,
                            background: `linear-gradient(to right, var(--color-step-from), var(--color-step-to))`,
                          }}
                        >
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>{h.domain}</div>
                            <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{new Date(h.scannedAt).toLocaleTimeString()}</div>
                          </div>
                          <div className="history-meta flex items-center gap-2 shrink-0">
                            <VerdictPill verdict={h.verdict} />
                            <ChevronRight className="size-4" style={{ color: 'var(--color-text-muted)' }} />
                          </div>
                        </button>
                      ))}
                      {history.length === 0 && (
                        <div className="text-sm" style={{ color: 'var(--color-text-muted)' }}>No scans yet, love.</div>
                      )}
                    </div>
                  </div>

                  <div
                    className="rounded-[28px] sm:rounded-[32px] p-4 sm:p-5 md:p-6 shadow-xl"
                    style={{
                      background: `linear-gradient(to bottom right, var(--gradient-soft-start), var(--gradient-soft-end))`,
                      border: `1px solid var(--color-border)`,
                      boxShadow: `0 20px 25px -5px var(--shadow-color)`,
                    }}
                  >
                    <div className="font-display text-xl flex items-center gap-2" style={{ color: 'var(--color-text-primary)' }}>
                      <Sparkles className="size-4" style={{ color: 'var(--color-text-accent)' }} />
                      Stay safe, stay soft
                    </div>
                    <p className="mt-2 text-sm leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                      URL Sentinel keeps your browsing gentle and secure. Always double-check suspicious links before sharing personal details.
                    </p>
                    <button
                      onClick={() => setGuidesOpen(true)}
                      className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium transition"
                      style={{ color: 'var(--color-text-accent)' }}
                    >
                      Learn more <ArrowRight className="size-4" />
                    </button>
                  </div>
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Footer */}
        <footer className="mt-24 pt-8 text-center" style={{ borderTop: `1px solid var(--color-footer-border)` }}>
          <div className="font-display text-lg" style={{ color: 'var(--color-text-primary)' }}>URL Sentinel • aesthetic security</div>
          <div className="mt-1 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
            Made with ♡ • Not a substitute for professional threat intel • © 2026 fhamyla
          </div>
        </footer>
      </div>

      <GuidesModal open={guidesOpen} onClose={() => setGuidesOpen(false)} />
    </div>
  );
}
