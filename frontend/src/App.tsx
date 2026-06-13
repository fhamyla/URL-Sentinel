import { useEffect, useMemo, useRef, useState } from "react";
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
} from "lucide-react";
import { cn } from "./utils/cn";

type Verdict = "safe" | "suspicious" | "phishing";

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



function VerdictPill({ verdict }: { verdict: Verdict }) {
  const cfg = {
    safe: {
      icon: ShieldCheck,
      label: "Safe & Sweet",
      cls: "from-emerald-200 to-teal-200 text-emerald-900",
    },
    suspicious: {
      icon: ShieldAlert,
      label: "Hmm… Suspicious",
      cls: "from-amber-200 to-peach-200 text-amber-900",
    },
    phishing: {
      icon: ShieldX,
      label: "Phishing Alert",
      cls: "from-rose-200 to-pink-300 text-rose-900",
    },
  }[verdict];
  const Icon = cfg.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full bg-gradient-to-r px-3 py-1 text-xs font-medium shadow-sm ring-1 ring-white/60",
        cfg.cls
      )}
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
    <div className="relative size-[150px]">
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
          stroke="rgba(255,255,255,0.5)"
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
        <div className="font-display text-5xl leading-none text-rose-950">{score}</div>
        <div className="mt-1 text-[10px] uppercase tracking-[0.18em] text-rose-900/70">
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
            className="fixed inset-0 z-50 bg-rose-950/20 backdrop-blur-sm"
            onClick={onClose}
          />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ type: "spring", duration: 0.5 }}
              className="relative max-h-[85vh] w-full max-w-3xl overflow-hidden rounded-[32px] border border-rose-200 bg-white/95 shadow-2xl shadow-rose-300/50 backdrop-blur-2xl"
            >
              <div className="relative max-h-[85vh] overflow-y-auto">
                <div className="sticky top-0 z-10 flex items-center justify-between border-b border-rose-100 bg-gradient-to-r from-pink-50 to-violet-50 px-8 py-6 backdrop-blur-xl">
                  <div className="flex items-center gap-3">
                    <div className="flex size-10 items-center justify-center rounded-2xl bg-gradient-to-br from-rose-400 to-fuchsia-500 shadow-md">
                      <BookOpen className="size-5 text-white" />
                    </div>
                    <div>
                      <h2 className="font-display text-2xl text-rose-950">How to use URL Sentinel</h2>
                      <p className="text-sm text-rose-700/70">Your gentle guide to safer browsing</p>
                    </div>
                  </div>
                  <button
                    onClick={onClose}
                    className="flex size-9 items-center justify-center rounded-full border border-rose-200 bg-white text-rose-600 transition hover:bg-rose-50"
                  >
                    <X className="size-4" />
                  </button>
                </div>

                <div className="px-8 py-8 space-y-8">
                  {/* Quick start */}
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Sparkles className="size-4 text-pink-500" />
                      <h3 className="font-display text-xl text-rose-950">Quick Start</h3>
                    </div>
                    <div className="space-y-3">
                      {[
                        { step: "1", title: "Paste your link", desc: "Copy any suspicious URL and paste it into the input field. We support http, https, and even bare domains." },
                        { step: "2", title: "Tap Analyze", desc: "Our gentle scanner checks 40+ signals including SSL, domain age, keywords, and patterns." },
                        { step: "3", title: "Read your aura", desc: "Get an instant verdict with risk score, confidence, and a pretty breakdown of what we found." },
                      ].map((item) => (
                        <div key={item.step} className="flex gap-4 rounded-2xl border border-rose-100 bg-gradient-to-r from-white to-rose-50/50 p-4">
                          <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-rose-400 to-fuchsia-500 text-sm font-medium text-white">
                            {item.step}
                          </div>
                          <div>
                            <div className="font-medium text-rose-950">{item.title}</div>
                            <div className="text-sm text-rose-800/70">{item.desc}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Understanding results */}
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Heart className="size-4 text-pink-500" />
                      <h3 className="font-display text-xl text-rose-950">Understanding results</h3>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-3">
                      {[
                        { verdict: "safe" as Verdict, title: "Safe & Sweet", desc: "Low risk. Standard security observed. Browse with peace.", color: "emerald" },
                        { verdict: "suspicious" as Verdict, title: "Hmm… Suspicious", desc: "Mixed signals. Be cautious with passwords or downloads.", color: "amber" },
                        { verdict: "phishing" as Verdict, title: "Phishing Alert", desc: "High risk. Likely trying to steal info. Avoid!", color: "rose" },
                      ].map((item) => (
                        <div key={item.verdict} className="rounded-2xl border border-rose-100 bg-white p-4">
                          <VerdictPill verdict={item.verdict} />
                          <div className="mt-3 text-sm font-medium text-rose-950">{item.title}</div>
                          <div className="mt-1 text-xs text-rose-700/70">{item.desc}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Signal garden */}
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Flower2 className="size-4 text-pink-500" />
                      <h3 className="font-display text-xl text-rose-950">Signal garden explained</h3>
                    </div>
                    <div className="rounded-2xl border border-rose-100 bg-gradient-to-br from-pink-50/50 to-violet-50/50 p-5">
                      <div className="grid gap-4 sm:grid-cols-2">
                        {[
                          { name: "SSL Charm", desc: "Checks for HTTPS encryption. No lock = no love." },
                          { name: "Domain Bloom", desc: "New domains are riskier. We check registration age." },
                          { name: "Whisper Words", desc: "Phishing loves urgent words like 'verify' or 'secure'." },
                          { name: "TLD Petal", desc: "Some domain endings (.tk, .xyz) are abused more." },
                        ].map((s) => (
                          <div key={s.name} className="flex gap-3">
                            <div className="mt-0.5 size-1.5 rounded-full bg-pink-400" />
                            <div>
                              <div className="text-sm font-medium text-rose-950">{s.name}</div>
                              <div className="text-xs text-rose-700/70">{s.desc}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Tips */}
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <HelpCircle className="size-4 text-pink-500" />
                      <h3 className="font-display text-xl text-rose-950">Pro tips</h3>
                    </div>
                    <div className="space-y-2.5">
                      {[
                        "Always check the domain carefully — phishers use lookalikes like 'paypaI' (capital i) instead of 'paypal'",
                        "Legit sites never ask for passwords via email links",
                        "When in doubt, type the URL directly instead of clicking",
                        "Look for the padlock, but remember: HTTPS alone doesn't guarantee safety",
                      ].map((tip, i) => (
                        <div key={i} className="flex gap-3 rounded-xl bg-white px-4 py-3 border border-rose-100">
                          <ArrowRight className="mt-0.5 size-4 shrink-0 text-pink-500" />
                          <span className="text-sm text-rose-800/80">{tip}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Footer */}
                  <div className="rounded-2xl bg-gradient-to-r from-rose-100 to-fuchsia-100 p-5 text-center border border-rose-200">
                    <p className="text-sm text-rose-900">
                      Stay curious, stay safe. URL Sentinel is your companion, not a replacement for critical thinking. 💖
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

  const fetchHistory = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/history");
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      } else {
        throw new Error();
      }
    } catch {
      const saved = localStorage.getItem("phish-history-girly");
      if (saved) {
        try {
          setHistory(JSON.parse(saved));
        } catch {}
      }
    }
  };

  const fetchModelInfo = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/model-info");
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
      localStorage.setItem("phish-history-girly", JSON.stringify(history.slice(0, 10)));
    }
  }, [history]);

  const handleAnalyze = async (u = url) => {
    if (!u.trim()) {
      inputRef.current?.focus();
      return;
    }

    const isValidUrl = (str: string) => {
      const pattern = /^(https?:\/\/)?((\d{1,3}\.){3}\d{1,3}|([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})(:\d+)?([/?].*)?$/;
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
      const res = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
    if (!result) return "from-pink-100 via-rose-50 to-violet-100";
    return result.verdict === "safe"
      ? "from-emerald-50 via-pink-50 to-violet-100"
      : result.verdict === "suspicious"
      ? "from-amber-50 via-pink-50 to-orange-100"
      : "from-rose-100 via-pink-100 to-fuchsia-100";
  }, [result]);

  return (
    <div className="min-h-screen bg-[#fffafc] text-rose-950 antialiased">
      {/* Soft background */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className={cn("absolute inset-0 bg-gradient-to-b", pastelBg)} />
        <img
          src="/images/girly-bg.jpg"
          alt=""
          className="absolute inset-0 h-full w-full object-cover opacity-[0.35] mix-blend-soft-light"
        />
        <div className="absolute inset-0 bg-[radial-gradient(60%_60%_at_50%_0%,rgba(255,255,255,0.8),transparent_70%)]" />
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-rose-300/60 to-transparent" />
      </div>

      <div className="mx-auto max-w-[1100px] px-6 py-8 lg:px-8 lg:py-12">
        {/* Header */}
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 rounded-[20px] bg-pink-300 blur-2xl opacity-40" />
              <div className="relative flex size-11 items-center justify-center rounded-[20px] bg-gradient-to-br from-rose-400 to-fuchsia-500 shadow-lg shadow-rose-300/50 ring-1 ring-white/70">
                <Flower2 className="size-5 text-white" />
              </div>
            </div>
            <div>
              <div className="font-display text-[22px] leading-none tracking-tight">URL Sentinel</div>
              <div className="flex items-center gap-1 text-[11px] uppercase tracking-widest text-rose-800/70">
                <Heart className="size-3" />
                aesthetic security
              </div>
            </div>
          </div>
          <nav className="hidden items-center gap-1.5 md:flex">
            <button
              onClick={() => setGuidesOpen(true)}
              className="rounded-full px-3 py-1.5 text-sm text-rose-900/70 hover:bg-white/60 hover:text-rose-950 transition"
            >
              Guides
            </button>
          </nav>
        </header>

        {/* Hero */}
        <section className="mx-auto mt-16 max-w-3xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="inline-flex items-center gap-2 rounded-full border border-rose-200 bg-white/70 px-3 py-1 text-xs text-rose-800 shadow-sm backdrop-blur">
              <Wand2 className="size-3.5 text-pink-500" />
              Soft security for the modern web
            </div>
            <h1 className="font-display mt-6 text-[48px] leading-[1.05] tracking-[-0.02em] sm:text-[64px]">
              Check links with
              <span className="relative mx-3 inline-block">
                <span className="relative z-10 bg-gradient-to-r from-rose-600 via-fuchsia-600 to-violet-600 bg-clip-text text-transparent italic px-2 pb-3 pt-1 -mx-2">
                  gentle
                </span>
                <span className="absolute inset-x-0 bottom-1 h-[10px] bg-pink-200/70 -rotate-1" />
              </span>
              intuition
            </h1>
            <p className="mx-auto mt-4 max-w-xl text-[15px] leading-relaxed text-rose-900/70">
              Paste any URL. We read its aura — SSL charms, domain bloom, whisper words — and give you a pretty, honest verdict.
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
              <div className="absolute -inset-1 rounded-[28px] bg-gradient-to-r from-pink-300 via-fuchsia-300 to-violet-300 opacity-60 blur-2xl transition group-focus-within:opacity-100" />
              <div className="relative flex items-center gap-2 rounded-[24px] border border-rose-200 bg-white/80 p-2.5 shadow-xl shadow-rose-200/50 backdrop-blur-xl">
                <div className="pl-3 pr-1 text-rose-400">
                  <Link2 className="size-5" />
                </div>
                <input
                  ref={inputRef}
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
                  placeholder="Paste a link, darling… e.g. https://secure-login-paypal.com"
                  className="h-14 w-full bg-transparent font-body text-[15px] placeholder-rose-300 outline-none"
                />
                <button
                  onClick={() => handleAnalyze()}
                  disabled={loading}
                  className="inline-flex h-12 items-center gap-2 rounded-[16px] bg-gradient-to-r from-rose-500 to-fuchsia-500 px-5 text-sm font-medium text-white shadow-md shadow-rose-300/50 transition hover:from-rose-600 hover:to-fuchsia-600 active:scale-[0.98] disabled:opacity-60"
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

            <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs text-rose-700/70">
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
                  className="rounded-full border border-rose-200 bg-white/70 px-3 py-1 hover:bg-white transition"
                >
                  {getDomain(s)}
                </button>
              ))}
            </div>
            {error && (
              <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50/80 px-4 py-2.5 text-xs text-rose-800 shadow-sm backdrop-blur inline-block">
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
                <div key={i} className="h-36 animate-pulse rounded-[28px] border border-rose-200 bg-white/60" />
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
              <div className="relative overflow-hidden rounded-[32px] border border-rose-200 bg-white/80 p-[1px] shadow-2xl shadow-rose-200/60 backdrop-blur-xl">
                <div className="absolute inset-0 bg-gradient-to-br from-pink-100/60 via-transparent to-violet-100/60" />
                <div className="relative grid gap-10 p-8 lg:grid-cols-[auto_1fr] lg:p-12">
                  <div className="flex flex-col items-center gap-4">
                    <RiskRing score={result.riskScore} />
                    <VerdictPill verdict={result.verdict} />
                    <div className="text-center">
                      <div className="font-display text-lg text-rose-950">{result.aura}</div>
                      <div className="text-xs text-rose-700/70">
                        Confidence {result.confidence}% • <Clock className="inline size-3" /> {new Date(result.scannedAt).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>

                  <div className="min-w-0">
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="font-display text-[28px] leading-tight text-rose-950">
                          {result.domain}
                        </div>
                        <div className="mt-1 truncate text-sm text-rose-800/70">{result.url}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(result.url);
                            setCopied(true);
                            setTimeout(() => setCopied(false), 1500);
                          }}
                          className="inline-flex items-center gap-1.5 rounded-full border border-rose-200 bg-white px-3 py-1.5 text-xs text-rose-900 hover:bg-rose-50 transition"
                        >
                          {copied ? <Check className="size-3.5 text-emerald-600" /> : <Copy className="size-3.5" />}
                          {copied ? "Copied" : "Copy"}
                        </button>
                        <a
                          href={result.url.startsWith("http") ? result.url : `http://${result.url}`}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1.5 rounded-full border border-rose-200 bg-white px-3 py-1.5 text-xs text-rose-900 hover:bg-rose-50 transition"
                        >
                          Open <ExternalLink className="size-3.5" />
                        </a>
                      </div>
                    </div>

                    <p className="mt-5 max-w-2xl font-body text-[15px] leading-relaxed text-rose-900/80">
                      {result.summary}
                    </p>

                    <div className="mt-6 flex flex-wrap gap-2">
                      {[
                        "Soft scan",
                        modelInfo ? `Model: ${modelInfo.model_name}` : "Heuristic + ML",
                        modelInfo && modelInfo.metrics[modelInfo.model_name]
                          ? `F1: ${(modelInfo.metrics[modelInfo.model_name].f1 * 100).toFixed(1)}%`
                          : null,
                        "Privacy first"
                      ].filter(Boolean).map((tag) => (
                        <span key={tag as string} className="inline-flex items-center gap-1 rounded-full bg-rose-100 px-3 py-1 text-xs text-rose-900 ring-1 ring-rose-200">
                          <Feather className="size-3" />
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Features */}
              <div className="mt-6 grid gap-5 lg:grid-cols-3">
                <div className="lg:col-span-2 rounded-[32px] border border-rose-200 bg-white/80 p-7 shadow-xl shadow-rose-200/50 backdrop-blur">
                  <div className="flex items-center justify-between">
                    <h3 className="font-display text-2xl text-rose-950">Signal garden</h3>
                    <span className="text-xs text-rose-700/70">6 charms inspected</span>
                  </div>
                  <div className="mt-6 grid gap-4 sm:grid-cols-2">
                    {result.features.map((f) => (
                      <div
                        key={f.id}
                        className="group relative overflow-hidden rounded-[20px] border border-rose-200 bg-gradient-to-b from-white to-rose-50/60 p-4 transition hover:shadow-lg hover:shadow-rose-200/60"
                      >
                        <div className="absolute right-0 top-0 h-20 w-20 translate-x-6 -translate-y-6 rounded-full bg-gradient-to-br from-pink-200/40 to-violet-200/40 blur-2xl" />
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-[11px] uppercase tracking-widest text-rose-700/60">{f.label}</div>
                            <div className="mt-1 font-medium text-rose-950">{f.value}</div>
                          </div>
                          <span
                            className={cn(
                              "inline-flex size-8 items-center justify-center rounded-full ring-1",
                              f.status === "good" && "bg-emerald-100 text-emerald-700 ring-emerald-200",
                              f.status === "warn" && "bg-amber-100 text-amber-700 ring-amber-200",
                              f.status === "bad" && "bg-rose-100 text-rose-700 ring-rose-200",
                              f.status === "neutral" && "bg-violet-100 text-violet-700 ring-violet-200"
                            )}
                          >
                            {f.status === "good" ? <Lock className="size-4" /> : f.status === "bad" ? <Unlock className="size-4" /> : <Info className="size-4" />}
                          </span>
                        </div>
                        <p className="mt-2 text-sm leading-relaxed text-rose-900/70">{f.description}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Side */}
                <div className="space-y-5">
                  <div className="rounded-[32px] border border-rose-200 bg-white/80 p-6 shadow-xl shadow-rose-200/50 backdrop-blur">
                    <h3 className="font-display text-xl text-rose-950 flex items-center gap-2">
                      <Heart className="size-4 text-pink-500" />
                      Recent scans
                    </h3>
                    <div className="mt-4 space-y-2.5">
                      {history.slice(0, 5).map((h) => (
                        <button
                          key={h.scannedAt + h.url}
                          onClick={() => setResult(h)}
                          className="group flex w-full items-center justify-between rounded-2xl border border-rose-100 bg-gradient-to-r from-white to-rose-50/70 px-3.5 py-3 text-left transition hover:border-rose-200 hover:shadow-md"
                        >
                          <div className="min-w-0">
                            <div className="truncate text-sm font-medium text-rose-950">{h.domain}</div>
                            <div className="text-xs text-rose-700/60">{new Date(h.scannedAt).toLocaleTimeString()}</div>
                          </div>
                          <div className="flex items-center gap-2">
                            <VerdictPill verdict={h.verdict} />
                            <ChevronRight className="size-4 text-rose-300 group-hover:text-rose-500" />
                          </div>
                        </button>
                      ))}
                      {history.length === 0 && (
                        <div className="text-sm text-rose-700/60">No scans yet, love.</div>
                      )}
                    </div>
                  </div>

                  <div className="rounded-[32px] border border-rose-200 bg-gradient-to-br from-pink-50 to-violet-50 p-6 shadow-xl shadow-rose-200/50">
                    <div className="font-display text-xl text-rose-950 flex items-center gap-2">
                      <Sparkles className="size-4 text-pink-500" />
                      Stay safe, stay soft
                    </div>
                    <p className="mt-2 text-sm leading-relaxed text-rose-900/70">
                      URL Sentinel keeps your browsing gentle and secure. Always double-check suspicious links before sharing personal details.
                    </p>
                    <button
                      onClick={() => setGuidesOpen(true)}
                      className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-rose-700 hover:text-rose-900 transition"
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
        <footer className="mt-24 border-t border-rose-200 pt-8 text-center">
          <div className="font-display text-lg text-rose-950">URL Sentinel • aesthetic security</div>
          <div className="mt-1 text-xs text-rose-700/70">
            Made with ♡ • Not a substitute for professional threat intel • © 2026
          </div>
        </footer>
      </div>

      <GuidesModal open={guidesOpen} onClose={() => setGuidesOpen(false)} />
    </div>
  );
}
