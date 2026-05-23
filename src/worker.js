import YahooFinance from "yahoo-finance2";

/*
 * Cloudflare Worker entry point for sector-relative-valuation.
 *
 * Routing: `run_worker_first = ["/api/*"]` in wrangler.jsonc routes only
 * /api/* paths through this Worker. Every other path is served straight
 * from the [assets] binding (./public/) and never invokes the isolate.
 *
 * GET  /api/yf?ticker=SYMBOL  -> proxies Yahoo Finance, returns
 *   { ticker, fetchedAt, cached, quote, summary, history }.
 * OPTIONS /api/yf             -> CORS preflight (24h max-age).
 * Anything else under /api/*  -> 404 with a structured JSON error.
 *
 * Edge cache: caches.default keyed on ${origin}/__cache/yf/${ticker};
 * cache.put wrapped in ctx.waitUntil so the response isn't blocked on
 * the write. TTL = env.CACHE_TTL_FUNDAMENTALS seconds (default 300).
 *
 * Errors: 400 for missing/invalid ticker, 404 for unknown symbol,
 * 502 for upstream failures. Stack traces never leak.
 */

const TICKER_PATTERN = /^[A-Z0-9.\-^=]{1,12}$/;
const SUMMARY_MODULES = [
  "defaultKeyStatistics",
  "financialData",
  "assetProfile",
  "price",
  "summaryDetail",
];

const yahooFinance = new YahooFinance({
  suppressNotices: ["yahooSurvey", "ripHistorical"],
});

function jsonResponse(body, init = {}) {
  const headers = new Headers(init.headers || {});
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("Access-Control-Allow-Origin", "*");
  return new Response(JSON.stringify(body), { ...init, headers });
}

function errorResponse(status, message) {
  return jsonResponse({ error: message }, { status });
}

function buildCacheKey(url, ticker) {
  return new Request(`${url.origin}/__cache/yf/${ticker}`, { method: "GET" });
}

async function handleYf(request, env, ctx) {
  const url = new URL(request.url);
  const rawTicker = (url.searchParams.get("ticker") || "").trim().toUpperCase();
  if (!rawTicker) return errorResponse(400, "Missing ?ticker= query parameter");
  if (!TICKER_PATTERN.test(rawTicker)) return errorResponse(400, "Invalid ticker symbol");

  const cache = caches.default;
  const cacheKey = buildCacheKey(url, rawTicker);
  const cached = await cache.match(cacheKey);
  if (cached) {
    const headers = new Headers(cached.headers);
    headers.set("X-Cache", "HIT");
    return new Response(cached.body, { status: cached.status, headers });
  }

  const ttl = Number(env?.CACHE_TTL_FUNDAMENTALS) || 300;
  const now = new Date();
  const oneYearAgo = new Date(now);
  oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);

  let quote;
  let summary;
  let history;
  try {
    [quote, summary, history] = await Promise.all([
      yahooFinance.quote(rawTicker),
      yahooFinance.quoteSummary(rawTicker, { modules: SUMMARY_MODULES }),
      yahooFinance.historical(rawTicker, { period1: oneYearAgo, period2: now, interval: "1d" }),
    ]);
  } catch (err) {
    const msg = err && err.message ? String(err.message) : "Upstream Yahoo Finance error";
    const status = /not found|invalid|no data/i.test(msg) ? 404 : 502;
    return errorResponse(status, msg);
  }

  const body = {
    ticker: rawTicker,
    fetchedAt: now.toISOString(),
    cached: false,
    quote: quote || null,
    summary: summary || null,
    history: Array.isArray(history) ? history : [],
  };

  const response = jsonResponse(body, {
    status: 200,
    headers: { "Cache-Control": `public, s-maxage=${ttl}`, "X-Cache": "MISS" },
  });

  const cacheable = jsonResponse(
    { ...body, cached: true },
    {
      status: 200,
      headers: { "Cache-Control": `public, s-maxage=${ttl}`, "X-Cache": "HIT" },
    },
  );

  ctx.waitUntil(cache.put(cacheKey, cacheable));
  return response;
}

function handleOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Max-Age": "86400",
    },
  });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/api/yf") {
      if (request.method === "GET") return handleYf(request, env, ctx);
      if (request.method === "OPTIONS") return handleOptions();
      return new Response(null, {
        status: 405,
        headers: { Allow: "GET, OPTIONS" },
      });
    }
    return errorResponse(404, `Unknown API path: ${url.pathname}`);
  },
};
