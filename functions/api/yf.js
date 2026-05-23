import YahooFinance from "yahoo-finance2";

/*
 * GET /api/yf?ticker=SYMBOL
 *
 * Proxies Yahoo Finance for the Individual Stock Analysis tab. Returns
 *   { ticker, fetchedAt, cached, quote, summary, history }
 * where `summary` is the union of the requested quoteSummary modules
 * and `history` is the trailing one-year daily OHLC array.
 *
 * Caching: Cloudflare's edge cache via caches.default, keyed on
 * `${origin}/__cache/yf/${ticker}`. Successful responses live for
 * CACHE_TTL_FUNDAMENTALS seconds (default 300 = 5 minutes). The
 * cache.put is wrapped in waitUntil so the response is not blocked
 * on the write.
 *
 * Failure modes: 400 for a malformed ticker, 404 for an unknown
 * symbol, 502 for upstream Yahoo errors. Error strings come from
 * upstream verbatim; no stack traces are leaked.
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

export async function onRequestGet({ request, env, waitUntil }) {
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

  if (typeof waitUntil === "function") {
    waitUntil(cache.put(cacheKey, cacheable));
  } else {
    await cache.put(cacheKey, cacheable);
  }

  return response;
}

export async function onRequestOptions() {
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
