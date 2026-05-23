# Deployment

The dashboard is a static SPA backed by a single Cloudflare Pages Function that proxies Yahoo Finance. The repo root is the asset directory; the Function lives at `functions/api/yf.js` and auto-routes to `/api/yf`.

## Local development

```sh
npm install
npm run dev
```

Then visit `http://127.0.0.1:8788`. Wrangler compiles the Pages Function, serves the static assets at the repo root, and bundles `yahoo-finance2` with `nodejs_compat`. Hot reloads on file changes.

Verification commands:

```sh
# Static assets
curl -sI http://127.0.0.1:8788/ | head -1
curl -sI http://127.0.0.1:8788/app.js | head -1
curl -sI http://127.0.0.1:8788/sector_analysis.csv | head -1

# Worker proxy - first call ~250ms uncached, second call ~3ms with x-cache: HIT
curl -s "http://127.0.0.1:8788/api/yf?ticker=AAPL" | jq '.quote.trailingPE'
curl -sI "http://127.0.0.1:8788/api/yf?ticker=AAPL" | grep -i x-cache

# Error paths
curl -s "http://127.0.0.1:8788/api/yf?ticker=NOTAREAL"  | jq    # -> 404
curl -s "http://127.0.0.1:8788/api/yf?ticker=BAD!CHAR" | jq    # -> 400
curl -s "http://127.0.0.1:8788/api/yf"                 | jq    # -> 400
```

## First-time Cloudflare Pages setup (dashboard, once)

1. Sign in at https://dash.cloudflare.com.
2. **Workers & Pages** -> **Create application** -> **Pages** tab -> **Connect to Git**.
3. Authorize Cloudflare to read `mackhaymond/sector-relative-valuation` and select it.
4. Configure the build:
   - **Project name**: `sector-relative-valuation`
   - **Production branch**: `main`
   - **Framework preset**: None
   - **Build command**: leave empty
   - **Build output directory**: `/`
   - **Root directory**: leave default
   - Production environment variables:
     - `CACHE_TTL_FUNDAMENTALS=300`
     - `CACHE_TTL_HISTORICAL=3600`
5. Click **Save and Deploy**. The first build resolves `yahoo-finance2` and `wrangler` from `package.json`.
6. After the first deploy succeeds, attach the custom domain:
   - In the Pages project, **Custom domains** -> **Set up a custom domain**.
   - Enter `valuation.mackhaymond.co` and confirm. Cloudflare auto-creates the CNAME on the `mackhaymond.co` zone (already on Cloudflare).
7. Verify in production:

   ```sh
   curl -sI https://valuation.mackhaymond.co/ | head -3
   curl -s "https://valuation.mackhaymond.co/api/yf?ticker=AAPL" | jq '.quote.trailingPE'
   ```

## Subsequent deploys

Pages auto-redeploys on every push to `main`. The data-refresh GHA (`.github/workflows/run-and-commit.yml`) commits new CSVs to `main`, which triggers a Pages redeploy that picks up the fresh data. No manual deploy step.

If a hotfix is ever needed without committing:

```sh
npm run deploy
```

That runs `wrangler pages deploy .` against the configured project.

## Configuration reference

| File | Purpose |
|---|---|
| `wrangler.jsonc` | Cloudflare runtime config: `compatibility_date`, `nodejs_compat`, `pages_build_output_dir`, `define` for `__dirname` shim, env vars. |
| `package.json` | `yahoo-finance2` dependency, `wrangler` devDep, `npm run dev` / `npm run deploy` scripts. |
| `functions/api/yf.js` | Pages Function bound to `/api/yf`. GET-only. Edge-cached via `caches.default` keyed on ticker, TTL = `CACHE_TTL_FUNDAMENTALS`. |
| `index.html`, `app.js`, `styles.css` | Static SPA assets. Served from repo root. |
| `sector_analysis.csv`, `sector_analysis_full.csv`, `weights.csv` | Per-ticker composites and per-sector Ridge weights. Refreshed by the GHA workflow. |

## Why `define: { "__dirname": "\"/\"" }`

`yahoo-finance2` transitively pulls in `@deno/shim-deno`, which references `__dirname` at module init. Cloudflare Workers do not expose `__dirname` in ESM context even with `nodejs_compat`. The `define` entry in `wrangler.jsonc` substitutes the literal string `"/"` at bundle time, satisfying the reference without changing any other Node-compat behavior.

## Bundle size budget

| Asset | Raw | Gzipped (on the wire) |
|---|---:|---:|
| `plotly-cartesian.min.js` (CDN, v2.35.3) | 1.36 MB | 446 KB |
| `papaparse.min.js` (CDN, v5.4.1) | 19 KB | 7 KB |
| `simple-statistics.min.js` (CDN, v7.8.7) | 24 KB | 9 KB |
| `app.js` | ~42 KB | ~10 KB |
| `styles.css` | ~9 KB | ~3 KB |
| `index.html` | ~8 KB | ~3 KB |
| **Initial JS+CSS+HTML total** | **~1.46 MB** | **~478 KB** |

The three CSVs are fetched lazily by the tab that needs them and are gzipped by Cloudflare on the wire (`sector_analysis_full.csv` is the largest at ~412 KB raw -> ~75 KB gzipped).

## Troubleshooting

- **`__dirname is not defined` at `wrangler pages dev` startup**: the `define` block in `wrangler.jsonc` should be substituting `__dirname` at bundle time. If you see this error after editing `wrangler.jsonc`, confirm the `define` block is still present and that `compatibility_flags` still includes `nodejs_compat`.
- **`HTTP 502` from `/api/yf`**: Yahoo Finance throttled the upstream call or the symbol is not on Yahoo. Retry after a few seconds; cached symbols serve from edge for 5 minutes.
- **Custom domain stuck on "Pending"**: a stale CNAME may already exist in DNS. Delete it manually in **DNS** -> **Records** and re-trigger custom-domain provisioning.
