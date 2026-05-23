# Deployment

The dashboard is a single Cloudflare Worker with a Static Assets binding. The Worker serves both the static SPA (from `./public/`) and the `/api/yf` proxy at `https://valuation.mackhaymond.co`.

## What runs where

| Path | Source | Handler |
|---|---|---|
| `/`, `/app.js`, `/styles.css`, `/sector_analysis*.csv`, `/weights.csv` | `./public/` | Cloudflare's edge asset server (no isolate hop) |
| `/api/yf?ticker=...` | `./src/worker.js` | The Worker's `fetch()` handler via `run_worker_first: ["/api/*"]` |

Single product. One `wrangler.jsonc`. The Worker only spins up an isolate when a `/api/*` path is hit; every other request is answered directly from the asset cache.

## Local development

```sh
npm install
npm run dev   # = wrangler dev, default port 8787
```

Smoke test:

```sh
curl -sI http://127.0.0.1:8787/
curl -s "http://127.0.0.1:8787/api/yf?ticker=AAPL" | jq '.quote.trailingPE'
curl -sI "http://127.0.0.1:8787/api/yf?ticker=AAPL" | grep -i x-cache  # HIT on warm calls
curl -s "http://127.0.0.1:8787/api/yf?ticker=NOTAREAL" | jq     # 404
curl -s "http://127.0.0.1:8787/api/yf?ticker=BAD%21CHAR" | jq    # 400
```

`wrangler dev` honors `wrangler.jsonc` exactly as production does — same `nodejs_compat`, same `define:` substitution for `__dirname`, same asset binding, same `run_worker_first` routing.

## Deploy

```sh
npm run deploy   # = wrangler deploy
```

The first deploy provisions:

- The `*.workers.dev` URL (currently `https://sector-relative-valuation.spyicydev.workers.dev`).
- The custom domain `valuation.mackhaymond.co` because `wrangler.jsonc` carries:

  ```jsonc
  "routes": [
    { "pattern": "valuation.mackhaymond.co", "custom_domain": true }
  ]
  ```

  `custom_domain: true` auto-creates the routing record on the same Cloudflare account the zone is on. No manual DNS step required.

Subsequent deploys diff the asset set and only re-upload what changed (typically 0 files unless the CSVs were refreshed by the data-refresh GHA).

## Watching logs

```sh
npm run tail   # = wrangler tail
```

Streams live request logs from the production Worker. Useful for the cutover window or when debugging a specific ticker.

## Configuration reference

| File | Purpose |
|---|---|
| `wrangler.jsonc` | Worker config: name, `main` entry, `compatibility_date`, `compatibility_flags = ["nodejs_compat"]`, `routes` (custom domain), `assets` binding, `define` (bundle-time `__dirname` shim), `vars`, `observability`. |
| `package.json` | `yahoo-finance2` dep, `wrangler` devDep, `npm run dev` / `npm run deploy` / `npm run tail` scripts. |
| `src/worker.js` | Single-file Worker. Handles `/api/yf` (GET + OPTIONS). |
| `public/` | Asset directory. Served by `env.ASSETS`. |
| `.github/workflows/deploy-worker.yml` | Optional CI: deploys on push to `main` if `CLOUDFLARE_API_TOKEN` is configured. |
| `.github/workflows/run-and-commit.yml` | Data refresh: runs the Python pipeline, moves CSVs into `./public/`, commits. The push triggers `deploy-worker.yml`. |

## `define:` rationale

`yahoo-finance2` transitively depends on `@deno/shim-deno`, which references `__dirname` at module init. Cloudflare Workers do not expose `__dirname` in ESM context. `wrangler.jsonc` substitutes the literal string `"/"` at bundle time:

```jsonc
"define": {
  "__dirname": "\"/\"",
  "__filename": "\"/index.js\""
}
```

Workers honors `define:`. (The prior Pages Function did not — that's why an earlier commit introduced a `globalThis.__dirname = "/"` runtime shim. The shim is unnecessary under Workers and was removed when `src/worker.js` was authored.)

## Custom domain provisioning

The current production domain `valuation.mackhaymond.co` was bound by the `wrangler deploy` that picked up the `routes` block in `wrangler.jsonc`. Cloudflare auto-handled the routing record because:

1. The OAuth/API token has `workers:write` and `zone:read`.
2. The zone `mackhaymond.co` is on the same Cloudflare account that owns the Worker.
3. `custom_domain: true` makes Cloudflare create the routing layer internally — no separate DNS write needed.

If you ever migrate to a different domain on a zone *not* on this account, the user must add a CNAME record manually (`<subdomain>` -> `<worker-name>.<account-slug>.workers.dev`) before the route resolves.

## CI deploy (optional)

`.github/workflows/deploy-worker.yml` deploys on push to `main` when `CLOUDFLARE_API_TOKEN` is configured as a repository secret. The token needs:

- **Account:Read** on the Cloudflare account
- **Workers Scripts:Edit** on the Cloudflare account
- **Zone:Read** on the `mackhaymond.co` zone (so `custom_domain: true` can validate the zone)

Create the token at https://dash.cloudflare.com/profile/api-tokens with the **Edit Cloudflare Workers** template, then add it to the GitHub repo as `CLOUDFLARE_API_TOKEN`. The workflow runs on every push to `main` and on the data-refresh GHA's commit.

If the secret is not present, the workflow is a no-op (skipped at the auth step) — push-to-deploy is optional, manual `npm run deploy` always works.

## Bundle size

| Asset | Raw | Gzipped (on the wire) |
|---|---:|---:|
| `plotly-cartesian.min.js` (CDN, v2.35.3) | 1 358 887 B | 446 KB |
| `papaparse.min.js` (CDN, v5.4.1) | 19 469 B | 7 KB |
| `simple-statistics.min.js` (CDN, v7.8.7) | 24 073 B | 9 KB |
| `public/app.js` | 41 850 B | ~12 KB |
| `public/styles.css` | 8 536 B | ~3 KB |
| `public/index.html` | 8 042 B | ~3 KB |
| **Initial JS+CSS+HTML total** | **~1.46 MB** | **~478 KB** |
| `src/worker.js` | 4 097 B | (server-side, never sent to browser) |

Cloudflare Pages had a per-deployment file size limit of 25 MB and a project file count limit of 20 000; Workers Static Assets enforces the same limits and is well within budget here (6 asset files, largest is 412 KB).

## Troubleshooting

- **`__dirname is not defined` during `wrangler dev` or `wrangler deploy`**: check that `wrangler.jsonc` still has the `define` block with `__dirname` and `__filename`. The block is the entire workaround for `yahoo-finance2`'s Deno shim.
- **`HTTP 530` on the custom domain**: the route binding failed or was never created. Run `wrangler deploy` again; the `routes` block in `wrangler.jsonc` should re-provision it. If the zone is moved to a different Cloudflare account, the auto-provisioning stops working and a manual CNAME is required.
- **`HTTP 502` from `/api/yf`**: Yahoo Finance throttled or returned a transient error. Retry after a few seconds; cached symbols serve from the edge for `CACHE_TTL_FUNDAMENTALS` seconds (default 300).
- **Stale CSV data**: the data-refresh GHA writes new CSVs into `public/` and pushes them. Pages-style auto-redeploy is replaced by `deploy-worker.yml` (if configured) or a manual `npm run deploy`.
