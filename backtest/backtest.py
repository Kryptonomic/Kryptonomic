#!/usr/bin/env python3
"""Backtest the copy-trade plan in strategy/COPY_TRADE_PLAN.md against a wallet's real trades.

Pulls the wallet's recent transactions from Solana RPC, rebuilds every buy and sell
(SOL, wSOL, USDC proceeds and limit-order fills), downloads minute candles from
GeckoTerminal, then replays the plan minute by minute.

    python3 backtest/backtest.py                       # default wallet, last 1000 txs
    python3 backtest/backtest.py --wallet <addr> --slip 0.02 --mult 1.0
    python3 backtest/backtest.py --cache .cache        # reuse downloaded data

Set HELIUS_API_KEY in the environment to use Helius RPC (much faster, no rate-limit stalls).

Only the Python standard library is needed.
"""
import argparse, collections, json, os, time, urllib.error, urllib.request

WALLET = "ffQUA7xYw32XT9Kjgak5pekN3hSLAgUyhSdYkUMGdmd"
RPC = (f"https://mainnet.helius-rpc.com/?api-key={os.environ['HELIUS_API_KEY']}"
       if os.environ.get("HELIUS_API_KEY") else "https://api.mainnet.solana.com")
WSOL = "So11111111111111111111111111111111111111112"
USD_MINTS = {"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",   # USDC
             "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"}   # USDT
GECKO = "https://api.geckoterminal.com/api/v2/networks/solana"


# ---------------------------------------------------------------- fetching
def http_json(url, body=None, tries=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, json.dumps(body).encode() if body else None,
                {"Content-Type": "application/json", "Accept": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(min(30, 2 * (i + 1)))
        except Exception:
            time.sleep(min(30, 2 * (i + 1)))
    return None


def rpc(method, params):
    j = http_json(RPC, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    return j.get("result") if j else None


def cached(path, build):
    if path and os.path.exists(path):
        return json.load(open(path))
    data = build()
    if path:
        json.dump(data, open(path, "w"))
    return data


def fetch_txs(wallet, limit):
    sigs = rpc("getSignaturesForAddress", [wallet, {"limit": limit}]) or []
    out = {}
    for n, s in enumerate(sigs):
        out[s["signature"]] = rpc("getTransaction", [s["signature"], {
            "encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}])
        if n % 50 == 0:
            print(f"  fetched {n}/{len(sigs)} transactions", flush=True)
        time.sleep(0.02 if os.environ.get("HELIUS_API_KEY") else 0.15)
    return out


# ---------------------------------------------------------------- ledger
def build_ledger(txs, wallet):
    """One row per successful tx: wallet's SOL, USD-stable and token deltas."""
    rows = []
    for sig, t in txs.items():
        if not t or t["meta"]["err"]:
            continue
        m = t["meta"]
        keys = [k["pubkey"] for k in t["transaction"]["message"]["accountKeys"]]
        if wallet not in keys:
            continue
        i = keys.index(wallet)
        tok = collections.defaultdict(float)
        for sign, arr in ((-1, m.get("preTokenBalances", [])), (1, m.get("postTokenBalances", []))):
            for b in arr:
                if b.get("owner") == wallet:
                    tok[b["mint"]] += sign * float(b["uiTokenAmount"]["uiAmount"] or 0)
        sol = (m["postBalances"][i] - m["preBalances"][i]) / 1e9 + tok.pop(WSOL, 0)
        usd = sum(tok.pop(u, 0) for u in USD_MINTS)
        rows.append(dict(sig=sig, t=t["blockTime"], sol=sol, usd=usd,
                         tok={k: v for k, v in tok.items() if abs(v) > 1e-9},
                         signer=keys[0] == wallet,
                         logs=" ".join(l for l in m.get("logMessages") or [] if "Instruction:" in l)))
    rows.sort(key=lambda r: r["t"])
    return rows


def limit_fill(tx, row):
    """For a limit-order fill paid to the wallet: (mint, tokens sold) from the escrow account."""
    m = tx["meta"]
    for b in m["preTokenBalances"]:
        if b["mint"] in USD_MINTS or b["mint"] == WSOL:
            continue
        post = [q for q in m["postTokenBalances"] if q["accountIndex"] == b["accountIndex"]]
        after = float(post[0]["uiTokenAmount"]["uiAmount"] or 0) if post else 0
        sold = float(b["uiTokenAmount"]["uiAmount"] or 0) - after
        if sold > 0:
            return b["mint"], sold
    return None, 0


def sol_price_fn(rows):
    """SOL/USD over time, taken from the wallet's own USDC<->SOL conversions."""
    pts = [(r["t"], -r["usd"] / r["sol"]) for r in rows
           if not r["tok"] and abs(r["sol"]) > 0.05 and abs(r["usd"]) > 1 and r["sol"] * r["usd"] < 0]
    if not pts:
        pts = [(0, 150.0)]
    return lambda t: min(pts, key=lambda p: abs(p[0] - t))[1]


def trades_by_coin(rows, txs):
    px = sol_price_fn(rows)
    buys, sells = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        value = r["sol"] + r["usd"] / px(r["t"])            # in SOL
        if len(r["tok"]) == 1:
            mint, amt = next(iter(r["tok"].items()))
            if amt > 0 and value < 0 and r["signer"]:
                buys[mint].append((r["t"], -value, -value * px(r["t"]) / amt))  # (t, SOL, USD/token)
            elif amt < 0 and value > 0.0005:
                sells[mint].append((r["t"], -amt, value))                        # (t, tokens, SOL)
        elif not r["tok"] and "ExecuteLimitOrder" in r["logs"] and value > 0:
            mint, sold = limit_fill(txs[r["sig"]], r)
            if mint:
                sells[mint].append((r["t"], sold, value))
    return buys, sells, px


def fetch_candles(mint, first_t):
    j = http_json(f"{GECKO}/tokens/{mint}/pools?page=1")
    time.sleep(2.2)
    if not j or not j.get("data"):
        return None
    pools = sorted(j["data"], key=lambda d: float(d["attributes"].get("reserve_in_usd") or 0), reverse=True)
    pool = pools[0]["attributes"]["address"]
    c = http_json(f"{GECKO}/pools/{pool}/ohlcv/minute?aggregate=1&limit=1000&currency=usd"
                  f"&token={mint}&before_timestamp={first_t + 1000 * 60}")
    time.sleep(2.2)
    return (c or {}).get("data", {}).get("attributes", {}).get("ohlcv_list")


# ---------------------------------------------------------------- the plan
def simulate(coin, p):
    """Replay one coin: copy his buys and sells, plus 'take initials at 2x'."""
    c, sp = coin["candles"], coin["sol_usd"]
    qty = pos_cost = pos_cash = spent = cash = 0.0
    ntx = bi = si = 0
    his_hold = 0.0
    init_done = False
    for x in c:                                   # x = [t, open, high, low, close, vol], USD
        while bi < len(coin["buys"]) and coin["buys"][bi][0] < x[0] + 60:
            _, sol, price = coin["buys"][bi]; bi += 1
            his_hold += sol * sp / price
            if qty <= 1e-15:                      # new position (first buy or re-entry)
                pos_cost = pos_cash = 0.0; init_done = False
            elif init_done:                       # initials already out: don't add back in
                continue
            amt = min(sol * p["mult"], max(0.0, p["max_per_coin"] - pos_cost))
            if amt > 0:
                qty += amt * (1 - p["fee"]) * sp / (price * (1 + p["slip"]))
                pos_cost += amt; spent += amt; ntx += 1
        if qty > 1e-15 and not init_done:
            avg = pos_cost * sp / qty
            if x[2] >= avg * p["initials_at"]:
                need = max(0.0, pos_cost - pos_cash)
                val = qty * avg * p["initials_at"] / sp * (1 - p["fee"])
                f = min(1.0, need / val) if val > 0 else 0.0
                if f > 0:
                    got = val * f; cash += got; pos_cash += got; qty *= 1 - f; ntx += 1
                init_done = True
        while si < len(coin["sells"]) and coin["sells"][si][0] < x[0] + 60:
            _, tokens, sol = coin["sells"][si]; si += 1
            if his_hold <= 0:
                continue
            f = min(1.0, tokens / his_hold)
            his_hold = max(0.0, his_hold - tokens)
            if his_hold < 1e-9:
                f = 1.0
            if qty > 1e-15:
                got = qty * f * (sol / tokens) * (1 - p["slip"]) * (1 - p["fee"])
                cash += got; pos_cash += got; qty *= 1 - f; ntx += 1
    if qty > 1e-15:                               # still open: mark at last candle
        cash += qty * c[-1][4] / sp * (1 - p["fee"]); ntx += 1
    return cash - spent - p["prio"] * ntx, spent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wallet", default=WALLET)
    ap.add_argument("--limit", type=int, default=1000, help="how many recent txs to pull")
    ap.add_argument("--cache", default=".cache", help="folder for downloaded data ('' to disable)")
    ap.add_argument("--mult", type=float, default=1.0, help="your size as a multiple of his")
    ap.add_argument("--max-per-coin", type=float, default=999.0, help="SOL cap per coin")
    ap.add_argument("--initials-at", type=float, default=2.0, help="take initials at this multiple")
    ap.add_argument("--slip", type=float, default=0.02, help="how much worse than his price you fill")
    ap.add_argument("--fee", type=float, default=0.01, help="bot fee per swap")
    ap.add_argument("--prio", type=float, default=0.003, help="priority fee + tip per tx, SOL")
    a = ap.parse_args()
    if a.cache:
        os.makedirs(a.cache, exist_ok=True)
    path = lambda n: os.path.join(a.cache, f"{a.wallet[:8]}_{n}.json") if a.cache else None

    print("Loading transactions...")
    txs = cached(path("txs"), lambda: fetch_txs(a.wallet, a.limit))
    rows = build_ledger(txs, a.wallet)
    buys, sells, px = trades_by_coin(rows, txs)
    print(f"{len(rows)} txs, {len(buys)} coins bought. Loading price candles (rate-limited, ~5s/coin)...")
    candles = cached(path("candles"), lambda: {m: fetch_candles(m, b[0][0]) for m, b in buys.items()})

    coins = []
    for m, b in buys.items():
        c = candles.get(m)
        if not c:
            continue
        c = sorted(x for x in c if x[0] >= b[0][0] - 60)
        if len(c) < 3 or not 0.2 < c[0][4] / b[0][2] < 5:       # wrong pool / bad data
            continue
        coins.append(dict(mint=m, candles=c, buys=b, sells=sorted(sells.get(m, [])),
                          sol_usd=px(b[0][0]), t0=b[0][0]))

    p = dict(mult=a.mult, max_per_coin=a.max_per_coin, initials_at=a.initials_at,
             slip=a.slip, fee=a.fee, prio=a.prio)
    his = dict(p, mult=1.0, max_per_coin=1e9, initials_at=1e9, slip=0.0, fee=0.0, prio=0.0)
    res = [(co, simulate(co, p), simulate(co, his)) for co in coins]
    if not res:
        print("No coins with usable price data.")
        return
    mine = sum(r[1][0] for r in res); dep = sum(r[1][1] for r in res)
    theirs = sum(r[2][0] for r in res); tdep = sum(r[2][1] for r in res)
    wins = sum(r[1][0] > 0 for r in res)
    print(f"\nCoins with price data: {len(res)}  (period: {time.strftime('%Y-%m-%d', time.gmtime(min(c['t0'] for c in coins)))}"
          f" to {time.strftime('%Y-%m-%d', time.gmtime(max(c['t0'] for c in coins)))})")
    print(f"HIM : {theirs:+.2f} SOL on {tdep:.1f} SOL deployed")
    print(f"PLAN: {mine:+.2f} SOL on {dep:.1f} SOL deployed ({100 * mine / dep:+.1f}%), winners {wins}/{len(res)}")
    by_day = collections.defaultdict(float)
    for co, (pnl, _), _ in res:
        by_day[time.strftime("%m-%d", time.gmtime(co["t0"]))] += pnl
    print("PLAN by day:", "  ".join(f"{d}:{v:+.1f}" for d, v in sorted(by_day.items())))
    print("Best coins:", ", ".join(f"{co['mint'][:6]} {pnl:+.2f}" for co, (pnl, _), _ in
                                   sorted(res, key=lambda r: -r[1][0])[:5]))


if __name__ == "__main__":
    main()
