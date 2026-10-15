"""QA only: screenshot every page of the running app with real Streamlit."""

import json
import os
import sys

from playwright.sync_api import sync_playwright

PAGES = ["command", "corners", "geometry", "shape", "drift", "anatomy", "spc", "patterns", "remeasure", "height",
         "target", "compensation", "events", "passport", "data", "reports"]
BASE, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
log: list[str] = []


def wait_idle(page, timeout=180_000):
    try:
        page.wait_for_selector("[data-testid='stApp'][data-test-script-state='notRunning']", timeout=timeout)
    except Exception as e:  # noqa: BLE001
        log.append(f"wait timeout: {e!s:.120}")
    page.wait_for_timeout(2500)


def shoot(page, path, width=1440):
    h = page.evaluate("""() => {
        const m = document.querySelector('[data-testid="stMain"]') || document.querySelector('section.main');
        const inner = m ? m.scrollHeight : 0;
        return Math.max(document.body.scrollHeight, inner);
    }""")
    page.set_viewport_size({"width": width, "height": int(min(max(h, 900), 16000))})
    page.wait_for_timeout(1500)
    page.screenshot(path=path, full_page=True)
    page.set_viewport_size({"width": width, "height": 900})


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("console", lambda m: log.append(f"console {m.type}: {m.text[:300]}") if m.type == "error" else None)
    page.on("pageerror", lambda e: log.append(f"pageerror: {e!s:.300}"))
    page.goto(BASE + "/")
    wait_idle(page)
    shoot(page, f"{OUT}/00_start.png")
    for i, key in enumerate(PAGES, 1):
        try:
            page.goto(f"{BASE}/?demo" if key == "command" else f"{BASE}/{key}?demo")
            wait_idle(page)
            shoot(page, f"{OUT}/{i:02d}_{key}.png")
            main = page.locator("[data-testid='stMain']")
            tabs = main.get_by_role("tab")
            for j in range(1, tabs.count()):
                tabs.nth(j).click()
                page.wait_for_timeout(2000)
                shoot(page, f"{OUT}/{i:02d}_{key}_tab{j}.png")
            n_exc = page.locator("[data-testid='stException']").count()
            if n_exc:
                log.append(f"{key}: {n_exc} exception box(es): "
                           + page.locator("[data-testid='stException']").first.inner_text()[:800])
        except Exception as e:  # noqa: BLE001
            log.append(f"{key}: screenshot failed: {e!s:.300}")
    try:
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(f"{BASE}/?demo")
        wait_idle(page)
        shoot(page, f"{OUT}/90_command_phone.png", width=390)
    except Exception as e:  # noqa: BLE001
        log.append(f"phone: {e!s:.300}")
    browser.close()

with open(f"{OUT}/log.json", "w") as fh:
    json.dump(log, fh, indent=1)
print("\n".join(log))
