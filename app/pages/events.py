import json

import numpy as np
import pandas as pd
import streamlit as st

from app import data, state, ui
from rickhard import events as ev_mod
from rickhard import fixloop as fx
from rickhard import metrics
from rickhard.charts import comp_charts as ccmp
from rickhard.charts import fix_svg
from rickhard.charts import theme as T
from rickhard.stats import significance_label

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "events", "Every change to the line starts a new set-up. Log what is done and the loop answers: did "
                            "it work, what still costs FPY, and what does it take to reach the target?")
ui.require(a.first)
cfg = st.column_config

top = st.container()   # filled last: it needs the action log as edited below

# ---- Action log (the editor's input only changes when actions are added elsewhere) ---------------
records = state.events()
sig = json.dumps(records, sort_keys=True, default=str)
if st.session_state.get("_ev_sig") != sig or "_ev_base" not in st.session_state:
    st.session_state["_ev_base"] = ev_mod.normalize_events(records)
    st.session_state["_ev_ver"] = st.session_state.get("_ev_ver", 0) + 1

ui.section("Action log", "One row per action; add a row with the + button below the table.  \n"
                         "**Done** actions changed the line at their date and start a new set-up. **Planned** ones are "
                         "next steps: they appear in the brief and change nothing until you set them to Done.  \n"
                         "**Aims at** says what the action is meant to fix — the loop then checks that very thing.  \n"
                         "dX / dY / dYaw are optional: a correction you applied, compared with what was measured.")
edited = st.data_editor(
    st.session_state["_ev_base"], key=f"ev_editor_{st.session_state['_ev_ver']}", num_rows="dynamic",
    hide_index=True, width="stretch",
    column_order=["Date", "Status", "Title", "Target", "Scope", "Category", "Owner", "Notes", "dX", "dY", "dYaw"],
    column_config={
        "Date": cfg.DatetimeColumn("Date / time", format="YYYY-MM-DD HH:mm", required=True),
        "Status": cfg.SelectboxColumn("Status", options=ev_mod.STATUSES, default="Done", required=True),
        "Title": cfg.TextColumn("What was done / will be done", width="large", required=True),
        "Target": cfg.SelectboxColumn("Aims at", options=ev_mod.TARGETS, default=ev_mod.TARGET_GENERAL, required=True),
        "Scope": cfg.SelectboxColumn("Scope", options=ev_mod.SCOPES, default="All", required=True),
        "Category": cfg.SelectboxColumn("Category", options=ev_mod.CATEGORIES, default="Adjustment"),
        "Owner": cfg.TextColumn("Owner"),
        "Notes": cfg.TextColumn("Notes", width="medium"),
        "dX": cfg.NumberColumn("dX mm", format="%+.2f"), "dY": cfg.NumberColumn("dY mm", format="%+.2f"),
        "dYaw": cfg.NumberColumn("dYaw °", format="%+.3f"),
    })
new_records = ev_mod.events_to_records(edited)
state.set_events(new_records)
st.session_state["_ev_sig"] = json.dumps(new_records, sort_keys=True, default=str)

cp = fx.unexplained_shifts(data.change_points(c.key, a)["table"], new_records)
if len(cp):
    with st.expander(f"{len(cp)} process shift(s) in the data with no logged change within 2 days — what happened then?",
                     icon=":material/bolt:"):
        st.dataframe(cp[["Date", "Variable", "Before", "After", "Shift", "p"]], hide_index=True, width="stretch",
                     column_config={
                         "Date": cfg.DatetimeColumn(format="YYYY-MM-DD HH:mm"), "Before": cfg.NumberColumn(format="%+.3f"),
                         "After": cfg.NumberColumn(format="%+.3f"), "Shift": cfg.NumberColumn(format="%+.3f"),
                         "p": cfg.NumberColumn("p-value", format="%.1e")})
        st.caption("The placement moved on these dates and the log does not say why. A shift nobody can explain is "
                   "worth an investigation: something changed on the line.")

        def _add_detected():
            for _, row in cp.iterrows():
                var = str(row["Variable"])
                unit = "°" if "Yaw" in var else " mm"
                scope = "Type S" if var.startswith("Type S") else ("Type M" if var.startswith("Type M") else "All")
                target = "Yaw" if "Yaw" in var else ("Shift X" if "Shift X" in var else "Shift Y")
                state.add_event({"Date": pd.Timestamp(row["Date"]).isoformat(), "Scope": scope, "Category": "Other",
                                 "Title": f"Detected shift: {var} {row['Shift']:+.3f}{unit}",
                                 "Notes": "Found by change-point detection — replace with what was done.",
                                 "dX": None, "dY": None, "dYaw": None, "Status": "Done", "Target": target, "Owner": ""})

        st.button("Add them to the log as placeholders", icon=":material/add:", on_click=_add_detected)

# ---- The loop ----------------------------------------------------------------------------------
loop = data.fix_loop(c)
su, led, sc, cur, target = loop["setups"], loop["ledger"], loop["score"], loop["current"], loop["target"]

with top:
    s, p = loop["setup"], loop["proof"]
    if s is not None:
        gap = s["FPY"] - target
        since = "the start" if pd.isna(s["EventNo"]) and s["Setup"] == 1 else f"{pd.Timestamp(s['Start']):%b %d}"
        accent = T.STATUS["good"] if gap >= 0 else (T.STATUS["warning"] if gap > -5 else T.STATUS["critical"])
        cards = [{"label": f"FPY · {loop['story']['which']}", "value": ui.fmt_pct(s["FPY"]),
                  "delta": f"{gap:+.1f} pp vs {target:g}% target", "good": bool(gap >= 0),
                  "sub": f"{int(s['N']):,} modules since {since} · 95% CI {s['CI_lo']:.1f}–{s['CI_hi']:.1f}%",
                  "accent": accent},
                 {"label": "FPY is at least", "value": ui.fmt_pct(p["lower"]),
                  "sub": "cautious bound, 95% confidence — "
                         + {"proven": "**target demonstrated**", "below": "**below target**, not by chance",
                            "open": "target neither shown nor ruled out"}[p["state"]]}]
        if p["zero_fail_n"] is not None:
            cards.append({"label": "First-pass OK in a row", "value": f"{p['streak']:,}",
                          "sub": f"best {p['best_streak']:,} · **{p['zero_fail_n']:,}** in a row demonstrate {target:g}%"})
        lost = float(loop["budget"]["Loss"].sum()) if len(loop["budget"]) else 0.0
        big = loop["budget"].iloc[0] if len(loop["budget"]) else None
        cards.append({"label": "Lost today", "value": f"{lost:.1f}", "unit": "pp",
                      "sub": (f"largest: **{big['Mechanism']}** {big['Loss']:.1f} pp · " if big is not None else "")
                             + f"the target allows {max(0.0, 100 - target):.1f}"})
        cards.append({"label": "Changes that worked", "value": f"{sc['worked']} of {sc['judged']}" if sc["judged"] else "–",
                      "sub": " · ".join(f"{sc[k]} {lab}" for k, lab in (("no_effect", "no effect"),
                                                                       ("backfired", "backfired"),
                                                                       ("superseded", "superseded")) if sc[k])
                             or ("log a change to start the loop" if not sc["done"] else "of the changes judged so far")})
        cards.append({"label": "Open", "value": f"{sc['verifying'] + sc['planned']}",
                      "sub": f"{sc['verifying']} being verified · {sc['planned']} planned"})
        ui.kpi_cards(cards)
        if cur is not None and int(cur["Setup"]) != int(s["Setup"]):
            st.info(f"The line was changed again on **{pd.Timestamp(cur['Start']):%b %d, %H:%M}** "
                    f"(“{cur['StartedBy']}”): {ui.plural(int(cur['N']), 'module')} since — too few to judge. The "
                    "figures describe the set-up before it.", icon=":material/hourglass_top:")
    ui.section("FPY by set-up", "A set-up is the production between two logged changes — the line as it was run then. "
                                "The step is the FPY of the set-up, the band its 95% confidence interval; the numbered "
                                "marks are the rows of the action log.")
    drawing = fix_svg.setup_timeline_svg(su, metrics.daily_fpy(a.first), fx.changes(new_records, a.types), target)
    if drawing:
        ui.svg_box(drawing, "First-pass yield of every set-up over time", max_width=1500)
    st.page_link("app/pages/brief.py", label="The same story on one page for management — Executive Brief",
                 icon=":material/article:")

# ---- Did it work? ------------------------------------------------------------------------------
ui.section("Did it work?", "  \n".join(f"**{v}** — {t}" for v, t in fx.VERDICT_HELP.items()))
if led.empty:
    st.info("No action logged yet. Add one above — the first change you log splits the data into a before and an "
            "after.", icon=":material/event_note:")
else:
    chip = {"good": "background-color:rgba(12,163,12,.16);color:#0b7a0b;font-weight:600",
            "bad": "background-color:rgba(208,59,59,.16);color:#b02a2a;font-weight:600",
            "warn": "background-color:rgba(250,178,25,.20);font-weight:600",
            "info": "background-color:rgba(42,120,214,.14);font-weight:600",
            "neutral": "background-color:rgba(137,135,129,.22);font-weight:600"}
    show = led[["No", "Date", "Title", "Target", "Verdict"]].copy()
    show["FPY"] = [f"{b:.1f}% → {x:.1f}%" if np.isfinite(b) and np.isfinite(x) else "–"
                   for b, x in zip(led["FPY_before"], led["FPY_after"])]
    show["Delta"] = [f"{d:+.1f}" if np.isfinite(d) else "–" for d in led["FPY_delta"]]
    show["Modules"] = [f"{b} / {x}" for b, x in zip(led["N_before"], led["N_after"])]
    show["Detail"] = led["Detail"]
    styled = show.style.map(lambda v: chip.get(fx.VERDICT_LEVEL.get(str(v), ""), ""), subset=["Verdict"])
    st.dataframe(styled, hide_index=True, width="stretch", height=min(420, 40 + 36 * len(show)), column_config={
        "No": cfg.NumberColumn("#", format="%d", width="small"),
        "Date": cfg.DatetimeColumn("When", format="MMM DD, HH:mm"), "Title": cfg.TextColumn("Action", width="large"),
        "Target": cfg.TextColumn("Aims at"), "FPY": cfg.TextColumn("FPY before → after"),
        "Delta": cfg.TextColumn("Δ pp"), "Modules": cfg.TextColumn("Modules before / after"),
        "Detail": cfg.TextColumn("Evidence", width="large")})
    st.caption(f"Each action is judged on up to {fx.WINDOW} modules right before and right after it, in its scope, "
               f"never across another logged change. It takes {fx.MIN_N} modules after a change to judge it; with "
               f"{fx.SETTLED_N} and no significant difference it counts as no effect.")

    done = led[led["Status"] != "Planned"]
    if len(done):
        labels = [f"{int(n)} · {d:%b %d} · {t_} ({s_})" for n, d, t_, s_ in zip(done["No"], done["Date"], done["Title"],
                                                                            done["Scope"])]
        state.ensure_option("p_ev_sel", labels, labels[-1])
        var_opts = {"Shift X (fit)": ("Fit_Tx", "mm"), "Shift Y (fit)": ("Fit_Ty", "mm"), "Yaw (fit)": ("Fit_Rot", "°"),
                    "Worst corner |dev|": ("WorstAbs", "mm"), "Shape residual": ("ShapeMax", "mm")}
        ctl = st.columns([3, 1.4], vertical_alignment="bottom")
        with ctl[0]:
            sel = st.selectbox("Look closer at an action", labels, key="p_ev_sel")
        e = done.iloc[labels.index(sel)]
        default_var = {"Shift X": "Shift X (fit)", "Shift Y": "Shift Y (fit)", "Yaw": "Yaw (fit)",
                       "Shape": "Shape residual"}.get(e["Target"], "Worst corner |dev|")
        if st.session_state.get("_ev_var_for") != sel:      # a new action: show the variable it aims at
            st.session_state["_ev_var_for"] = sel
            st.session_state["p_ev_var"] = default_var
        with ctl[1]:
            state.ensure_option("p_ev_var", list(var_opts), default_var)
            var = st.selectbox("Variable to plot", list(var_opts), key="p_ev_var")
        v = fx.verify(a.first, new_records, int(e["No"]))
        msg = f"**{v['verdict']}** — {v['detail']}"
        {"good": st.success, "bad": st.error, "warn": st.warning, "info": st.info, "neutral": st.info}[v["level"]](msg)
        ba = v["compare"]
        if ba is not None and ba.get("enough"):
            tb = ba["table"].set_index("Column")
            cards = [{"label": "FPY before → after", "value": f"{ba['fpy_before']:.1f}% → {ba['fpy_after']:.1f}%",
                      "delta": f"{ba['fpy_after'] - ba['fpy_before']:+.1f} pp",
                      "good": bool(ba["fpy_after"] >= ba["fpy_before"]), "sub": significance_label(ba["fpy_p"])},
                     {"label": "Modules compared", "value": f"{ba['n_before']} / {ba['n_after']}",
                      "sub": "before / after, inside the neighbouring set-ups"}]
            for col, lab, unit in (("Fit_Tx", "Shift X", "mm"), ("Fit_Ty", "Shift Y", "mm"), ("Fit_Rot", "Yaw", "°")):
                if col in tb.index:
                    rr = tb.loc[col]
                    nd = 3 if unit == "°" else 2
                    cards.append({"label": f"{lab} level", "value": f"{rr['Before']:+.{nd}f} → {rr['After']:+.{nd}f}",
                                  "unit": unit, "sub": f"Δ {rr['Change']:+.{nd}f} · {significance_label(rr['p (mean)'])} · "
                                                       f"σ ×{rr['SD ratio']:.2f}"})
            ui.kpi_cards(cards)
            if v["delivered"]:
                st.caption(f"Planned against measured — {v['delivered']}.")
        col, unit = var_opts[var]
        types = ev_mod.scope_types(e["Scope"])
        ev_scope = fx.changes(new_records, types)
        f_scope = a.first[a.first["BatteryType"].isin(types)]
        left, right = st.columns([3, 2], gap="large")
        with left:
            ui.chart(ccmp.event_timeline(ev_mod.rolling_series(f_scope, "PASS", 40), ev_scope, "FPY (rolling 40)", "%",
                                         e["Date"], target=r.fpy_target), "ev_fpy")
            ui.chart(ccmp.event_timeline(ev_mod.rolling_series(f_scope, col, 25), ev_scope, var, unit, e["Date"]),
                     "ev_var")
        with right:
            if ba is not None and ba.get("enough") and col in ba["before"] and col in ba["after"]:
                ui.chart(ccmp.before_after_dist(ba["before"][col], ba["after"][col], var, unit,
                                                title=f"{var}: before vs after"), "ev_dist")
                tab = ba["table"][["Variable", "Before", "After", "Change", "p (mean)", "SD before", "SD after",
                                   "p (spread)"]]
                st.dataframe(tab, hide_index=True, width="stretch", height=420, column_config={
                    "Before": cfg.NumberColumn(format="%+.3f"), "After": cfg.NumberColumn(format="%+.3f"),
                    "Change": cfg.NumberColumn(format="%+.3f"), "p (mean)": cfg.NumberColumn(format="%.3f"),
                    "SD before": cfg.NumberColumn(format="%.3f"), "SD after": cfg.NumberColumn(format="%.3f"),
                    "p (spread)": cfg.NumberColumn(format="%.3f")})
                st.caption("p (mean): Welch t-test. p (spread): Brown–Forsythe test on the variances. Small p (< 0.05) "
                           "= a real change, not noise.")

# ---- What still costs FPY ------------------------------------------------------------------------
if len(su) and (su["N"] > 0).any():
    ui.section("What still costs FPY, and what it takes",
               "Both describe one set-up: the newest one with enough modules, unless you pick another.")
    with_modules = su[su["N"] > 0]
    opts = [f"Set-up {int(x['Setup'])} · since {pd.Timestamp(x['Start']):%b %d} · {int(x['N']):,} modules · "
            f"{x['FPY']:.1f}%" for _, x in with_modules.iterrows()]
    judged = fx.judged_setup(su)
    default = opts[list(with_modules["Setup"]).index(judged["Setup"])]
    if st.session_state.get("_ev_setup_for") != (c.key, sig):   # new data or log: back to the newest set-up
        st.session_state["_ev_setup_for"] = (c.key, sig)
        st.session_state["p_ev_setup"] = default
    state.ensure_option("p_ev_setup", opts, default)
    pick = st.selectbox("Set-up", opts, key="p_ev_setup")
    no = int(with_modules.iloc[opts.index(pick)]["Setup"])
    one = loop if no == int(judged["Setup"]) else data.fix_loop(c, setup_no=no)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Where the missing points go**")
        if len(one["budget"]):
            ui.svg_box(fix_svg.loss_budget_svg(one["budget"], target), "FPY points lost by failure mechanism", 760)
        else:
            st.success("No module failed its first measurement in this set-up.")
        st.caption(one["story"]["left"])
    with right:
        st.markdown(f"**What it takes to reach {target:g}%**")
        if len(one["bridge"]):
            ui.svg_box(fix_svg.bridge_svg(one["bridge"], target), "What it takes to reach the FPY target", 760)
        st.caption(one["story"]["path"] + " Each lever is simulated on this set-up's own modules, cheapest first; "
                                          "the split between levers depends on their order.")
    pr = one["proof"]
    if pr["n"] and np.isfinite(pr["lower"]):
        st.caption(f"**What the data can claim** — {pr['n']:,} modules, {pr['fails']} NOK: with 95% confidence the FPY "
                   f"of this set-up is between {pr['lower']:.1f}% and {pr['upper']:.1f}% (one-sided exact bounds). "
                   + one["story"]["proof"])
    lev = one["bridge"].iloc[1:] if len(one["bridge"]) > 1 else one["bridge"]
    with st.expander("The levers in numbers", icon=":material/table_rows:"):
        st.dataframe(lev[["Step", "Delta", "FPY", "How", "Owner"]], hide_index=True, width="stretch", column_config={
            "Step": cfg.TextColumn("Lever"), "Delta": cfg.NumberColumn("Gain (pp)", format="%+.1f"),
            "FPY": cfg.NumberColumn("FPY after (%)", format="%.1f"), "How": cfg.TextColumn("How", width="large")})
    with st.expander("Every set-up", icon=":material/table_rows:"):
        st.dataframe(su[["Label", "Start", "End", "StartedBy", "N", "NOK", "FPY", "CI_lo", "CI_hi", "PerDay"]],
                     hide_index=True, width="stretch", column_config={
            "Label": cfg.TextColumn("Set-up"), "Start": cfg.DatetimeColumn("From", format="YYYY-MM-DD HH:mm"),
            "End": cfg.DatetimeColumn("To", format="YYYY-MM-DD HH:mm"),
            "StartedBy": cfg.TextColumn("Started by", width="large"), "N": cfg.NumberColumn("Modules", format="%d"),
            "NOK": cfg.NumberColumn("NOK", format="%d"), "FPY": cfg.NumberColumn("FPY", format="%.1f%%"),
            "CI_lo": cfg.NumberColumn("CI low", format="%.1f%%"), "CI_hi": cfg.NumberColumn("CI high", format="%.1f%%"),
            "PerDay": cfg.NumberColumn("Modules / day", format="%.0f")})
