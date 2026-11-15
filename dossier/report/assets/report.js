/* The page script of the report. No library, no network except the review API of the dashboard (window.DOSSIER_API). */
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("dossier-data").textContent);
  var G = D.graph, NODES = G.nodes;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var CARDS = {};
  D.cards.forEach(function (c) { CARDS[c] = 1; });
  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* no storage */ } }
  };
  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "cls") e.className = attrs[k];
      else if (k === "text") e.textContent = attrs[k];
      else if (attrs[k] !== null && attrs[k] !== undefined && attrs[k] !== false) e.setAttribute(k, attrs[k] === true ? "" : attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function icon(name) {
    var s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("class", "i"); s.setAttribute("aria-hidden", "true");
    var u = document.createElementNS("http://www.w3.org/2000/svg", "use");
    u.setAttribute("href", "#i:" + name); s.appendChild(u); return s;
  }
  function toast(msg) {
    var t = $("#toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(toast.t); toast.t = setTimeout(function () { t.hidden = true; }, 3500);
  }

  /* ---------------------------------------------------------------- theme and navigation */
  var root = document.documentElement;
  function setTheme(t) { if (t) root.setAttribute("data-theme", t); else root.removeAttribute("data-theme"); }
  var saved = store.get("dossier-theme"); if (saved) setTheme(saved);
  $("#theme").addEventListener("click", function () {
    var dark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    var next = dark ? "light" : "dark"; setTheme(next); store.set("dossier-theme", next);
  });
  function navOpen(on) { document.body.classList.toggle("nav-open", on); $("#menu").setAttribute("aria-expanded", on ? "true" : "false"); }
  $("#menu").addEventListener("click", function () { navOpen(true); });
  $("#sidex").addEventListener("click", function () { navOpen(false); });
  $$(".navl a").forEach(function (a) { a.addEventListener("click", function () { navOpen(false); }); });
  var secs = $$("section.sec"), links = {};
  $$(".navl a").forEach(function (a) { links[a.getAttribute("data-nav")] = a; });
  function spy() {
    var cur = secs[0], y = window.innerHeight * 0.3;
    secs.forEach(function (s) { if (s.getBoundingClientRect().top <= y) cur = s; });
    Object.keys(links).forEach(function (k) {
      var on = cur && cur.getAttribute("data-sec") === k;
      links[k].classList.toggle("on", on);
      if (on) links[k].setAttribute("aria-current", "true"); else links[k].removeAttribute("aria-current");
    });
  }
  var spyT; window.addEventListener("scroll", function () { clearTimeout(spyT); spyT = setTimeout(spy, 40); }, { passive: true });

  /* ---------------------------------------------------------------- following a link to a record */
  function openAncestors(e) {
    for (var n = e; n && n !== document.body; n = n.parentElement) if (n.tagName === "DETAILS" && !n.open) n.open = true;
  }
  var lastTarget = null;
  function goTo(id, scroll) {
    var e = document.getElementById(id);
    if (!e) return false;
    if (lastTarget) lastTarget.classList.remove("is-target");
    openAncestors(e);
    if (e.tagName === "DETAILS") e.open = true;
    e.classList.add("is-target"); lastTarget = e;
    if (scroll !== false) e.scrollIntoView({ block: "start" });
    var sm = e.closest("summary"); if (!sm && e.focus) { e.setAttribute("tabindex", "-1"); }
    setTimeout(function () { if (lastTarget === e) e.classList.remove("is-target"); }, 4000);
    return true;
  }
  function onHash() {
    var id = decodeURIComponent(location.hash.slice(1));
    if (id) goTo(id); spy();
  }
  window.addEventListener("hashchange", onHash);
  function navigateTo(id) {
    if (location.hash === "#" + id) { goTo(id); return; }
    location.hash = id;
  }
  document.addEventListener("click", function (ev) {
    var a = ev.target.closest && ev.target.closest("a[href^='#']");
    if (a && !ev.defaultPrevented && !a.closest(".why-panel")) {
      var id = decodeURIComponent(a.getAttribute("href").slice(1));
      if (id && document.getElementById(id)) { ev.preventDefault(); navigateTo(id); hidePeek(); }
      return;
    }
    var g = ev.target.closest && ev.target.closest("svg.dz-diagram [data-ref]");
    if (g) { var r = g.getAttribute("data-ref"); if (CARDS[r]) navigateTo(r); }
  });
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" || ev.key === " ") {
      var g = ev.target.closest && ev.target.closest("svg.dz-diagram [data-ref]");
      if (g && ev.target === g) { ev.preventDefault(); var r = g.getAttribute("data-ref"); if (CARDS[r]) navigateTo(r); }
    }
    if (ev.key === "Escape") { if (!$("#why-panel").hidden) closeWhy(); hidePeek(); navOpen(false); }
  });
  /* A card opened by hand does not change the address. A click on the id of a card puts it in the address. */

  /* ---------------------------------------------------------------- preview of a record on hover or focus */
  var peek = $("#peek"), peekT;
  function hidePeek() { clearTimeout(peekT); peek.hidden = true; }
  function showPeek(target, id) {
    var n = NODES[id]; if (!n) return;
    peek.innerHTML = "";
    peek.appendChild(el("span", { cls: "pk-k", text: (D.kinds[n.kind] || n.kind) + (n.status ? " - " + String(n.status).replace(/_/g, " ") : "") }));
    peek.appendChild(el("b", { text: n.title || id }));
    if (n.summary) peek.appendChild(el("div", { cls: "pk-s", text: n.summary }));
    peek.hidden = false;
    var r = target.getBoundingClientRect(), w = peek.offsetWidth, h = peek.offsetHeight;
    var x = Math.min(Math.max(8, r.left), window.innerWidth - w - 8);
    var y = r.bottom + 6; if (y + h > window.innerHeight - 8) y = Math.max(8, r.top - h - 6);
    peek.style.left = x + "px"; peek.style.top = y + "px";
  }
  function peekId(t) {
    if (!t || !t.closest) return null;
    var a = t.closest("a.ref, a.self, svg.dz-diagram [data-ref]");
    if (!a || a.closest(".why-panel") && false) return null;
    var id = a.getAttribute("data-ref") || (a.getAttribute("href") || "").slice(1);
    return NODES[id] ? { a: a, id: id } : null;
  }
  document.addEventListener("mouseover", function (ev) {
    var p = peekId(ev.target); clearTimeout(peekT);
    if (p) peekT = setTimeout(function () { showPeek(p.a, p.id); }, 220); else if (!peek.hidden) hidePeek();
  });
  document.addEventListener("focusin", function (ev) { var p = peekId(ev.target); if (p) showPeek(p.a, p.id); else hidePeek(); });
  document.addEventListener("focusout", hidePeek);
  window.addEventListener("scroll", hidePeek, { passive: true });

  /* ---------------------------------------------------------------- open all, close all, print */
  document.addEventListener("click", function (ev) {
    var b = ev.target.closest && ev.target.closest("[data-xall]");
    if (!b) return;
    var open = b.getAttribute("data-xall") === "open";
    $$("details", b.closest("section")).forEach(function (d) { if (!$(".filters", d)) d.open = open; });
  });
  var printed = [];
  window.addEventListener("beforeprint", function () {
    printed = $$("details").filter(function (d) { return !d.open; });
    printed.forEach(function (d) { d.open = true; });
  });
  window.addEventListener("afterprint", function () { printed.forEach(function (d) { d.open = false; }); printed = []; });

  /* ---------------------------------------------------------------- the Why panel: rests on, affects */
  var edgesFrom = {}, edgesTo = {};
  G.edges.forEach(function (e) {
    (edgesFrom[e.from] = edgesFrom[e.from] || []).push(e);
    (edgesTo[e.to] = edgesTo[e.to] || []).push(e);
  });
  function upstream(id) {
    function walk(i, rel, note, seen, d) {
      var kids = [];
      if (d > 0) (edgesFrom[i] || []).forEach(function (e) {
        if (seen.indexOf(e.to) < 0) kids.push(walk(e.to, e.rel, e.note, seen.concat([i]), d - 1));
      });
      return { id: i, rel: rel, note: note, children: kids };
    }
    return walk(id, "", "", [], 12);
  }
  function downstream(id) {
    var out = [], todo = [id];
    while (todo.length) {
      var i = todo.shift();
      (edgesTo[i] || []).forEach(function (e) {
        if (out.indexOf(e.from) < 0 && e.from !== id) { out.push(e.from); todo.push(e.from); }
      });
    }
    return out;
  }
  function idLink(id) {
    if (CARDS[id]) return el("a", { cls: "ref", href: "#" + id, "data-ref": id, text: id });
    return el("code", { cls: "noref", text: id });
  }
  function treeNode(t, depth) {
    var n = NODES[t.id] || { kind: "", title: t.id }, prm = D.params[t.id];
    var li = el("li");
    var row = el("div", { cls: "tn" + (prm && prm.leaf ? " tn-leaf" : "") });
    var kids = t.children;
    var tog = null;
    if (kids.length) tog = el("button", { type: "button", cls: "tn-t", "aria-expanded": depth < 1 ? "true" : "false", "aria-label": "Show what it rests on" });
    row.appendChild(tog || el("span", { cls: "tn-sp" }));
    if (t.rel) row.appendChild(el("span", { cls: "tn-rel", text: D.rel[t.rel] || t.rel.replace(/_/g, " ") }));
    row.appendChild(idLink(t.id));
    if (n.title && n.title !== t.id) row.appendChild(el("span", { cls: "tn-ttl", text: n.title }));
    if (prm) {
      row.appendChild(el("span", { cls: "tn-val", text: prm.display }));
      if (prm.expr) row.appendChild(el("span", { cls: "expr", text: prm.expr, title: "The expression of this number" }));
      if (prm.leaf) row.appendChild(el("span", { cls: "tn-src", text: prm.source ? "source: " + prm.source : "no source document" }));
    } else if (n.kind && n.kind !== "parameter") {
      row.appendChild(el("span", { cls: "chip", text: D.kinds[n.kind] || n.kind }));
      if (n.status) row.appendChild(el("span", { cls: "tn-note", text: String(n.status).replace(/_/g, " ") }));
    }
    if (t.note) row.appendChild(el("span", { cls: "tn-note", text: t.note }));
    li.appendChild(row);
    if (kids.length) {
      var ul = el("ul"); ul.hidden = depth >= 1;
      var built = false;
      var build = function () { if (built) return; built = true; kids.forEach(function (k) { ul.appendChild(treeNode(k, depth + 1)); }); };
      if (!ul.hidden) build();
      li.appendChild(ul);
      tog.addEventListener("click", function () {
        var open = tog.getAttribute("aria-expanded") === "true";
        if (!open) build();
        tog.setAttribute("aria-expanded", open ? "false" : "true"); ul.hidden = open;
      });
    }
    return li;
  }
  function expandAll(root, on) {
    $$(".tn-t", root).forEach(function (t) {
      if ((t.getAttribute("aria-expanded") === "true") !== on) t.click();
    });
    if (on) { var again = $$(".tn-t[aria-expanded=false]", root); if (again.length) expandAll(root, on); }
  }
  var whyId = null, whyOpener = null;
  function closeWhy() {
    var p = $("#why-panel"); p.hidden = true; whyId = null;
    $$(".why[aria-expanded=true]").forEach(function (b) { b.setAttribute("aria-expanded", "false"); });
    if (whyOpener && document.contains(whyOpener)) whyOpener.focus();
  }
  function openWhy(id, btn, tab) {
    var p = $("#why-panel"); whyId = id; if (btn) whyOpener = btn;
    $$(".why[aria-expanded=true]").forEach(function (b) { b.setAttribute("aria-expanded", "false"); });
    $$('.why[data-why="' + id + '"]').forEach(function (b) { b.setAttribute("aria-expanded", "true"); });
    var n = NODES[id] || { title: id, kind: "" };
    p.innerHTML = "";
    var close = el("button", { type: "button", cls: "btn ico", "aria-label": "Close the panel", title: "Close (Escape)" }, [icon("cross")]);
    close.addEventListener("click", closeWhy);
    p.appendChild(el("div", { cls: "wp-h" }, [
      el("div", {}, [el("div", { cls: "kind", text: D.kinds[n.kind] || n.kind }), el("div", { cls: "wp-t", text: n.title || id }),
                     el("div", { cls: "wp-id", text: id })]), close]));
    var up = upstream(id), down = downstream(id);
    var tabs = el("div", { cls: "wp-tabs", role: "tablist" });
    var body = el("div", { cls: "wp-b", role: "tabpanel", tabindex: "0" });
    function tabBtn(key, label, count) {
      var b = el("button", { type: "button", cls: "wp-tab", role: "tab", "aria-selected": "false", "data-tab": key }, [label + " (" + count + ")"]);
      b.addEventListener("click", function () { show(key); }); tabs.appendChild(b); return b;
    }
    var nUp = 0; (function c(t) { t.children.forEach(function (k) { nUp++; c(k); }); })(up);
    tabBtn("up", "What it rests on", nUp); tabBtn("down", "What it affects", down.length);
    p.appendChild(tabs); p.appendChild(body);
    function show(key) {
      $$(".wp-tab", tabs).forEach(function (b) { b.setAttribute("aria-selected", b.getAttribute("data-tab") === key ? "true" : "false"); });
      body.innerHTML = "";
      if (key === "up") {
        if (!up.children.length) { body.appendChild(el("p", { text: "This record rests on no other record in the trace." })); return; }
        body.appendChild(el("p", { cls: "muted", text: "Each line is a record that this record rests on, down to the named numbers with their source documents." }));
        var bar = el("p"), ex = el("button", { type: "button", cls: "btn", text: "Open all levels" }), co = el("button", { type: "button", cls: "btn", text: "Close all levels" });
        ex.addEventListener("click", function () { expandAll(tree, true); }); co.addEventListener("click", function () { expandAll(tree, false); });
        bar.appendChild(ex); bar.appendChild(document.createTextNode(" ")); bar.appendChild(co); body.appendChild(bar);
        var tree = el("ul", { cls: "tree" });
        up.children.forEach(function (k) { tree.appendChild(treeNode(k, 0)); });
        body.appendChild(tree);
      } else {
        if (!down.length) { body.appendChild(el("p", { text: "No other record rests on this record." })); return; }
        body.appendChild(el("p", { cls: "muted", text: "These records rest on this record, nearest first. A change here changes them." }));
        var groups = {}, order = [];
        down.forEach(function (i) { var k = (NODES[i] || {}).kind || ""; if (!groups[k]) { groups[k] = []; order.push(k); } groups[k].push(i); });
        order.forEach(function (k) {
          body.appendChild(el("div", { cls: "dn-grp", text: (D.kinds[k] || k) + " (" + groups[k].length + ")" }));
          var ul = el("ul", { cls: "dn-list" });
          groups[k].forEach(function (i) {
            var nn = NODES[i] || {};
            ul.appendChild(el("li", {}, [idLink(i), el("span", { cls: "tn-ttl", text: nn.title && nn.title !== i ? nn.title : "" }),
                                         nn.status && (k === "compliance" || k === "finding") ? el("span", { cls: "chip", text: String(nn.status).replace(/_/g, " ") }) : null]));
          });
          body.appendChild(ul);
        });
      }
    }
    show(tab || "up");
    p.hidden = false;
    $(".wp-tab[aria-selected=true]", p).focus();
  }
  document.addEventListener("click", function (ev) {
    var b = ev.target.closest && ev.target.closest(".why[data-why]");
    if (b) {
      ev.preventDefault(); ev.stopPropagation();
      var id = b.getAttribute("data-why");
      if (whyId === id && !$("#why-panel").hidden) closeWhy(); else openWhy(id, b);
      return;
    }
    var a = ev.target.closest && ev.target.closest("#why-panel a.ref");
    if (a) { var t = a.getAttribute("data-ref"); ev.preventDefault(); navigateTo(t); }
  }, true);

  /* ---------------------------------------------------------------- filters of the decisions */
  (function () {
    var box = $("#filters"); if (!box) return;
    var chips = $$(".fchip", box), q = $("#flt-q"), count = $("#flt-n"), none = $("#flt-none");
    var cards = $$("#decisions details.card"), groups = $$("#decisions .stage-group");
    var texts = cards.map(function (c) { return c.textContent.toLowerCase(); });
    function apply() {
      var sel = {};
      chips.forEach(function (c) {
        if (c.getAttribute("aria-pressed") === "true") (sel[c.getAttribute("data-f")] = sel[c.getAttribute("data-f")] || []).push(c.getAttribute("data-v"));
      });
      var words = (q.value || "").toLowerCase().split(/\s+/).filter(Boolean), shown = 0;
      cards.forEach(function (c, i) {
        var rv = c.getAttribute("data-review");
        var ok = (!sel.type || sel.type.indexOf(c.getAttribute("data-type")) >= 0) &&
          (!sel.stage || sel.stage.indexOf(c.getAttribute("data-stage")) >= 0) &&
          (!sel.review || sel.review.indexOf(rv) >= 0) &&
          (!sel.origin || sel.origin.indexOf(c.getAttribute("data-origin")) >= 0) &&
          (!sel.needs || c.getAttribute("data-needs") === "1" || (rv === "rejected" || rv === "changes_requested" || rv === "stale")) &&
          words.every(function (w) { return texts[i].indexOf(w) >= 0; });
        c.hidden = !ok; if (ok) shown++;
      });
      groups.forEach(function (g) { g.hidden = !$$("details.card:not([hidden])", g).length; });
      count.textContent = shown; none.hidden = shown > 0;
      if (words.length && shown && shown <= 12) cards.forEach(function (c) { if (!c.hidden) c.open = true; });
    }
    chips.forEach(function (c) { c.addEventListener("click", function () {
      c.setAttribute("aria-pressed", c.getAttribute("aria-pressed") === "true" ? "false" : "true"); apply(); }); });
    q.addEventListener("input", apply);
    $("#flt-clear").addEventListener("click", function () {
      chips.forEach(function (c) { c.setAttribute("aria-pressed", "false"); }); q.value = ""; apply(); });
    window.__dossierFilters = apply;
    var hash = decodeURIComponent(location.hash.slice(1));
    var f = new URLSearchParams(location.search).get("show");
    if (f === "needs") { var nb = $(".fchip.needs", box); if (nb) { nb.setAttribute("aria-pressed", "true"); apply(); } }
  })();
  /* A link to a card that a filter hides: clear the filters first. */
  var oldGo = goTo;
  goTo = function (id, scroll) {
    var e = document.getElementById(id);
    if (e && e.hidden && $("#flt-clear")) $("#flt-clear").click();
    return oldGo(id, scroll);
  };

  /* ---------------------------------------------------------------- reviews */
  var API = window.DOSSIER_API || null;
  var REV = D.reviews;
  var STATE_TEXT = D.text;
  function stateOf(id) { var r = REV[id]; if (!r) return null; return r.stale ? "stale" : r.state; }
  var TONE = { open: ["none", "circle"], accepted: ["pass", "check"], rejected: ["fail", "cross"], changes_requested: ["warn", "pencil"], stale: ["stale", "refresh"] };
  function badge(state, id) {
    var t = TONE[state] || ["none", "dash"];
    var b = el("span", { cls: "badge t-" + t[0], "data-v": state, "data-review-badge": id }, [icon(t[1]), el("span", { cls: "bw", text: D.words[state] || state })]);
    return b;
  }
  function fill(tpl, a, b) { var i = 0; return tpl.replace(/\{\}/g, function () { return [a, b][i++]; }); }
  function renderReviewBlock(blk) {
    var id = blk.getAttribute("data-review-of"), r = REV[id]; if (!r) return;
    var state = stateOf(id); blk.setAttribute("data-state", state);
    var now = $(".rv-now", blk); if (!now) return;
    now.innerHTML = ""; now.appendChild(badge(state, id));
    var tpl = STATE_TEXT[r.state] || STATE_TEXT.open;
    if (r.by || r.date) now.appendChild(el("span", { cls: "rv-meta", text: fill(tpl, r.by || "A person", r.date || "a date that is not known") }));
    else now.appendChild(el("span", { cls: "rv-meta", text: STATE_TEXT.none }));
    if (r.stale) now.appendChild(el("span", { cls: "rv-stale", text: STATE_TEXT.stale }));
    if (r.comment) now.appendChild(el("q", { cls: "rv-comment", text: r.comment }));
  }
  function refreshReviews() {
    $$("[data-review-badge]").forEach(function (b) {
      var id = b.getAttribute("data-review-badge"), s = stateOf(id);
      if (!s || b.closest(".review")) return;
      var nb = badge(s, id); b.parentNode.replaceChild(nb, b);
    });
    $$(".fld.review").forEach(renderReviewBlock);
    $$("details.card[data-review]").forEach(function (c) { var s = stateOf(c.id); if (s) c.setAttribute("data-review", s); });
    attention(); progress(); if (window.__dossierFilters) window.__dossierFilters();
    var open = Object.keys(REV).filter(function (i) { return stateOf(i) === "open"; }).length;
    $$("[data-open-count]").forEach(function (e) { e.textContent = open; });
    sideFoot();
  }
  function reviewRule(id) {
    var s = stateOf(id), f = D.flags[id]; if (!f) return null;
    if (s === "rejected" || s === "changes_requested" || s === "stale") return s;
    if (s === "accepted") return null;
    if (f[0]) return "needs_review";
    if (f[1] === false) return "not_conservative";
    if (f[2]) return "agent";
    return null;
  }
  function attention() {
    var list = $("ol.att"); if (!list) return;
    $$("[data-att-dyn]", list).forEach(function (e) { e.remove(); });
    var order = Object.keys(D.rules), items = [];
    Object.keys(D.flags).forEach(function (id) {
      var rule = reviewRule(id); if (rule) items.push([order.indexOf(rule), id, rule]);
    });
    items.sort(function (a, b) { return a[0] - b[0]; });
    var ico = { fail: "cross", warn: "question", stale: "refresh", serious: "flag", info: "agent", none: "dash" };
    items.forEach(function (x) {
      var r = D.rules[x[2]], f = D.flags[x[1]];
      var li = el("li", { cls: "att-i t-" + r[0], "data-att-dyn": x[2] }, [
        el("span", { cls: "att-c" }, [icon(ico[r[0]] || "dash"), " ", r[1]]),
        el("span", { cls: "att-b" }, [idLink(x[1]), " ", el("span", { cls: "rl-t", text: f[4] }), " ", r[2]])]);
      list.appendChild(li);
    });
  }
  function progress() {
    $$(".rvp[data-rv-kind]").forEach(function (box) {
      var kind = box.getAttribute("data-rv-kind"), ids = Object.keys(REV).filter(function (i) { return (NODES[i] || {}).kind === kind; });
      var order = ["accepted", "rejected", "changes_requested", "stale", "open"], n = {};
      order.forEach(function (s) { n[s] = 0; }); ids.forEach(function (i) { n[stateOf(i)]++; });
      var nn = $(".rvp-n", box); if (nn) nn.textContent = n.accepted + " of " + ids.length + " accepted";
      var bar = $(".segbar", box);
      if (bar) { bar.innerHTML = ""; order.forEach(function (s) { if (n[s]) bar.appendChild(el("span", { cls: "seg t-" + TONE[s][0], style: "flex:" + n[s] + " 1 0", title: n[s] + " " + D.words[s] })); }); }
      var lg = $(".rvp-l", box);
      if (lg) { lg.innerHTML = ""; var first = true; order.forEach(function (s) { if (!n[s]) return;
        if (!first) lg.appendChild(document.createTextNode(" · ")); first = false;
        lg.appendChild(el("span", { cls: "lg t-" + TONE[s][0] }, [el("b", { text: n[s] }), " " + D.words[s]])); }); }
    });
  }
  function sideFoot() {
    var f = $("#side-foot"); if (!f) return;
    var open = Object.keys(REV).filter(function (i) { return stateOf(i) === "open"; }).length;
    f.textContent = Object.keys(REV).length ? open + " of " + Object.keys(REV).length + " records have no review." : "";
  }
  function addForm(blk) {
    if ($(".rv-form", blk)) return;
    var id = blk.getAttribute("data-review-of");
    var name = el("input", { type: "text", id: "rvn-" + id, placeholder: "Your name", autocomplete: "name", value: store.get("dossier-name") || "" });
    var com = el("textarea", { "aria-label": "Comment on this record", placeholder: "Comment (needed when you reject or request changes)" });
    var err = el("span", { cls: "rv-err", role: "alert" });
    function send(state) {
      var by = name.value.trim();
      if (!by) { err.textContent = "Write your name first."; name.focus(); return; }
      if ((state === "rejected" || state === "changes_requested") && !com.value.trim()) { err.textContent = "Write a comment for this review."; com.focus(); return; }
      err.textContent = ""; store.set("dossier-name", by);
      post(id, state, by, com.value.trim(), err, function () { com.value = ""; });
    }
    function btn(cls, label, state) {
      var b = el("button", { type: "button", cls: "btn rv-btn " + cls, text: label }); b.addEventListener("click", function () { send(state); }); return b;
    }
    blk.appendChild(el("div", { cls: "rv-form" }, [
      el("div", { cls: "row" }, [el("label", { for: "rvn-" + id, text: "Name" }), name]), com,
      el("div", { cls: "row" }, [btn("accept", "Accept", "accepted"), btn("changes", "Request changes", "changes_requested"),
                                  btn("reject", "Reject", "rejected"), btn("", "Open again", "open"), err])]));
    var cli = $(".rv-cli", blk); if (cli) cli.hidden = true;
  }
  function post(id, state, by, comment, err, done) {
    fetch(API + "/reviews", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record: id, state: state, by: by, comment: comment }) })
      .then(function (r) { if (!r.ok) throw new Error("The server answered " + r.status + "."); return r.json(); })
      .then(function (rv) {
        var x = rv.review || rv;
        REV[id] = { state: x.state || state, by: x.by || by, date: x.date || new Date().toISOString().slice(0, 10), comment: x.comment !== undefined ? x.comment : comment, stale: false };
        refreshReviews(); if (done) done(); toast("Review saved for " + id + ".");
      })
      .catch(function (e) { err.textContent = "The review was not saved. " + e.message; });
  }
  if (API) {
    $$(".fld.review").forEach(addForm);
    fetch(API + "/reviews").then(function (r) { return r.json(); }).then(function (j) {
      var rs = j.reviews || {};
      Object.keys(rs).forEach(function (id) {
        var x = rs[id]; REV[id] = { state: x.state, by: x.by || "", date: x.date || "", comment: x.comment || "", stale: !!x.stale };
      });
      refreshReviews();
    }).catch(function () { toast("The page could not load the current reviews."); });
  }
  sideFoot();

  /* ---------------------------------------------------------------- start */
  window.addEventListener("load", function () { if (location.hash) onHash(); else spy(); });
  spy();
})();
