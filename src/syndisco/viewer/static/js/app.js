/*
 * SynDisco Viewer: interface.
 * Depends on strings.js, data.js, search-core.js and the vendored fflate.
 * All data is rendered with textContent; file contents never become HTML.
 */
(function () {
  "use strict";

  var T = window.SDVStrings;
  var D = window.SDVData;

  var DEFAULT_CONFIG = {
    title: T.appName,
    version: "",
    showPrompts: false,
    stripPromptsOnDownload: false,
    highlightMentions: false,
    defaultGroupBy: "auto",
    defaultSort: "newest",
    rememberData: true,
    pageSize: 100,
    datasetsIndex: "datasets/index.json",
  };
  var OPEN_GROUPS_BY_DEFAULT = 10;
  var SPEAKER_COLORS = 8;

  var state = {
    config: DEFAULT_CONFIG,
    datasets: [],
    autoload: null,
    records: [],
    byId: new Map(),
    label: "",
    facets: null,
    query: "",
    terms: [],
    hits: null, // Map id -> [message indices] while searching
    groupBy: "auto",
    sort: "newest",
    filters: emptyFilters(),
    groupOpen: new Map(),
    limits: new Map(),
    order: [],
    selectedId: null,
    showPrompts: false,
    strip: false,
    highlightMentions: false,
    remember: true,
    timelineIndex: 0,
    importing: false,
  };

  function emptyFilters() {
    return { models: new Set(), speakers: new Set(), from: "", to: "", min: "", max: "", meta: {}, speakerQuery: "" };
  }

  // ------------------------------------------------------------ helpers

  function $(id) { return document.getElementById(id); }

  /** Create an element. attrs: class, text, on<event>, --css-var, others as attributes. */
  function h(tag, attrs, children) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) return;
        if (k === "class") el.className = v;
        else if (k === "text") el.textContent = v;
        else if (k.slice(0, 2) === "on") el.addEventListener(k.slice(2), v);
        else if (k.slice(0, 2) === "--") el.style.setProperty(k, v);
        else if (k === "style") Object.assign(el.style, v);
        else el.setAttribute(k, v === true ? "" : v);
      });
    }
    (children || []).forEach(function (c) {
      if (c === null || c === undefined || c === false) return;
      el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return el;
  }

  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

  function debounce(fn, ms) {
    var t;
    return function () {
      var args = arguments;
      clearTimeout(t);
      t = setTimeout(function () { fn.apply(null, args); }, ms);
    };
  }

  function nextFrame() { return new Promise(function (r) { setTimeout(r, 0); }); }

  function slug(text) {
    return String(text || "discussions").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "discussions";
  }

  function isTyping(el) {
    return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
  }

  function escapeRegExp(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  /** Append text to parent with search terms wrapped in <mark>. */
  function appendHighlighted(parent, text, terms) {
    if (!terms.length) { parent.appendChild(document.createTextNode(text)); return parent; }
    var re = new RegExp(terms.map(escapeRegExp).join("|"), "gi");
    var last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (!m[0].length) { re.lastIndex++; continue; }
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      parent.appendChild(h("mark", { text: m[0] }));
      last = m.index + m[0].length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
    return parent;
  }

  // ------------------------------------------------------------ colours

  function speakerColors(record) {
    if (record._colors) return record._colors;
    var next = 0;
    record._colors = record.speakerStats.map(function (s) {
      return s.seedOnly ? "var(--seed)" : "var(--s" + (next++ % SPEAKER_COLORS) + ")";
    });
    return record._colors;
  }

  function messageColor(record, m) {
    return m.isSeed ? "var(--seed)" : speakerColors(record)[m.speaker];
  }

  // -------------------------------------------------------- mentions

  /**
   * A matcher that finds any participant's name written out in message
   * text (no "@" or other marker required) and reports which speaker it
   * belongs to, so it can be coloured to match. Built once per record
   * and cached on it, since the set of names doesn't change.
   *
   * Matching is case-insensitive and requires a non-letter/digit
   * boundary on both sides, so a short name doesn't light up inside an
   * unrelated word. When several names could match at the same
   * position (one name contains another), the longest wins.
   */
  function mentionMatcher(record) {
    if (record._mentions !== undefined) return record._mentions;
    var names = record.speakers.filter(function (n) { return n; });
    if (!names.length) return (record._mentions = null);
    var byLower = {};
    names.forEach(function (n, i) { byLower[n.toLowerCase()] = i; });
    var sorted = names.slice().sort(function (a, b) { return b.length - a.length; });
    var pattern = sorted.map(escapeRegExp).join("|");
    try {
      var re = new RegExp("(?<![^\\W_])(?:" + pattern + ")(?![^\\W_])", "gi");
      return (record._mentions = { re: re, indexOf: byLower });
    } catch (e) {
      // lookbehind not supported in this engine; skip the feature quietly
      return (record._mentions = null);
    }
  }

  /** Append text to parent, colouring any participant name found in it
   * (if enabled) and highlighting search terms inside and outside of it. */
  function appendMessageText(parent, text, record, terms) {
    var matcher = state.highlightMentions ? mentionMatcher(record) : null;
    if (!matcher) return appendHighlighted(parent, text, terms);
    var re = matcher.re;
    re.lastIndex = 0;
    var last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (!m[0].length) { re.lastIndex++; continue; }
      if (m.index > last) appendHighlighted(parent, text.slice(last, m.index), terms);
      var idx = matcher.indexOf[m[0].toLowerCase()];
      var span = h("span", { class: "mention", "--speaker": speakerColors(record)[idx] });
      appendHighlighted(span, m[0], terms);
      parent.appendChild(span);
      last = m.index + m[0].length;
    }
    if (last < text.length) appendHighlighted(parent, text.slice(last), terms);
    return parent;
  }

  /** The turn-order strip: one segment per message, merged when consecutive. */
  function stripBackground(record) {
    var n = record.messages.length;
    if (!n) return "var(--rule)";
    var stops = [];
    var start = 0;
    for (var i = 1; i <= n; i++) {
      var prev = messageColor(record, record.messages[i - 1]);
      if (i === n || messageColor(record, record.messages[i]) !== prev) {
        stops.push(prev + " " + (start / n * 100).toFixed(3) + "% " + (i / n * 100).toFixed(3) + "%");
        start = i;
      }
    }
    return "linear-gradient(to right, " + stops.join(", ") + ")";
  }

  // ------------------------------------------------------------- status

  var statusTimer = null;

  function setStatus(opts) {
    var el = $("status");
    clearTimeout(statusTimer);
    if (!opts) { el.hidden = true; clear(el); return; }
    clear(el);
    el.hidden = false;
    el.classList.toggle("is-error", !!opts.error);
    el.appendChild(h("span", { text: opts.text }));
    if (opts.progress !== undefined) {
      var bar = h("span");
      bar.style.width = Math.round(opts.progress * 100) + "%";
      el.appendChild(h("span", { class: "progress", "aria-hidden": "true" }, [bar]));
    }
    if (opts.details && opts.details.length) {
      el.appendChild(h("details", null, [
        h("summary", { text: T.showDetails }),
        h("ul", null, opts.details.map(function (d) { return h("li", { text: d.path + ": " + d.reason }); })),
      ]));
    }
    if (opts.dismissible) {
      el.appendChild(h("button", { type: "button", class: "btn btn-small btn-quiet", text: T.dismiss, onclick: function () { setStatus(null); } }));
    }
    if (opts.autoHide) statusTimer = setTimeout(function () { setStatus(null); }, opts.autoHide);
  }

  // ------------------------------------------------------------- search

  var searcher = {
    worker: null,
    corpus: null,
    docs: [],
    seq: 0,
    pending: new Map(),

    start: function () {
      try {
        this.worker = new Worker("js/search-worker.js");
        var self = this;
        this.worker.onmessage = function (e) {
          var cb = self.pending.get(e.data.seq);
          if (cb) { self.pending.delete(e.data.seq); cb(e.data); }
        };
        this.worker.onerror = function (e) {
          if (e && e.preventDefault) e.preventDefault();
          self.useFallback();
        };
      } catch (e) {
        this.worker = null;
      }
    },

    useFallback: function () {
      if (this.worker) { this.worker.terminate(); this.worker = null; }
      this.corpus = window.SDVSearch.buildCorpus(this.docs);
      var self = this;
      this.pending.forEach(function (cb, seq) {
        if (cb.query !== undefined) cb(self.localSearch(cb.query));
        else cb({});
      });
      this.pending.clear();
    },

    localSearch: function (query) {
      var terms = window.SDVSearch.parseQuery(query);
      return { terms: terms, results: window.SDVSearch.search(this.corpus || [], terms) };
    },

    index: function (records) {
      this.docs = records.map(function (r) {
        var metaText = Object.keys(r.meta).map(function (k) { return k + " " + D.metaValueLabel(r.meta[k]); }).join(" ");
        return {
          id: r.id,
          fields: [r.path, r.speakers.join(" "), r.models.join(" "), metaText].join(" "),
          messages: r.messages.map(function (m) { return m.text; }),
        };
      });
      if (this.worker) {
        var seq = ++this.seq;
        this.worker.postMessage({ type: "index", docs: this.docs, seq: seq });
        this.corpus = null;
      } else {
        this.corpus = window.SDVSearch.buildCorpus(this.docs);
      }
    },

    query: function (query) {
      var self = this;
      if (!this.worker) return Promise.resolve(this.localSearch(query));
      return new Promise(function (resolve) {
        var seq = ++self.seq;
        resolve.query = query;
        self.pending.set(seq, resolve);
        self.worker.postMessage({ type: "search", query: query, seq: seq });
      });
    },
  };

  var searchSeq = 0;

  function runSearch() {
    var query = state.query.trim();
    var mySeq = ++searchSeq;
    if (!query) {
      state.terms = [];
      state.hits = null;
      renderAll();
      return;
    }
    searcher.query(query).then(function (res) {
      if (mySeq !== searchSeq) return;
      state.terms = res.terms || [];
      state.hits = new Map((res.results || []).map(function (r) { return [r.id, r.hits]; }));
      state.limits.clear();
      renderAll();
    });
  }

  // -------------------------------------------------------- loading data

  function addRecords(files, replace, onProgress) {
    if (replace) { state.records = []; state.byId = new Map(); }
    var errors = [];
    var added = 0;
    var i = 0;
    function batch() {
      var end = Math.min(i + 250, files.length);
      for (; i < end; i++) {
        var f = files[i];
        var path = D.normalizePath(f.path);
        var id = path;
        var n = 2;
        while (state.byId.has(id)) id = path.replace(/(\.json)?$/i, " (" + n++ + ")$1");
        var parsed = D.parseDiscussion(id, f.text);
        if (parsed.error) { errors.push({ path: path, reason: parsed.error }); continue; }
        if (id !== path) errors.push({ path: path, reason: T.errDuplicate });
        state.records.push(parsed.record);
        state.byId.set(id, parsed.record);
        added++;
      }
      if (onProgress) onProgress(i, files.length);
      if (i < files.length) return nextFrame().then(batch);
      return Promise.resolve({ added: added, errors: errors });
    }
    return batch();
  }

  /** Common path for every import. result: {files, errors, ignored} */
  function ingest(result, opts) {
    var errors = (result.errors || []).slice();
    if (!result.files.length && !errors.length) {
      setStatus({ text: T.nothingFound, error: true, dismissible: true });
      return Promise.resolve();
    }
    return addRecords(result.files, opts.replace, function (done, total) {
      if (total > 200) setStatus({ text: T.reading(done, total), progress: done / total });
    }).then(function (res) {
      errors = errors.concat(res.errors);
      state.label = opts.label || state.label || T.localData;
      if (opts.restoreHash) readHash();
      afterDataChanged();
      if (opts.persist !== false) persist();
      var skipped = errors.filter(function (e) { return e.reason !== T.errDuplicate; }).length;
      var text = T.loaded(res.added) + (skipped ? " " + T.skipped(skipped) : "");
      var warnings = state.records.filter(function (r) { return r.warnings.length; }).slice(0, 200)
        .map(function (r) { return { path: r.path, reason: r.warnings.join("; ") }; });
      setStatus({
        text: text,
        error: res.added === 0,
        details: errors.concat(warnings),
        dismissible: true,
        autoHide: errors.length || warnings.length ? null : 5000,
      });
    });
  }

  function loadLocal(items) {
    if (!items.length) return;
    state.importing = true;
    setStatus({ text: T.reading(0, items.length), progress: 0 });
    D.readLocalItems(items, function (done, total) {
      setStatus({ text: T.reading(done, total), progress: done / total });
    }).then(function (res) {
      var hadData = state.records.length > 0;
      return ingest(res, { replace: false, label: hadData && state.label !== T.localData ? state.label + " + " + T.localData : T.localData });
    }).catch(function (e) {
      setStatus({ text: T.loadFailed(T.localData, e.message || e), error: true, dismissible: true });
    }).then(function () {
      state.importing = false;
    });
  }

  function loadDataset(url, label, opts) {
    opts = opts || {};
    setStatus({ text: T.downloading, progress: 0 });
    return D.loadUrl(url, function (done, total) {
      setStatus({ text: T.reading(done, total), progress: done / total });
    }).then(function (res) {
      return ingest(res, { replace: true, label: label || url, persist: false, restoreHash: opts.restoreHash });
    }).catch(function (e) {
      setStatus({ text: T.loadFailed(label || url, e.message || e), error: true, dismissible: true });
      showEmptyIfNeeded();
    });
  }

  function persist() {
    if (!state.config.rememberData || !state.remember) return;
    D.cache.save(state.records.map(function (r) { return { path: r.path, text: r.raw }; }), state.label);
  }

  function clearAll() {
    if (state.records.length && !window.confirm(T.clearConfirm)) return;
    state.records = [];
    state.byId = new Map();
    state.selectedId = null;
    state.label = "";
    state.query = "";
    $("search").value = "";
    state.hits = null;
    state.filters = emptyFilters();
    D.cache.clear();
    var url = new URL(location.href);
    url.search = "";
    url.hash = "";
    history.replaceState(null, "", url.href);
    setStatus(null);
    afterDataChanged();
  }

  // ------------------------------------------------------------- facets

  function computeFacets() {
    var models = new Map(), speakers = new Map(), meta = new Map(), folders = new Set();
    var minDate = null, maxDate = null;
    function bump(map, key) { map.set(key, (map.get(key) || 0) + 1); }
    state.records.forEach(function (r) {
      folders.add(r.folder);
      r.models.forEach(function (m) { bump(models, m); });
      r.speakers.forEach(function (s) { bump(speakers, s); });
      Object.keys(r.meta).forEach(function (k) {
        if (!meta.has(k)) meta.set(k, new Map());
        bump(meta.get(k), D.metaValueLabel(r.meta[k]));
      });
      if (r.date) {
        if (!minDate || r.date < minDate) minDate = r.date;
        if (!maxDate || r.date > maxDate) maxDate = r.date;
      }
    });
    // only keys with a manageable number of distinct values become filters
    var metaKeys = [];
    meta.forEach(function (values, key) {
      if (values.size <= 60) metaKeys.push(key);
    });
    metaKeys.sort();
    state.facets = {
      models: sortedEntries(models),
      speakers: sortedEntries(speakers),
      meta: meta,
      metaKeys: metaKeys,
      folderCount: folders.size,
      minDate: minDate,
      maxDate: maxDate,
    };
  }

  function sortedEntries(map) {
    return Array.from(map.entries()).sort(function (a, b) { return a[0].localeCompare(b[0], "en", { numeric: true }); });
  }

  // ---------------------------------------------------- filter and group

  function activeFilterCount() {
    var f = state.filters, n = 0;
    if (f.models.size) n++;
    if (f.speakers.size) n++;
    if (f.from || f.to) n++;
    if (f.min !== "" || f.max !== "") n++;
    Object.keys(f.meta).forEach(function (k) { if (f.meta[k].size) n++; });
    return n;
  }

  function passes(r) {
    var f = state.filters;
    if (state.hits && !state.hits.has(r.id)) return false;
    if (f.models.size && !r.models.some(function (m) { return f.models.has(m); })) return false;
    if (f.speakers.size && !r.speakers.some(function (s) { return f.speakers.has(s); })) return false;
    if (f.from || f.to) {
      var day = D.dayKey(r.date);
      if (!day) return false;
      if (f.from && day < f.from) return false;
      if (f.to && day > f.to) return false;
    }
    var count = r.messages.length;
    if (f.min !== "" && count < Number(f.min)) return false;
    if (f.max !== "" && count > Number(f.max)) return false;
    for (var k in f.meta) {
      if (f.meta[k].size) {
        var value = k in r.meta ? D.metaValueLabel(r.meta[k]) : "";
        if (!f.meta[k].has(value)) return false;
      }
    }
    return true;
  }

  var SORTERS = {
    newest: function (a, b) { return (b.date || 0) - (a.date || 0) || a.path.localeCompare(b.path); },
    oldest: function (a, b) { return (a.date || Infinity) - (b.date || Infinity) || a.path.localeCompare(b.path); },
    path: function (a, b) { return a.path.localeCompare(b.path, "en", { numeric: true }); },
    most: function (a, b) { return b.messages.length - a.messages.length || a.path.localeCompare(b.path); },
    fewest: function (a, b) { return a.messages.length - b.messages.length || a.path.localeCompare(b.path); },
  };

  function effectiveGroupBy() {
    var g = state.groupBy;
    if (g === "auto") return state.facets && state.facets.folderCount > 1 ? "folder" : "none";
    if (g.indexOf("meta:") === 0 && !(state.facets && state.facets.meta.has(g.slice(5)))) return "none";
    return g;
  }

  function groupKeys(r, g) {
    switch (g) {
      case "folder": return [r.folder];
      case "model": return r.models.length ? r.models : [""];
      case "speaker": return r.speakers.length ? r.speakers : [""];
      case "day": return [D.dayKey(r.date)];
      case "month": return [D.monthKey(r.date)];
      case "none": return ["*"];
      default:
        var key = g.slice(5);
        return [key in r.meta ? D.metaValueLabel(r.meta[key]) : ""];
    }
  }

  function groupLabel(key, g) {
    if (g === "folder") return key || T.topLevel;
    if (g === "day" || g === "month") return key || T.unknownDate;
    return key === "" ? T.noValue : key;
  }

  function computeView() {
    var g = effectiveGroupBy();
    var list = state.records.filter(passes).sort(SORTERS[state.sort] || SORTERS.newest);
    var groups = new Map();
    list.forEach(function (r) {
      groupKeys(r, g).forEach(function (key) {
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(r);
      });
    });
    var keys = Array.from(groups.keys());
    var desc = g === "day" || g === "month";
    keys.sort(function (a, b) {
      if (a === "" && b !== "") return 1;
      if (b === "" && a !== "") return -1;
      return desc ? b.localeCompare(a) : a.localeCompare(b, "en", { numeric: true });
    });
    var seen = new Set();
    var order = [];
    keys.forEach(function (k) {
      groups.get(k).forEach(function (r) { if (!seen.has(r.id)) { seen.add(r.id); order.push(r.id); } });
    });
    state.order = order;
    return {
      groupBy: g,
      total: list.length,
      list: list,
      groups: keys.map(function (k) { return { key: k, label: groupLabel(k, g), records: groups.get(k) }; }),
    };
  }

  // ---------------------------------------------------------- rendering

  var view = null;

  function renderAll() {
    view = computeView();
    renderControls();
    renderCatalogue();
    renderDetail();
    writeHash();
  }

  function afterDataChanged() {
    computeFacets();
    searcher.index(state.records);
    if (state.selectedId && !state.byId.has(state.selectedId)) state.selectedId = null;
    state.limits.clear();
    showEmptyIfNeeded();
    renderHeader();
    renderFilters();
    if (state.query.trim()) runSearch(); else renderAll();
  }

  function showEmptyIfNeeded() {
    var empty = state.records.length === 0;
    $("empty").hidden = !empty;
    $("layout").hidden = empty;
    $("btn-clear").hidden = empty;
    renderHeader();
  }

  function renderHeader() {
    var label = $("dataset-label");
    if (state.records.length && state.label) {
      clear(label);
      label.appendChild(document.createTextNode(T.loadedFrom + ": "));
      label.appendChild(h("strong", { text: state.label }));
      label.hidden = false;
    } else {
      label.hidden = true;
    }
  }

  function renderControls() {
    var group = $("group-by");
    var current = state.groupBy;
    clear(group);
    Object.keys(T.groupOptions).forEach(function (k) {
      group.appendChild(h("option", { value: k, text: T.groupOptions[k] }));
    });
    (state.facets ? state.facets.metaKeys : []).forEach(function (k) {
      group.appendChild(h("option", { value: "meta:" + k, text: T.metaGroupPrefix + k }));
    });
    group.value = current;
    if (group.value !== current) { group.value = "auto"; }
    $("sort-by").value = state.sort;
    $("filters-summary").textContent = T.filtersActive(activeFilterCount());
    $("results-count").textContent = T.resultsCount(view.total, state.records.length);
    $("btn-download-results").disabled = view.total === 0;
  }

  function renderCatalogue() {
    var nav = clear($("catalogue"));
    if (!view.total) {
      nav.appendChild(h("p", { class: "no-results", text: T.noResults }));
      return;
    }
    var frag = document.createDocumentFragment();
    var pageSize = Number(state.config.pageSize) || 100;
    var grouped = view.groupBy !== "none";

    view.groups.forEach(function (group, gi) {
      var groupId = view.groupBy + "|" + group.key;
      var open = state.groupOpen.has(groupId) ? state.groupOpen.get(groupId) : gi < OPEN_GROUPS_BY_DEFAULT;
      var section = h("section", { class: "group" + (open ? "" : " is-collapsed") });

      if (grouped) {
        section.appendChild(h("div", { class: "group-head" }, [
          h("button", {
            type: "button", class: "group-toggle", "aria-expanded": String(open),
            onclick: function () { state.groupOpen.set(groupId, !open); renderCatalogue(); },
          }, [
            h("span", { class: "group-name", text: group.label }),
            h("span", { class: "group-count", text: group.records.length.toLocaleString("en") }),
          ]),
          h("button", {
            type: "button", class: "btn btn-small btn-quiet group-download", text: T.download,
            "aria-label": T.downloadGroup + ": " + group.label, title: T.downloadGroup,
            onclick: function () { downloadMany(group.records, group.label); },
          }),
        ]));
      }

      if (open || !grouped) {
        var limit = state.limits.get(groupId) || pageSize;
        var ul = h("ul", { class: "items" });
        group.records.slice(0, limit).forEach(function (r) {
          ul.appendChild(h("li", null, [renderItem(r, grouped ? view.groupBy : "none")]));
        });
        section.appendChild(ul);
        if (group.records.length > limit) {
          var more = Math.min(pageSize, group.records.length - limit);
          section.appendChild(h("button", {
            type: "button", class: "btn btn-small show-more", text: T.showMore(more),
            onclick: function () { state.limits.set(groupId, limit + pageSize); renderCatalogue(); },
          }));
        }
      }
      frag.appendChild(section);
    });
    nav.appendChild(frag);
  }

  function renderItem(r, groupBy) {
    var hits = state.hits ? state.hits.get(r.id) : null;
    var title = h("span", { class: "item-name" });
    if (r.folder && groupBy !== "folder") title.appendChild(h("span", { class: "item-folder", text: r.folder + "/" }));
    title.appendChild(document.createTextNode(r.title));

    var stats = h("span", { class: "item-stats" }, [
      h("span", { text: T.messages(r.messages.length) }),
      h("span", { text: T.speakers(r.speakers.length) }),
    ]);

    var snippet = null;
    var snippetSource = null;
    if (hits && hits.length) snippetSource = r.messages[hits[0]];
    else if (!state.terms.length) snippetSource = r.messages.find(function (m) { return !m.isSeed; }) || r.messages[0];
    if (snippetSource) {
      snippet = h("span", { class: "item-snippet" });
      snippet.appendChild(h("b", { text: snippetSource.name + ": " }));
      appendHighlighted(snippet, excerpt(snippetSource.text, state.terms), state.terms);
    }

    var strip = h("span", { class: "strip", "aria-hidden": "true" });
    strip.style.background = stripBackground(r);

    var children = [
      h("span", { class: "item-title" }, [title, titleIsDate(r) ? null : h("span", { class: "item-date", text: r.dateLabel })]),
      strip,
      stats,
      snippet,
    ];
    if (hits) {
      children.push(h("span", { class: "item-hits", text: hits.length ? T.matchingMessages(hits.length) : T.matchedDetails }));
    }
    return h("button", {
      type: "button", class: "item", "data-id": r.id,
      "aria-current": r.id === state.selectedId ? "true" : "false",
      onclick: function () { select(r.id, { fromList: true }); },
    }, children);
  }

  /** True when the file name is itself the timestamp, as syndisco names files. */
  function titleIsDate(r) {
    var d = D.parseTimestamp(r.title);
    return !!(d && r.date && Math.abs(d - r.date) < 60000);
  }

  function excerpt(text, terms) {
    var flat = text.replace(/\s+/g, " ").trim();
    if (!terms.length) return flat.length > 180 ? flat.slice(0, 180) + "…" : flat;
    var lower = flat.toLowerCase();
    var at = -1;
    terms.forEach(function (t) { var i = lower.indexOf(t); if (i >= 0 && (at < 0 || i < at)) at = i; });
    if (at < 0) return flat.slice(0, 180);
    var start = Math.max(0, at - 60);
    if (start > 0) {
      var space = flat.indexOf(" ", start);
      if (space >= 0 && space < at) start = space + 1;
    }
    return (start > 0 ? "…" : "") + flat.slice(start, start + 180) + (start + 180 < flat.length ? "…" : "");
  }

  // ------------------------------------------------------------- detail

  function select(id, opts) {
    opts = opts || {};
    if (!state.byId.has(id)) return;
    state.selectedId = id;
    state.timelineIndex = 0;
    var items = document.querySelectorAll(".item");
    for (var i = 0; i < items.length; i++) {
      items[i].setAttribute("aria-current", items[i].getAttribute("data-id") === id ? "true" : "false");
    }
    $("layout").classList.add("show-detail");
    renderDetail();
    writeHash();
    var detail = $("detail");
    detail.scrollTop = 0;
    if (opts.fromList && window.matchMedia("(max-width: 52rem)").matches) detail.focus({ preventScroll: true });
    var hits = state.hits && state.hits.get(id);
    if (hits && hits.length) scrollToMessage(hits[0], false);
    if (!opts.fromList) {
      var item = document.querySelector('.item[data-id="' + cssEscape(id) + '"]');
      if (item) item.scrollIntoView({ block: "nearest" });
    }
  }

  function cssEscape(s) { return window.CSS && CSS.escape ? CSS.escape(s) : s.replace(/["\\]/g, "\\$&"); }

  function step(delta) {
    if (!state.order.length) return;
    var i = state.order.indexOf(state.selectedId);
    var next = i < 0 ? 0 : i + delta;
    if (next < 0 || next >= state.order.length) return;
    select(state.order[next]);
  }

  function renderDetail() {
    var detail = clear($("detail"));
    var r = state.selectedId && state.byId.get(state.selectedId);
    if (!r) {
      detail.appendChild(h("div", { class: "detail-placeholder" }, [h("p", { text: T.noSelection })]));
      return;
    }
    var pos = state.order.indexOf(r.id);
    var inner = h("article", { class: "detail-inner", "aria-labelledby": "detail-title" });

    inner.appendChild(h("div", { class: "detail-nav" }, [
      h("button", {
        type: "button", class: "btn btn-back", text: T.back,
        onclick: function () { $("layout").classList.remove("show-detail"); },
      }),
      h("span", { class: "spacer" }),
      h("button", { type: "button", class: "btn", text: T.previous, disabled: pos <= 0, onclick: function () { step(-1); }, "aria-keyshortcuts": "k" }),
      h("button", { type: "button", class: "btn", text: T.next, disabled: pos < 0 || pos >= state.order.length - 1, onclick: function () { step(1); }, "aria-keyshortcuts": "j" }),
      h("button", { type: "button", class: "btn btn-primary", text: T.downloadJson, onclick: function () { D.downloadRecord(r, state.strip); } }),
    ]));

    var head = h("header", { class: "detail-head" }, [
      r.folder ? h("p", { class: "detail-path", text: r.folder + "/" }) : null,
      h("h2", { id: "detail-title", class: "detail-title", text: r.title }),
      h("p", { class: "detail-stats" }, [
        h("span", { text: r.dateLabel }),
        h("span", { text: T.messages(r.messages.length) }),
        h("span", { text: T.speakers(r.speakers.length) }),
        r.models.length ? h("span", { text: r.models.join(", ") }) : null,
      ]),
    ]);
    r.warnings.forEach(function (w) { head.appendChild(h("p", { class: "detail-warning", text: w })); });
    inner.appendChild(head);

    if (r.messages.length) inner.appendChild(renderTimeline(r));
    inner.appendChild(renderParticipants(r));
    var metaKeys = Object.keys(r.meta);
    if (metaKeys.length) {
      inner.appendChild(h("section", { class: "panel" }, [
        h("h3", { text: T.metadata }),
        renderDl(r.meta),
      ]));
    }
    inner.appendChild(h("h3", { class: "transcript-title", id: "transcript-title", text: T.transcript }));
    inner.appendChild(renderTranscript(r));
    detail.appendChild(inner);
  }

  function renderDl(obj) {
    var dl = h("dl", { class: "meta-list" });
    Object.keys(obj).forEach(function (k) {
      var v = obj[k];
      dl.appendChild(h("dt", { text: k }));
      dl.appendChild(h("dd", { text: typeof v === "object" && v !== null ? JSON.stringify(v, null, 2) : String(v) }));
    });
    return dl;
  }

  function renderTimeline(r) {
    var n = r.messages.length;
    var cursor = h("span", { class: "strip-cursor", "aria-hidden": "true" });
    var strip = h("div", {
      class: "strip strip-large", role: "slider", tabindex: "0",
      "aria-label": T.timelineLabel, "aria-valuemin": "1", "aria-valuemax": String(n),
    }, [cursor]);
    strip.style.background = stripBackground(r);

    function setIndex(i, scroll) {
      i = Math.max(0, Math.min(n - 1, i));
      state.timelineIndex = i;
      var m = r.messages[i];
      cursor.style.left = "calc(" + ((i + 0.5) / n * 100) + "% - 1px)";
      strip.setAttribute("aria-valuenow", String(i + 1));
      strip.setAttribute("aria-valuetext", T.messageOf(i + 1, n) + ", " + m.name);
      strip.title = T.messageOf(i + 1, n) + ": " + m.name;
      if (scroll) scrollToMessage(i, true);
    }
    function indexAt(clientX) {
      var rect = strip.getBoundingClientRect();
      return Math.floor((clientX - rect.left) / rect.width * n);
    }
    strip.addEventListener("click", function (e) { setIndex(indexAt(e.clientX), true); });
    strip.addEventListener("mousemove", function (e) {
      var i = Math.max(0, Math.min(n - 1, indexAt(e.clientX)));
      strip.title = T.messageOf(i + 1, n) + ": " + r.messages[i].name;
    });
    strip.addEventListener("keydown", function (e) {
      var i = state.timelineIndex;
      var map = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: n - 1, PageDown: i + 10, PageUp: i - 10 };
      if (e.key in map) { e.preventDefault(); setIndex(map[e.key], true); }
    });
    setIndex(state.timelineIndex, false);
    var parts = [strip];
    var hits = state.hits && state.hits.get(r.id);
    if (hits && hits.length) {
      var ticks = h("div", { class: "hit-ticks", "aria-hidden": "true" });
      hits.forEach(function (i) {
        var tick = h("span", { onclick: function () { setIndex(i, true); } });
        tick.style.left = (i / n * 100) + "%";
        tick.style.width = "max(3px, " + (100 / n) + "%)";
        ticks.appendChild(tick);
      });
      parts.push(ticks);
    }
    parts.push(h("p", { class: "hint", text: hits && hits.length ? T.timelineHits(hits.length) : T.timelineLabel }));
    return h("div", { class: "timeline" }, parts);
  }

  function renderParticipants(r) {
    var colors = speakerColors(r);
    var list = h("ul", { class: "participants" });
    r.speakerStats.forEach(function (s, i) {
      var metaText = s.seedOnly ? T.seedModelLabel : (s.model || T.noModel);
      var li = h("li", { class: "participant", "--speaker": colors[i] }, [
        h("span", { class: "participant-name", text: s.name }),
        h("span", { class: "participant-meta" }, [
          h("span", { text: metaText }), document.createTextNode(", "), h("span", { text: T.messages(s.count) }),
        ]),
      ]);
      if (state.showPrompts && !s.seedOnly) {
        if (!s.prompts.length) {
          li.appendChild(h("p", { class: "participant-meta", text: T.noPrompt, style: { gridColumn: "1 / -1" } }));
        } else {
          s.prompts.forEach(function (p, pi) {
            li.appendChild(h("details", null, [
              h("summary", { text: T.promptLabel + (s.prompts.length > 1 ? " " + (pi + 1) : "") }),
              h("pre", { class: "prompt-text", text: p }),
            ]));
          });
          if (s.prompts.length > 1) li.appendChild(h("p", { class: "participant-meta", text: T.multiplePrompts(s.prompts.length), style: { gridColumn: "1 / -1" } }));
        }
      }
      list.appendChild(li);
    });
    var panel = h("section", { class: "panel" }, [h("h3", { text: T.participants }), list]);
    if (!state.showPrompts && r.hasPrompts) panel.appendChild(h("p", { class: "prompts-note", text: T.promptsHidden }));
    return panel;
  }

  function renderTranscript(r) {
    var ol = h("ol", { class: "transcript", "aria-labelledby": "transcript-title" });
    var n = r.messages.length;
    r.messages.forEach(function (m, i) {
      var head = h("header", { class: "message-head" }, [
        h("span", { class: "message-name", text: m.name }),
        m.isSeed ? h("span", { class: "seed-badge", text: T.seedComment }) : (m.model ? h("span", { class: "message-model", text: m.model }) : null),
        h("span", { class: "message-index", text: String(i + 1), "aria-label": T.messageOf(i + 1, n) }),
      ]);
      var text = appendMessageText(h("div", { class: "message-text" }), m.text, r, state.terms);
      ol.appendChild(h("li", {
        id: "msg-" + i, class: "message" + (m.isSeed ? " is-seed" : ""), "--speaker": messageColor(r, m),
      }, [head, text, m.extra ? h("div", { class: "message-extra" }, [renderDl(m.extra)]) : null]));
    });
    return ol;
  }

  function scrollToMessage(i, flash) {
    var el = document.getElementById("msg-" + i);
    if (!el) return;
    el.scrollIntoView({ block: "start", behavior: flash && !reducedMotion() ? "smooth" : "auto" });
    if (flash) {
      el.classList.remove("is-flash");
      void el.offsetWidth;
      el.classList.add("is-flash");
    }
  }

  function reducedMotion() { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; }

  // ------------------------------------------------------------ filters

  function renderFilters() {
    var body = clear($("filters-body"));
    var f = state.filters;
    var facets = state.facets;
    if (!facets) return;

    function checkList(entries, selected, onChange, extraFilter) {
      var list = h("div", { class: "check-list" });
      entries.forEach(function (e) {
        if (extraFilter && !extraFilter(e[0])) return;
        var label = e[0] === "" ? T.noValue : e[0];
        list.appendChild(h("label", { title: label }, [
          h("input", {
            type: "checkbox", checked: selected.has(e[0]),
            onchange: function (ev) {
              if (ev.target.checked) selected.add(e[0]); else selected.delete(e[0]);
              onChange();
            },
          }),
          h("span", { class: "label-text", text: label }),
          h("span", { class: "count", text: e[1].toLocaleString("en") }),
        ]));
      });
      return list;
    }
    function changed() { state.limits.clear(); renderAll(); }

    if (facets.models.length) {
      body.appendChild(h("fieldset", { class: "filter-group" }, [
        h("legend", { text: T.filterModels }),
        checkList(facets.models, f.models, changed),
      ]));
    }

    if (facets.speakers.length) {
      var speakerBox = h("div");
      var renderSpeakers = function () {
        clear(speakerBox).appendChild(checkList(facets.speakers, f.speakers, changed, function (name) {
          return !f.speakerQuery || name.toLowerCase().indexOf(f.speakerQuery.toLowerCase()) >= 0 || f.speakers.has(name);
        }));
      };
      var group = h("fieldset", { class: "filter-group" }, [h("legend", { text: T.filterSpeakers })]);
      if (facets.speakers.length > 8) {
        group.appendChild(h("input", {
          type: "text", placeholder: T.filterSpeakersSearch, "aria-label": T.filterSpeakersSearch, value: f.speakerQuery,
          oninput: function (e) { f.speakerQuery = e.target.value; renderSpeakers(); },
        }));
      }
      group.appendChild(speakerBox);
      renderSpeakers();
      body.appendChild(group);
    }

    if (facets.minDate) {
      body.appendChild(h("fieldset", { class: "filter-group" }, [
        h("legend", { text: T.filterDates }),
        h("div", { class: "range-row" }, [
          h("label", null, [T.filterFrom, h("input", {
            type: "date", value: f.from, min: D.dayKey(facets.minDate), max: D.dayKey(facets.maxDate),
            onchange: function (e) { f.from = e.target.value; changed(); },
          })]),
          h("label", null, [T.filterTo, h("input", {
            type: "date", value: f.to, min: D.dayKey(facets.minDate), max: D.dayKey(facets.maxDate),
            onchange: function (e) { f.to = e.target.value; changed(); },
          })]),
        ]),
      ]));
    }

    body.appendChild(h("fieldset", { class: "filter-group" }, [
      h("legend", { text: T.filterLength }),
      h("div", { class: "range-row" }, [
        h("label", null, [T.filterMin, h("input", {
          type: "number", min: "0", inputmode: "numeric", value: f.min,
          oninput: debounce(function (e) { f.min = e.target.value; changed(); }, 250),
        })]),
        h("label", null, [T.filterMax, h("input", {
          type: "number", min: "0", inputmode: "numeric", value: f.max,
          oninput: debounce(function (e) { f.max = e.target.value; changed(); }, 250),
        })]),
      ]),
    ]));

    facets.metaKeys.forEach(function (key) {
      var selected = f.meta[key] || (f.meta[key] = new Set());
      var entries = sortedEntries(facets.meta.get(key));
      var missing = state.records.length - entries.reduce(function (s, e) { return s + e[1]; }, 0);
      if (missing > 0) entries.push(["", missing]);
      body.appendChild(h("fieldset", { class: "filter-group" }, [
        h("legend", { text: key }),
        checkList(entries, selected, changed),
      ]));
    });

    body.appendChild(h("button", {
      type: "button", class: "btn btn-small", text: T.clearFilters,
      onclick: function () { state.filters = emptyFilters(); renderFilters(); changed(); },
    }));
  }

  // ------------------------------------------------------------ export

  function downloadMany(records, name) {
    var unique = Array.from(new Map(records.map(function (r) { return [r.id, r]; })).values());
    if (!unique.length) return;
    setStatus({ text: T.preparingDownload });
    setTimeout(function () {
      try {
        D.downloadZip(unique, slug(name) + ".zip", state.strip);
        setStatus(null);
      } catch (e) {
        setStatus({ text: T.loadFailed(name, e.message || e), error: true, dismissible: true });
      }
    }, 30);
  }

  // ----------------------------------------------------------- URL state

  function writeHash() {
    var p = new URLSearchParams();
    if (state.selectedId) p.set("d", state.selectedId);
    if (state.query.trim()) p.set("q", state.query.trim());
    if (state.groupBy !== (state.config.defaultGroupBy || "auto")) p.set("g", state.groupBy);
    if (state.sort !== (state.config.defaultSort || "newest")) p.set("s", state.sort);
    var f = state.filters, fo = {};
    if (f.models.size) fo.m = Array.from(f.models);
    if (f.speakers.size) fo.p = Array.from(f.speakers);
    if (f.from) fo.from = f.from;
    if (f.to) fo.to = f.to;
    if (f.min !== "") fo.min = f.min;
    if (f.max !== "") fo.max = f.max;
    Object.keys(f.meta).forEach(function (k) {
      if (f.meta[k].size) (fo.x || (fo.x = {}))[k] = Array.from(f.meta[k]);
    });
    if (Object.keys(fo).length) p.set("f", JSON.stringify(fo));
    var hash = p.toString();
    var url = location.pathname + location.search + (hash ? "#" + hash : "");
    if (url !== location.pathname + location.search + location.hash) history.replaceState(null, "", url);
  }

  /** Restore view state from the URL hash. Rendering happens afterwards. */
  function readHash() {
    var p = new URLSearchParams(location.hash.slice(1));
    if (p.has("q")) { state.query = p.get("q"); $("search").value = state.query; }
    if (p.has("g")) state.groupBy = p.get("g");
    if (p.has("s") && SORTERS[p.get("s")]) state.sort = p.get("s");
    if (p.has("f")) {
      try {
        var fo = JSON.parse(p.get("f"));
        var f = emptyFilters();
        (fo.m || []).forEach(function (v) { f.models.add(v); });
        (fo.p || []).forEach(function (v) { f.speakers.add(v); });
        f.from = fo.from || ""; f.to = fo.to || "";
        f.min = fo.min !== undefined ? String(fo.min) : "";
        f.max = fo.max !== undefined ? String(fo.max) : "";
        Object.keys(fo.x || {}).forEach(function (k) { f.meta[k] = new Set(fo.x[k]); });
        state.filters = f;
      } catch (e) { /* ignore a malformed link */ }
    }
    if (p.has("d") && state.byId.has(p.get("d"))) {
      state.selectedId = p.get("d");
      $("layout").classList.add("show-detail");
    }
  }

  // ---------------------------------------------------------------- init

  function applyStaticText() {
    var c = state.config;
    document.title = c.title;
    $("app-title").textContent = c.title;
    $("app-version").textContent = c.version || "";
    var text = {
      "btn-add-files": T.addFiles, "btn-add-folder": T.addFolder, "btn-options": T.options, "btn-clear": T.clearData,
      "opt-prompts-label": T.showPrompts, "opt-mentions-label": T.highlightMentions,
      "opt-strip-label": T.stripPrompts, "opt-strip-hint": T.stripPromptsHint,
      "opt-remember-label": T.rememberData, "search-label": T.searchLabel, "search-help": T.searchHelp,
      "group-label": T.groupBy, "sort-label": T.sortBy, "btn-download-results": T.downloadResults,
      "empty-title": T.emptyTitle, "empty-body": T.emptyBody, "empty-privacy": T.emptyPrivacy,
      "btn-empty-files": T.chooseFiles, "btn-empty-folder": T.chooseFolder, "datasets-title": T.datasetsTitle,
      "drop-text": T.dropOverlay,
    };
    Object.keys(text).forEach(function (id) { $(id).textContent = text[id]; });
    $("search").placeholder = T.searchPlaceholder;
    var sort = clear($("sort-by"));
    Object.keys(T.sortOptions).forEach(function (k) { sort.appendChild(h("option", { value: k, text: T.sortOptions[k] })); });
    $("opt-prompts").checked = state.showPrompts;
    $("opt-mentions").checked = state.highlightMentions;
    $("opt-strip").checked = state.strip;
    $("opt-remember").checked = state.remember;
    $("opt-remember").closest("label").hidden = !c.rememberData;
  }

  function bindEvents() {
    $("btn-add-files").addEventListener("click", function () { $("input-files").click(); });
    $("btn-add-folder").addEventListener("click", function () { $("input-folder").click(); });
    $("btn-empty-files").addEventListener("click", function () { $("input-files").click(); });
    $("btn-empty-folder").addEventListener("click", function () { $("input-folder").click(); });
    ["input-files", "input-folder"].forEach(function (id) {
      $(id).addEventListener("change", function (e) {
        var items = D.itemsFromInput(e.target.files);
        e.target.value = "";
        loadLocal(items);
      });
    });
    $("btn-clear").addEventListener("click", clearAll);

    var optionsBtn = $("btn-options"), panel = $("options-panel");
    function setOptions(open) { panel.hidden = !open; optionsBtn.setAttribute("aria-expanded", String(open)); }
    optionsBtn.addEventListener("click", function () { setOptions(panel.hidden); });
    document.addEventListener("click", function (e) {
      if (!panel.hidden && !panel.contains(e.target) && e.target !== optionsBtn) setOptions(false);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !panel.hidden) { setOptions(false); optionsBtn.focus(); }
    });
    $("opt-prompts").addEventListener("change", function (e) { state.showPrompts = e.target.checked; renderDetail(); });
    $("opt-mentions").addEventListener("change", function (e) { state.highlightMentions = e.target.checked; renderDetail(); });
    $("opt-strip").addEventListener("change", function (e) { state.strip = e.target.checked; });
    $("opt-remember").addEventListener("change", function (e) {
      state.remember = e.target.checked;
      if (state.remember) persist(); else D.cache.clear();
    });

    var onSearch = debounce(function () { runSearch(); }, 180);
    $("search").addEventListener("input", function (e) { state.query = e.target.value; onSearch(); });
    $("group-by").addEventListener("change", function (e) { state.groupBy = e.target.value; state.limits.clear(); renderAll(); });
    $("sort-by").addEventListener("change", function (e) { state.sort = e.target.value; renderAll(); });
    $("btn-download-results").addEventListener("click", function () {
      downloadMany(view.list, state.label || "discussions");
    });

    document.addEventListener("keydown", function (e) {
      if (e.ctrlKey || e.metaKey || e.altKey || isTyping(document.activeElement)) return;
      if (!state.records.length) return;
      if (e.key === "/") { e.preventDefault(); $("search").focus(); }
      else if (e.key === "j") step(1);
      else if (e.key === "k") step(-1);
    });

    var depth = 0, overlay = $("drop-overlay");
    function hasFiles(e) { return e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], "Files") >= 0; }
    window.addEventListener("dragenter", function (e) { if (hasFiles(e)) { depth++; overlay.hidden = false; e.preventDefault(); } });
    window.addEventListener("dragover", function (e) { if (hasFiles(e)) e.preventDefault(); });
    window.addEventListener("dragleave", function () { depth = Math.max(0, depth - 1); if (!depth) overlay.hidden = true; });
    window.addEventListener("drop", function (e) {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      overlay.hidden = true;
      D.itemsFromDrop(e.dataTransfer).then(loadLocal);
    });
  }

  function fetchJson(url) {
    return fetch(url, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });
  }

  function renderDatasets() {
    var list = clear($("datasets-list"));
    $("datasets").hidden = !state.datasets.length;
    state.datasets.forEach(function (d) {
      list.appendChild(h("li", null, [h("button", {
        type: "button", class: "dataset-btn",
        onclick: function () { openDataset(d); },
      }, [h("strong", { text: d.name || d.url }), d.description ? h("span", { text: d.description }) : null])]));
    });
  }

  function openDataset(d) {
    var url = new URL(location.href);
    url.searchParams.set("data", d.url);
    url.hash = "";
    history.pushState(null, "", url.href);
    loadDataset(d.url, d.name);
  }

  function startup() {
    var params = new URLSearchParams(location.search);
    var dataUrl = params.get("data");
    if (dataUrl) {
      var known = state.datasets.find(function (d) { return d.url === dataUrl; });
      return loadDataset(dataUrl, known ? known.name : dataUrl, { restoreHash: true });
    }
    var auto = state.autoload && state.datasets.find(function (d) { return d.id === state.autoload || d.name === state.autoload; });
    if (auto) return loadDataset(auto.url, auto.name, { restoreHash: true });
    if (state.config.rememberData && state.remember) {
      return D.cache.load().then(function (saved) {
        // files opened while this was loading take precedence
        if (state.records.length || state.importing) return;
        if (saved && saved.files && saved.files.length) {
          setStatus({ text: T.restoring });
          return ingest({ files: saved.files, errors: [] }, { replace: true, label: saved.label, persist: false, restoreHash: true });
        }
        showEmptyIfNeeded();
      });
    }
    showEmptyIfNeeded();
    return Promise.resolve();
  }

  /** A yes/no URL query parameter: true/false, or undefined if absent or unrecognised. */
  function boolParam(name) {
    var raw = new URLSearchParams(location.search).get(name);
    if (raw === null) return undefined;
    raw = raw.toLowerCase();
    if (["1", "true", "on", "yes"].indexOf(raw) >= 0) return true;
    if (["0", "false", "off", "no"].indexOf(raw) >= 0) return false;
    return undefined;
  }

  function init() {
    bindEvents();
    searcher.start();
    fetchJson("config.json").catch(function () { return {}; }).then(function (cfg) {
      state.config = Object.assign({}, DEFAULT_CONFIG, cfg || {});
      state.showPrompts = !!state.config.showPrompts;
      state.strip = !!state.config.stripPromptsOnDownload;
      // "?mentions=" overrides config.json's default for this page only.
      var mentionsParam = boolParam("mentions");
      state.highlightMentions = mentionsParam !== undefined ? mentionsParam : !!state.config.highlightMentions;
      state.groupBy = state.config.defaultGroupBy || "auto";
      state.sort = SORTERS[state.config.defaultSort] ? state.config.defaultSort : "newest";
      applyStaticText();
      return fetchJson(state.config.datasetsIndex).catch(function () { return {}; });
    }).then(function (index) {
      state.datasets = Array.isArray(index && index.datasets) ? index.datasets.filter(function (d) { return d && d.url; }) : [];
      state.autoload = index && index.autoload;
      renderDatasets();
      return startup();
    });
    window.addEventListener("popstate", function () {
      var dataUrl = new URLSearchParams(location.search).get("data");
      if (dataUrl) loadDataset(dataUrl, dataUrl);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
