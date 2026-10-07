/*
 * SynDisco Viewer: reading, validating and exporting discussion files.
 *
 * A discussion file is what syndisco's Logs.export() writes:
 *   { "timestamp": "...", "logs": [ {name, text, model, prompt}, ... ] }
 * Any other keys, at the top level or on messages, are kept and shown as
 * metadata. The original file text is kept untouched so downloads are
 * byte-for-byte identical unless prompts are removed.
 */
(function (global) {
  "use strict";

  var T = global.SDVStrings;
  var CORE_ENTRY_KEYS = { name: 1, text: 1, model: 1, prompt: 1 };
  var CORE_TOP_KEYS = { logs: 1, timestamp: 1 };
  var SEED_MODEL = "hardcoded";

  // ---------------------------------------------------------------- dates

  function pad(n) { return (n < 10 ? "0" : "") + n; }

  /** Parse syndisco's "%y-%m-%d-%H-%M(-%S)" format, or ISO dates. */
  function parseTimestamp(value) {
    if (typeof value !== "string" || !value) return null;
    var m = /^(\d{2}|\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})(?:-(\d{2}))?$/.exec(value.trim());
    if (m) {
      var year = m[1].length === 2 ? 2000 + Number(m[1]) : Number(m[1]);
      var d = new Date(year, Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]), Number(m[6] || 0));
      return isNaN(d.getTime()) ? null : d;
    }
    if (/^\d{4}-\d{2}-\d{2}/.test(value)) {
      var iso = new Date(value);
      return isNaN(iso.getTime()) ? null : iso;
    }
    return null;
  }

  function formatDate(d) {
    if (!d) return T.unknownDate;
    return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) +
      " " + pad(d.getHours()) + ":" + pad(d.getMinutes());
  }

  function dayKey(d) { return d ? d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) : ""; }
  function monthKey(d) { return d ? d.getFullYear() + "-" + pad(d.getMonth() + 1) : ""; }

  // ---------------------------------------------------------------- paths

  function normalizePath(path) {
    return String(path || "").replace(/\\/g, "/").replace(/^\.?\/+/, "");
  }

  function splitPath(path) {
    var i = path.lastIndexOf("/");
    return i < 0 ? { folder: "", fileName: path } : { folder: path.slice(0, i), fileName: path.slice(i + 1) };
  }

  function stem(fileName) { return fileName.replace(/\.json$/i, ""); }

  function metaValueLabel(value) {
    if (value === null || value === undefined) return "";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  // ---------------------------------------------------------------- parse

  /**
   * Turn one file's text into a discussion record.
   * Returns {record} or {error}.
   */
  function parseDiscussion(path, text) {
    var obj;
    try {
      obj = JSON.parse(text);
    } catch (e) {
      return { error: T.errNotJson };
    }
    if (!obj || typeof obj !== "object" || Array.isArray(obj) || !("logs" in obj)) {
      return { error: T.errNoLogs };
    }
    if (!Array.isArray(obj.logs)) return { error: T.errLogsNotList };

    var messages = [];
    var speakers = [];
    var speakerIndex = {};
    var models = {};
    var prompts = {};
    var warnings = [];
    var missingModel = false;
    var words = 0;

    for (var i = 0; i < obj.logs.length; i++) {
      var e = obj.logs[i];
      if (!e || typeof e !== "object" || typeof e.name !== "string" || typeof e.text !== "string") {
        return { error: T.errBadEntry(i) };
      }
      var model = typeof e.model === "string" ? e.model : "";
      if (!("model" in e)) missingModel = true;
      var isSeed = model === SEED_MODEL;
      var extra = null;
      for (var k in e) {
        if (Object.prototype.hasOwnProperty.call(e, k) && !CORE_ENTRY_KEYS[k]) {
          (extra || (extra = {}))[k] = e[k];
        }
      }
      if (!(e.name in speakerIndex)) {
        speakerIndex[e.name] = speakers.length;
        speakers.push(e.name);
      }
      if (model && !isSeed) models[model] = true;
      if (typeof e.prompt === "string" && e.prompt) {
        var list = prompts[e.name] || (prompts[e.name] = []);
        if (list.indexOf(e.prompt) < 0) list.push(e.prompt);
      }
      words += e.text.split(/\s+/).filter(Boolean).length;
      messages.push({
        name: e.name,
        text: e.text,
        model: model,
        isSeed: isSeed,
        speaker: speakerIndex[e.name],
        extra: extra,
      });
    }
    if (missingModel) warnings.push(T.errNoModel);

    var meta = {};
    for (var key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key) && !CORE_TOP_KEYS[key]) meta[key] = obj[key];
    }

    var parts = splitPath(path);
    var date = parseTimestamp(obj.timestamp) || parseTimestamp(stem(parts.fileName));

    var speakerStats = speakers.map(function (name) {
      var count = 0, model = "", seedOnly = true;
      messages.forEach(function (m) {
        if (m.name !== name) return;
        count++;
        if (!m.isSeed) { seedOnly = false; if (m.model) model = m.model; }
      });
      return { name: name, count: count, model: model, seedOnly: seedOnly, prompts: prompts[name] || [] };
    });

    return {
      record: {
        id: path,
        path: path,
        folder: parts.folder,
        fileName: parts.fileName,
        title: stem(parts.fileName),
        raw: text,
        timestampRaw: typeof obj.timestamp === "string" ? obj.timestamp : "",
        date: date,
        dateLabel: formatDate(date),
        messages: messages,
        speakers: speakers,
        speakerStats: speakerStats,
        models: Object.keys(models).sort(),
        meta: meta,
        hasPrompts: Object.keys(prompts).length > 0,
        words: words,
        warnings: warnings,
      },
    };
  }

  // -------------------------------------------------------------- export

  /** File text for download; optionally with every prompt emptied. */
  function exportText(record, stripPrompts) {
    if (!stripPrompts) return record.raw;
    var obj = JSON.parse(record.raw);
    obj.logs.forEach(function (e) {
      if (e && typeof e === "object" && "prompt" in e) e.prompt = "";
    });
    return JSON.stringify(obj, null, 4);
  }

  function downloadBlob(blob, fileName) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 10000);
  }

  function downloadRecord(record, stripPrompts) {
    var blob = new Blob([exportText(record, stripPrompts)], { type: "application/json" });
    downloadBlob(blob, record.fileName);
  }

  function downloadZip(records, zipName, stripPrompts) {
    var files = {};
    records.forEach(function (r) {
      files[r.path] = [global.fflate.strToU8(exportText(r, stripPrompts)), { level: 6 }];
    });
    var bytes = global.fflate.zipSync(files);
    downloadBlob(new Blob([bytes], { type: "application/zip" }), zipName);
  }

  // ------------------------------------------------------------- loading

  function isJson(name) { return /\.json$/i.test(name); }
  function isZip(name) { return /\.zip$/i.test(name); }

  function isHidden(path) {
    return path.split("/").some(function (p) { return p.charAt(0) === "." || p === "__MACOSX"; });
  }

  /** Expand zip bytes into [{path, text}] for every JSON file inside. */
  function unzipJson(bytes, prefix) {
    var entries = global.fflate.unzipSync(bytes, {
      filter: function (f) { return isJson(f.name) && !isHidden(normalizePath(f.name)); },
    });
    var out = [];
    Object.keys(entries).sort().forEach(function (name) {
      var path = normalizePath(name);
      out.push({ path: prefix ? prefix + "/" + path : path, text: global.fflate.strFromU8(entries[name]) });
    });
    return out;
  }

  function readAsText(file) {
    return file.text ? file.text() : new Response(file).text();
  }

  function readAsBytes(file) {
    return (file.arrayBuffer ? file.arrayBuffer() : new Response(file).arrayBuffer())
      .then(function (buf) { return new Uint8Array(buf); });
  }

  /**
   * Read local files. items: [{file, path}]. Zips are expanded; their
   * files are placed under a folder named after the zip when more than
   * one item is opened. Calls onProgress(done, total).
   */
  function readLocalItems(items, onProgress) {
    var relevant = items.filter(function (it) {
      return !isHidden(it.path) && (isJson(it.path) || isZip(it.path));
    });
    var ignored = items.length - relevant.length;
    var zipCount = relevant.filter(function (it) { return isZip(it.path); }).length;
    var out = [];
    var errors = [];
    var done = 0;

    return runLimited(relevant, 16, function (it) {
      var p;
      if (isZip(it.path)) {
        var prefix = relevant.length > 1 || zipCount > 1 ? it.path.replace(/\.zip$/i, "") : "";
        p = readAsBytes(it.file).then(function (bytes) {
          unzipJson(bytes, prefix).forEach(function (f) { out.push(f); });
        });
      } else {
        p = readAsText(it.file).then(function (text) { out.push({ path: it.path, text: text }); });
      }
      return p.catch(function (e) {
        errors.push({ path: it.path, reason: String((e && e.message) || e) });
      }).then(function () {
        done++;
        if (onProgress) onProgress(done, relevant.length);
      });
    }).then(function () {
      return { files: out, errors: errors, ignored: ignored };
    });
  }

  /** Run fn over items with at most `limit` promises in flight. */
  function runLimited(items, limit, fn) {
    var i = 0;
    function next() {
      if (i >= items.length) return Promise.resolve();
      var item = items[i++];
      return fn(item).then(next);
    }
    var workers = [];
    for (var w = 0; w < Math.min(limit, items.length); w++) workers.push(next());
    return Promise.all(workers);
  }

  /** Walk dropped folders (DataTransferItem entries) into [{file, path}]. */
  function itemsFromDrop(dataTransfer) {
    var entries = [];
    if (dataTransfer.items && dataTransfer.items.length && dataTransfer.items[0].webkitGetAsEntry) {
      for (var i = 0; i < dataTransfer.items.length; i++) {
        var entry = dataTransfer.items[i].webkitGetAsEntry();
        if (entry) entries.push(entry);
      }
    }
    if (!entries.length) {
      return Promise.resolve(Array.prototype.map.call(dataTransfer.files || [], function (f) {
        return { file: f, path: f.name };
      }));
    }
    var out = [];
    function walk(entry, prefix) {
      var path = prefix ? prefix + "/" + entry.name : entry.name;
      if (entry.isFile) {
        return new Promise(function (resolve) {
          entry.file(function (f) { out.push({ file: f, path: path }); resolve(); }, function () { resolve(); });
        });
      }
      if (entry.isDirectory) {
        var reader = entry.createReader();
        return new Promise(function (resolve) {
          var all = [];
          (function readBatch() {
            reader.readEntries(function (batch) {
              if (!batch.length) {
                resolve(Promise.all(all.map(function (c) { return walk(c, path); })));
              } else {
                all = all.concat(Array.prototype.slice.call(batch));
                readBatch();
              }
            }, function () { resolve(); });
          })();
        });
      }
      return Promise.resolve();
    }
    return Promise.all(entries.map(function (e) { return walk(e, ""); })).then(function () { return out; });
  }

  function itemsFromInput(fileList) {
    return Array.prototype.map.call(fileList, function (f) {
      return { file: f, path: normalizePath(f.webkitRelativePath || f.name) };
    });
  }

  function fetchOk(url) {
    return fetch(url, { credentials: "same-origin" }).then(function (res) {
      if (!res.ok) throw new Error("the server answered " + res.status + " " + res.statusText);
      return res;
    });
  }

  function baseName(url) {
    try {
      var p = new URL(url, location.href).pathname;
      return decodeURIComponent(p.slice(p.lastIndexOf("/") + 1)) || "data.json";
    } catch (e) {
      return "data.json";
    }
  }

  /**
   * Load a dataset from a URL. It can be:
   *  - a .zip of JSON files,
   *  - a single discussion JSON file,
   *  - a manifest: {"name": "...", "files": [{"path": "...", "url": "..."}]}
   *    with URLs relative to the manifest.
   */
  function loadUrl(url, onProgress) {
    var absolute = new URL(url, location.href).href;
    return fetchOk(absolute).then(function (res) {
      return res.arrayBuffer();
    }).then(function (buf) {
      var bytes = new Uint8Array(buf);
      if (bytes[0] === 0x50 && bytes[1] === 0x4b) {
        return { files: unzipJson(bytes, ""), errors: [], ignored: 0 };
      }
      var text = new TextDecoder("utf-8").decode(bytes);
      var obj;
      try { obj = JSON.parse(text); } catch (e) { throw new Error(T.errNotJson); }
      if (obj && Array.isArray(obj.files) && !("logs" in obj)) {
        return loadManifest(obj, absolute, onProgress);
      }
      return { files: [{ path: baseName(absolute), text: text }], errors: [], ignored: 0 };
    });
  }

  function loadManifest(manifest, manifestUrl, onProgress) {
    var entries = manifest.files.filter(function (f) { return f && typeof f.url === "string"; });
    var out = new Array(entries.length);
    var errors = [];
    var done = 0;
    return runLimited(entries.map(function (f, i) { return { f: f, i: i }; }), 8, function (it) {
      var path = normalizePath(it.f.path || baseName(it.f.url));
      return fetchOk(new URL(it.f.url, manifestUrl).href)
        .then(function (res) { return res.text(); })
        .then(function (text) { out[it.i] = { path: path, text: text }; })
        .catch(function (e) { errors.push({ path: path, reason: String(e.message || e) }); })
        .then(function () { done++; if (onProgress) onProgress(done, entries.length); });
    }).then(function () {
      return { files: out.filter(Boolean), errors: errors, ignored: 0 };
    });
  }

  // --------------------------------------------------- remembered data

  var DB_NAME = "syndisco-viewer";
  var STORE = "collection";

  function openDb() {
    return new Promise(function (resolve, reject) {
      if (!global.indexedDB) { reject(new Error("IndexedDB unavailable")); return; }
      var req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = function () { req.result.createObjectStore(STORE); };
      req.onsuccess = function () { resolve(req.result); };
      req.onerror = function () { reject(req.error); };
    });
  }

  function withStore(mode, fn) {
    return openDb().then(function (db) {
      return new Promise(function (resolve, reject) {
        var tx = db.transaction(STORE, mode);
        var result = fn(tx.objectStore(STORE));
        tx.oncomplete = function () { db.close(); resolve(result && result.result); };
        tx.onerror = function () { db.close(); reject(tx.error); };
        tx.onabort = function () { db.close(); reject(tx.error); };
      });
    });
  }

  var cache = {
    save: function (files, label) {
      return withStore("readwrite", function (s) {
        return s.put({ files: files, label: label, savedAt: Date.now() }, "current");
      }).catch(function () {});
    },
    load: function () {
      return withStore("readonly", function (s) { return s.get("current"); }).catch(function () { return null; });
    },
    clear: function () {
      return withStore("readwrite", function (s) { return s.delete("current"); }).catch(function () {});
    },
  };

  global.SDVData = {
    parseDiscussion: parseDiscussion,
    parseTimestamp: parseTimestamp,
    formatDate: formatDate,
    dayKey: dayKey,
    monthKey: monthKey,
    metaValueLabel: metaValueLabel,
    normalizePath: normalizePath,
    exportText: exportText,
    downloadRecord: downloadRecord,
    downloadZip: downloadZip,
    readLocalItems: readLocalItems,
    itemsFromDrop: itemsFromDrop,
    itemsFromInput: itemsFromInput,
    loadUrl: loadUrl,
    cache: cache,
  };
})(typeof self !== "undefined" ? self : this);
