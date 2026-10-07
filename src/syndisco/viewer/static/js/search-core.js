/*
 * SynDisco Viewer: full-text search.
 * Shared by the search worker and by the main page (used directly when
 * workers are unavailable, e.g. when index.html is opened from disk).
 *
 * Every word (or "quoted phrase") must appear somewhere in a discussion:
 * in a message, a speaker name, the file path or a metadata value.
 * Matching is case-insensitive substring matching.
 */
(function (global) {
  "use strict";

  function parseQuery(query) {
    var terms = [];
    var re = /"([^"]*)"|(\S+)/g;
    var m;
    while ((m = re.exec(String(query || ""))) !== null) {
      var term = (m[1] !== undefined ? m[1] : m[2]).trim().toLowerCase();
      if (term && terms.indexOf(term) < 0) terms.push(term);
    }
    return terms;
  }

  /** docs: [{id, fields: string, messages: [string]}] -> lowercased corpus */
  function buildCorpus(docs) {
    return docs.map(function (d) {
      return {
        id: d.id,
        fields: String(d.fields || "").toLowerCase(),
        messages: d.messages.map(function (t) { return String(t).toLowerCase(); }),
      };
    });
  }

  /** Returns [{id, hits: [message indices containing any term]}]. */
  function search(corpus, terms) {
    var results = [];
    if (!terms.length) return results;
    for (var d = 0; d < corpus.length; d++) {
      var doc = corpus[d];
      var hits = [];
      var all = true;
      for (var t = 0; t < terms.length && all; t++) {
        var term = terms[t];
        var found = doc.fields.indexOf(term) >= 0;
        for (var i = 0; i < doc.messages.length; i++) {
          if (doc.messages[i].indexOf(term) >= 0) {
            found = true;
            if (hits.indexOf(i) < 0) hits.push(i);
          }
        }
        if (!found) all = false;
      }
      if (all) results.push({ id: doc.id, hits: hits.sort(function (a, b) { return a - b; }) });
    }
    return results;
  }

  global.SDVSearch = { parseQuery: parseQuery, buildCorpus: buildCorpus, search: search };
})(typeof self !== "undefined" ? self : this);
