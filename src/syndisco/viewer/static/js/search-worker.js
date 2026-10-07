/* SynDisco Viewer: runs searches off the main thread. */
importScripts("search-core.js");

var corpus = [];

self.onmessage = function (event) {
  var msg = event.data;
  if (msg.type === "index") {
    corpus = self.SDVSearch.buildCorpus(msg.docs);
    self.postMessage({ type: "indexed", seq: msg.seq });
  } else if (msg.type === "search") {
    var terms = self.SDVSearch.parseQuery(msg.query);
    self.postMessage({ type: "results", seq: msg.seq, terms: terms, results: self.SDVSearch.search(corpus, terms) });
  }
};
