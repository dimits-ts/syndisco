# SynDisco Viewer: developer notes

A static web app for browsing discussion logs. It has no build step: edit
the files in `static/` and reload the page.

## Layout

| Path | Purpose |
|---|---|
| `static/index.html` | Page structure. Scripts are plain `<script>` tags. |
| `static/css/viewer.css` | All styles; light and dark colour tokens at the top. |
| `static/js/strings.js` | All user-facing text. |
| `static/js/data.js` | Reading files, zips, URLs and manifests; validation; export; remembered data (IndexedDB). |
| `static/js/search-core.js` | Full-text search, shared by the worker and the page. |
| `static/js/search-worker.js` | Runs searches off the main thread. Falls back to the page when workers are unavailable (e.g. `file://`). |
| `static/js/app.js` | The interface. |
| `static/js/vendor/` | Third-party libraries, vendored with their licences. |
| `static/config.json` | Default settings for a deployment. |
| `static/datasets/index.json` | Datasets listed on the start page (empty in the package; the Pages workflow replaces it with `viewer_datasets/`). |
| `server.py` | The local server behind `syndisco view`. Standard library only. |

## Data the viewer accepts

Any file written by `Logs.export()`. Extra keys at the top level or on
messages are shown as metadata and offered as filters and groupings for future support.
Downloads are the original bytes, unless prompts are removed, in which case
the `prompt` fields are emptied (not deleted) so SynDisco can still load the
file.

A dataset URL (`?data=...` or an `index.json` entry) can point to a `.zip`,
a single discussion `.json`, or a manifest:

```json
{"name": "My study", "files": [{"path": "exp1/a.json", "url": "files/exp1/a.json"}]}
```

Manifest URLs are relative to the manifest. Zips are faster for large
datasets because they need one request and should be preferred when viewing a multi-discussion experiment.

## `config.json`

| Key | Default | Meaning |
|---|---|---|
| `title` | `"SynDisco Viewer"` | Page title. |
| `showPrompts` | `false` | Show system prompts initially. `syndisco view --show-prompts` overrides it. |
| `stripPromptsOnDownload` | `false` | Empty prompts in downloads initially. |
| `defaultGroupBy` | `"auto"` | `auto`, `none`, `folder`, `model`, `speaker`, `day`, `month` or `meta:<key>`. |
| `defaultSort` | `"newest"` | `newest`, `oldest`, `path`, `most` or `fewest`. |
| `rememberData` | `true` | Allow reopening imported files on the next visit (stored in the browser only). |
| `pageSize` | `100` | Discussions shown per group before "Show more". |
| `datasetsIndex` | `"datasets/index.json"` | Where to find the published datasets list. |
| `version` | | Shown next to the title; filled in by `syndisco view` and the Pages workflow. |

## Testing

`tests/test_viewer_server.py` covers the server, and
`tests/test_viewer_roundtrip.py` runs `data.js` under Node.js (skipped when
Node.js is missing) to check that downloads reload with `Logs.from_file`.
For the interface, run `syndisco view <folder>` and check in a browser.

Currently untested on Node!