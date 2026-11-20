# SynDisco Viewer

HTML page export for browsing discussion logs.

## Layout

| Path | Purpose |
|---|---|
| `static/index.html` | :) |
| `static/css/viewer.css` | Styles; light and dark colour tokens at the top. |
| `static/js/strings.js` | All user-facing text. |
| `static/js/data.js` | Reading files, zips, URL, IndexedDB. |
| `static/js/search-core.js` | Full-text search, shared by the worker and the page. |
| `static/js/search-worker.js` | Runs searches off the main thread. Falls back to the page when workers are unavailable (e.g. `file://`). |
| `static/js/app.js` | UI. |
| `static/js/vendor/` | Third-party libraries. |
| `static/config.json` | Default settings for the page (see `config` section below). |
| `static/datasets/index.json` | Datasets listed on the start page (empty in the package; the Pages workflow replaces it with `viewer_datasets/`). |
| `export.py` | Builds the self-contained pages written by `syndisco view` and `syndisco export`. Standard library only. |

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

## Exported pages

`syndisco view SOURCE` and `syndisco export SOURCE TARGET` write the viewer
and a set of discussions into one HTML file (`view` to a temporary folder,
then opens it). `export.py` inlines the stylesheet, scripts and favicon
referenced by `index.html`, and adds a script before them that sets
`window.SYNDISCO_EMBEDDED = {config, label, files}`, where `files` is a
list of `{path, text}`.

When that object exists, `app.js` uses its config and files instead of
fetching `config.json`, the datasets index or a dataset, and searches
without a worker, since pages opened from disk can use neither. Without it,
the viewer behaves as a normal hosted page.

`export.py` only inlines tags written exactly as
`<link rel="stylesheet" href="...">` and `<script src="..." defer></script>`,
so keep new stylesheets and scripts in `index.html` in that form.

## `config.json`

| Key | Default | Meaning |
|---|---|---|
| `title` | `"SynDisco Viewer"` | Page title. |
| `showPrompts` | `true` | Show system prompts initially. |
| `stripPromptsOnDownload` | `false` | Empty prompts in downloads initially. |
| `defaultGroupBy` | `"auto"` | `auto`, `none`, `folder`, `model`, `speaker`, `day`, `month` or `meta:<key>`. |
| `defaultSort` | `"newest"` | `newest`, `oldest`, `path`, `most` or `fewest`. |
| `rememberData` | `true` | Allow reopening imported files on the next visit (stored in the browser only). Always off in exported pages. |
| `pageSize` | `100` | Discussions shown per group before "Show more". |
| `datasetsIndex` | `"datasets/index.json"` | Where to find the published datasets list. |
| `version` | | Shown next to the title; filled in by exported pages and the Pages workflow. |

## Testing

Currently untested on Node!