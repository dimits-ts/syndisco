# Third-party libraries

These files are vendored (copied into the repository) so the viewer works
offline and does not depend on a CDN. They are loaded with plain `<script>`
tags.

| File | Library | Version | Licence | Source |
|---|---|---|---|---|
| `fflate.min.js` | fflate (UMD build) | 0.8.3 | MIT, see `LICENSE-fflate.txt` | https://www.npmjs.com/package/fflate |

To update a library, replace the file with the UMD build of the new
version, update the table, and keep its licence file next to it.
