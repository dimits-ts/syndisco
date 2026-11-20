SynDisco Viewer
===============

SynDisco Viewer shows the discussions SynDisco produces in a web browser.
You can search every message, filter and group discussions, read each
transcript with its participants and metadata, and download discussions
again as JSON files that SynDisco can load.

Files are read in your browser and are never uploaded.

`Open SynDisco Viewer <viewer/index.html>`_


Viewing your own results
------------------------

Open a folder of results in the viewer:

.. code-block:: bash

   syndisco view path/to/output_dir

Writes a temporary page with every discussion in the folder,
including subfolders, and opens it in a browser. You can also pass a
single JSON file or a ``.zip``. The page stays in your system's temporary
folder until the system clears it.

To keep the page or share it, save it to a file instead:

.. code-block:: bash

   syndisco export path/to/output_dir results.html

An existing file with that name is overwritten. The page is a single,
self-contained file that works offline, so it can be sent to others or
attached to a paper. It shows the discussions as they were when you ran
the command; run it again to include new results. It contains the full
logs, including system prompts, even when they are hidden on screen, so
check them before sharing the file.

Both commands are also available as ``python -m syndisco view`` and
``python -m syndisco export``.

You can also open the online viewer and drag a results folder, JSON files
or a zip onto the page.


Using the viewer
----------------

- **Search** finds discussions where every word appears, in messages,
  participant names, folder names or metadata. Put phrases in quotes.
  Matches are highlighted and marked on the turn-order strip.
- **Group by** organises the list by folder, model, participant, day,
  month, or any extra key in your files.
- **Filters** narrow the list by model, participant, date and length.
- **Download JSON** saves one discussion. **Download results** saves
  everything that matches the current search and filters as a zip, and
  each group has its own download.
- The address bar keeps your search, filters and selected discussion, so
  you can share a link to exactly what you see.

Keyboard: ``/`` focuses search, ``j`` and ``k`` move to the next and
previous discussion, and the arrow keys move along the turn-order strip.


System prompts
--------------

Participants' system prompts are hidden by default; turn them on in
**Options**. Hiding them only affects the screen. To share files without
prompts, turn on **Remove system prompts from downloads** in **Options**.
Downloaded files keep an empty ``prompt`` field so SynDisco still loads
them.


Extra information in your files
-------------------------------

The viewer reads files written by ``Logs.export()``. Any additional key
you add to these files, at the top level (for example ``"experiment"``) or
on individual messages, is shown as metadata and can be used for filtering
and grouping.


Publishing datasets
-------------------

Datasets in the repository's ``viewer_datasets`` folder are published with
the online viewer and listed on its start page. See the ``README.md`` in
that folder for the steps. A published dataset can be shared as a link of
the form ``viewer/?data=datasets/my-study.zip``.