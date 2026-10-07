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

Run the viewer on your computer and point it at a folder of results:

.. code-block:: bash

   syndisco view path/to/output_dir

This opens the viewer in your browser with every discussion in the folder
loaded, including those in subfolders. You can also pass a single JSON file
or a ``.zip``. Without a path, the viewer opens empty and you can add files
yourself. The same command is available as ``python -m syndisco view``.

Options:

``--port PORT``
   Port to use (default 8765). If it is taken, the next free port is used.
``--no-browser``
   Start the server without opening a browser window.
``--show-prompts``
   Show participants' system prompts by default.
``--host HOST``
   Address to listen on (default ``127.0.0.1``, this computer only). Other
   addresses let other devices on your network read the served files.

The server only serves ``.json`` files inside the folder you give it, and
only to this computer by default. Stop it with Ctrl+C.

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
