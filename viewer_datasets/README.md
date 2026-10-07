# Published datasets for SynDisco Viewer

Files in this folder are published with the online viewer at
`/viewer/datasets/` by the `pages.yml` workflow. Datasets listed in
`index.json` appear on the viewer's start page, and each can be shared as a
link: `https://<site>/viewer/?data=datasets/<file>.zip`.

To publish a dataset:

1. Zip the folder of discussion JSON files written by SynDisco. Folders
   inside the zip are kept, so experiments stay grouped.
2. Put the zip here and add an entry to `index.json`:

   ```json
   {"name": "My study", "description": "One line for readers", "url": "datasets/my-study.zip"}
   ```

3. Push to `master`.

Anyone who opens a dataset can read every file in it, including system
prompts. The viewer's prompt toggle only hides them on screen, so remove
prompts before publishing if they should stay private (open the dataset in
the viewer, turn on "Remove system prompts from downloads" in Options, and
use "Download results").

Keep datasets small: GitHub Pages sites should stay under 1 GB in total.

`example.zip` contains made-up discussions that illustrate the format.
Delete it and its `index.json` entry if you do not want it published.
