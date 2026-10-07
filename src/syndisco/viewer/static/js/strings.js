/*
 * SynDisco Viewer: user-facing text.
 * All interface text lives here so it can be reviewed, or translated
 * later, in one place.
 */
(function (global) {
  "use strict";

  function plural(n, one, many) {
    return n.toLocaleString("en") + " " + (n === 1 ? one : many);
  }

  global.SDVStrings = {
    plural: plural,
    appName: "SynDisco Viewer",

    // empty state
    emptyTitle: "Open SynDisco discussions",
    emptyBody:
      "Drop discussion JSON files, a folder of them, or a .zip anywhere on this page. " +
      "Folders keep their structure, so experiments stay grouped.",
    emptyPrivacy: "Files are read in your browser. Nothing is uploaded.",
    chooseFiles: "Choose files",
    chooseFolder: "Choose folder",
    datasetsTitle: "Published datasets",
    dropOverlay: "Drop to open",

    // header
    addFiles: "Add files",
    addFolder: "Add folder",
    clearData: "Close all",
    clearConfirm: "Close all discussions? Files on your computer are not affected.",
    options: "Options",
    showPrompts: "Show system prompts",
    highlightMentions: "Highlight participant names mentioned in messages",
    stripPrompts: "Remove system prompts from downloads",
    stripPromptsHint: "Downloads keep the prompt field, left empty, so files still load in SynDisco.",
    rememberData: "Reopen these discussions next time",
    loadedFrom: "Dataset",
    localData: "Your files",

    // sidebar
    searchLabel: "Search",
    searchPlaceholder: "Search messages, speakers, folders",
    searchHelp: 'All words must appear. Use "quotes" for exact phrases. Press / to search.',
    groupBy: "Group by",
    sortBy: "Sort",
    groupOptions: {
      auto: "Automatic",
      none: "No grouping",
      folder: "Folder",
      model: "Model",
      speaker: "Participant",
      day: "Day",
      month: "Month",
    },
    metaGroupPrefix: "Metadata: ",
    sortOptions: {
      newest: "Newest first",
      oldest: "Oldest first",
      path: "File path",
      most: "Most messages",
      fewest: "Fewest messages",
    },
    filters: "Filters",
    filtersActive: function (n) { return n ? "Filters (" + n + " active)" : "Filters"; },
    clearFilters: "Clear filters",
    filterModels: "Model",
    filterSpeakers: "Participant",
    filterSpeakersSearch: "Find a participant",
    filterDates: "Date",
    filterFrom: "From",
    filterTo: "To",
    filterLength: "Number of messages",
    filterMin: "At least",
    filterMax: "At most",
    downloadResults: "Download results",
    download: "Download",
    downloadGroup: "Download this group",
    resultsCount: function (shown, total) {
      return shown === total
        ? plural(total, "discussion", "discussions")
        : shown.toLocaleString("en") + " of " + plural(total, "discussion", "discussions");
    },
    noResults: "No discussions match your search and filters.",
    showMore: function (n) { return "Show " + n.toLocaleString("en") + " more"; },
    noValue: "No value",
    topLevel: "Top level",
    unknownDate: "Unknown date",
    matchingMessages: function (n) { return plural(n, "matching message", "matching messages"); },
    matchedDetails: "Matched in file name, speakers or metadata",

    // discussion view
    back: "Back to list",
    previous: "Previous",
    next: "Next",
    downloadJson: "Download JSON",
    messages: function (n) { return plural(n, "message", "messages"); },
    speakers: function (n) { return plural(n, "participant", "participants"); },
    participants: "Participants",
    metadata: "Metadata",
    seedComment: "Seed comment",
    seedModelLabel: "Seed comment, not generated",
    promptLabel: "System prompt",
    promptsHidden: "System prompts are hidden. Turn them on in Options.",
    noPrompt: "No system prompt recorded.",
    multiplePrompts: function (n) { return "This participant used " + n + " different prompts."; },
    timelineLabel: "Turn order. Select a position to jump to that message.",
    timelineHits: function (n) {
      return "Turn order. Marks below show " + plural(n, "matching message", "matching messages") + "; select one to jump to it.";
    },
    transcript: "Messages",
    messageOf: function (i, n) { return "Message " + i + " of " + n; },
    noSelection: "Select a discussion from the list.",
    noModel: "Model not recorded",

    // import
    reading: function (done, total) {
      return "Reading " + done.toLocaleString("en") + " of " + plural(total, "file", "files");
    },
    downloading: "Downloading dataset",
    preparingDownload: "Preparing download",
    loaded: function (n) { return "Opened " + plural(n, "discussion", "discussions") + "."; },
    skipped: function (n) { return plural(n, "file was", "files were") + " skipped."; },
    showDetails: "Show details",
    dismiss: "Dismiss",
    restoring: "Reopening your previous discussions",
    loadFailed: function (what, why) { return "Could not open " + what + ": " + why; },
    nothingFound: "No discussion files were found. SynDisco Viewer reads .json files and .zip archives.",

    // validation reasons
    errNotJson: "not valid JSON",
    errNoLogs: 'missing the "logs" list',
    errLogsNotList: '"logs" is not a list',
    errBadEntry: function (i) { return "message " + (i + 1) + " needs text fields \"name\" and \"text\""; },
    errNoModel: 'some messages have no "model" field, so SynDisco cannot reload this file',
    errLegacyFormat: function (keys) {
      return "This looks like an older SynDisco export (" + keys.join(", ") +
        "), not today's format. Participants are shown, but system prompts are not. " +
        "Convert it first with scripts/migrate_legacy_logs.py.";
    },
    errDuplicate: "a file with the same path was already open; this one was renamed",
  };
})(typeof self !== "undefined" ? self : this);
