# Public release audit

The release is pushed only after the following gates pass on the exact Git
commit:

1. **Scope:** the tracked tree contains source, configuration, documentation,
   schema, and tests only.
2. **Privacy:** no credentials, private absolute paths, personal application
   material, benchmark PDFs, generated predictions, or run archives.
3. **Generalization:** no query allowlists, released query identifiers, answer
   constants, previous-output inputs, or replay switches.
4. **Behavior:** the public configuration invokes full generation, strict
   safety rejects an external parent, and the fallback obeys the organizer's
   nested contract.
5. **Build:** tests and bytecode compilation pass; the CLI imports and exposes
   its help; the wheel builds and loads the vendored validator.
6. **Repository:** tracked files have no symlinks or forbidden generated
   formats, `git diff --check` passes, and `git fsck` reports a valid object
   database.

The audit command output, commit, tracked-file count, and relevant hashes are
recorded in the release commit message and GitHub repository history. API keys
and publisher PDFs are supplied locally and are never included in this tree.
