# PERMISSIONS.md — Smart Healthcare

Permission policy for this task. Governs everything I do in this project until you change
it. Where this file and my general defaults disagree, this file wins.

Scope: the directory `/Users/tanushreenepal/Desktop/Smart Healthcare` and the data root
`~/physionet-data` (once approved). Nothing outside those two paths, with the single
exception of creating the approved conda environment.

Date: 2026-08-22

---

## ALLOWED WITHOUT ASKING

Safe, reversible, local, and confined to the scope above.

### Reading and inspection
- Read and search any file in the project directory.
- Read `Database use.docx.pdf` (read-only; never modified).
- Inspect source code, configuration, and test files.
- Inspect `git status` / `git diff` / `git log` if a repository exists.
- Inspect the local environment read-only: `python -V`, `pip list`, `conda env list`,
  `uname`, `df -h`, `ls`, `file`.
- Inspect already-downloaded dataset files (headers, shapes, record metadata).

### Writing files required by the approved plan
- Create and modify `PLAN.md`, `PERMISSIONS.md`, `IMPLEMENTATION.md`.
- Create and modify `README.md`, `.gitignore`, `requirements.txt`.
- Create and modify source files under `src/` and `scripts/` as described in PLAN.md.
- Create and modify tests under `tests/`.
- Create and delete files under the project's own `tests/` temp output and any
  `pytest` `tmp_path` directory.
- Write processed outputs into `$SMART_HEALTH_DATA/processed/` once the data root exists.

### Running local checks
- `pytest`, including focused single-test runs.
- Linters and formatters, if and when configured for this project.
- Type checking, if and when configured.
- Short local scripts that read project files or synthetic fixtures.
- Use packages already installed in the approved environment.

Everything in this section is reversible with a file edit or deletion and touches no
external system.

---

## ASK BEFORE DOING

Meaningful side effects, external systems, or costs. I stop, explain, and wait.

- **Installing or upgrading any dependency**, including creating the `smart-health` conda
  environment and every `pip install`. *(Modifies machine state outside the project.)*
- **Changing dependency versions** or editing pins after they are set.
- **Creating or modifying lockfiles** not required by an approved dependency change.
- **`git init`** in this folder, and any commit. *(Starts version history; needs your
  say-so on what is tracked.)*
- **Downloading MIT-BIH** (~100 MB over the network) or any other dataset.
- **Creating directories outside the project**, specifically `~/physionet-data`.
- **Any network request**, including read-only ones — no silent fetching.
- **Long-running or expensive commands**, including a full 48-record preprocessing pass, if
  it is likely to run for many minutes.
- **Refactors beyond the approved scope**, or restructuring files PLAN.md does not name.
- **Adding public API surface** beyond what the task requires.
- **Changing anything about how data is stored or laid out** after you have approved a
  layout.
- **Deleting or overwriting any file I did not create in this session.**
- Modifying CI/CD, Docker, or deployment configuration. *(None exists; creating it is also
  an ask.)*
- Changing environment configuration, `.zshrc`, shell profiles, or global conda settings.
- Database schema changes or migrations. *(Not applicable now; listed for completeness.)*

When approval is required I will: (1) stop before acting, (2) state exactly what I want to
run, (3) say why it is necessary, (4) name the possible side effects, (5) wait for your
explicit yes.

---

## NEVER DO WITHOUT EXPLICIT INSTRUCTION

Destructive, irreversible, credential-related, or reaching outside this machine.

### Data and environment
- Delete or overwrite `Database use.docx.pdf`, or any file you created.
- Delete downloaded datasets or processed outputs.
- Delete, rename, or modify the `base` or `fairml312` conda environments.
- Modify the system Python or `/usr/bin` anything.
- Reset, wipe, or "clean up" any environment.
- Run `rm -rf` on anything outside a `tmp_path` I created in this session.

### Credentials and secrets
- Access, read, print, log, or echo credentials, tokens, API keys, or `.env` contents.
- Commit `.env` files or credentials.
- Create accounts on PhysioNet or anywhere else on your behalf.
- Enter, store, or transmit your PhysioNet credentials.
- **Accept, sign, or submit a Data Use Agreement, ethics attestation, or CITI training
  record on your behalf.** These are legal commitments that only you can make.
- Rotate or revoke any credential.

### Publishing and external effects
- Add a git remote, push, or force-push.
- Merge branches or pull requests.
- Publish an Artifact or any page containing project data without asking.
- Deploy anything, anywhere. There is no production system in this project and I will not
  create one.
- Modify production infrastructure or databases. *(None exist.)*
- Send email, messages, or notifications.
- Make purchases or incur billed usage.
- Upload any patient-derived data to any external service, including AI APIs. **This holds
  even for the open demo subsets.** Clinical data leaves this machine only on your explicit
  instruction.

### Testing integrity
- Weaken, skip, `xfail`, or delete a valid test to make an implementation pass.
- Remove a test without documenting why in `IMPLEMENTATION.md`.
- Claim a test passed without having actually run it and seen the output.
- Claim something is verified when it was inferred.

---

## Least Privilege Rule

- Use the narrowest action that accomplishes the step. Read one file rather than scanning
  the tree; run one test rather than the whole suite when iterating.
- Do not request broad permission when a narrow one suffices. "Install `wfdb`" is not
  "install whatever seems useful."
- **Approval does not generalize.** Permission to download MIT-BIH is not permission to
  download MIMIC-III. Permission to create one conda environment is not permission to
  modify others. Permission to run `pytest` is not permission to run arbitrary commands.
- Approval applies to the exact scope approved, once, unless you say otherwise.
- If an action is ambiguous, potentially destructive, irreversible, security-sensitive, or
  touches an external system — **ask first**.

---

## Clinical-data note

This project handles de-identified patient data. Two rules follow and are not negotiable
without your explicit instruction:

1. **De-identified is not public.** Even openly licensed PhysioNet data stays on this
   machine unless you say otherwise.
2. **Nothing built here is a medical device.** Outputs are research artifacts. I will not
   describe any result as clinically validated, diagnostic, or fit for patient care, and
   will keep that caveat in the README.
