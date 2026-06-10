# Code Review Process

All non-trivial changes at Acme require **at least one approval** from a
code owner before merging. Trivial changes (typo fixes, dependency bumps
with green CI) may be self-merged after CI passes.

Code review SLAs:
- First response within **one business day**.
- Reviewers should leave one of: approval, requested changes, or a clear
  "I will get to this on \<date\>".

CODEOWNERS files in each repo route reviews to the right team. If no
code owner is available within SLA, the PR author may request a substitute
in #eng-help.

Large PRs (\>500 lines of diff) should be split. If a split isn't feasible,
schedule a 30-minute walk-through with the reviewer before posting.

Bypass: SEV-1 hotfixes may be merged with a one-line PR description and
post-hoc review within 24 hours.
