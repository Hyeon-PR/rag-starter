# Changelog (recent)

We ship updates every other Tuesday. Highlights from the last six months:

## v2.7 — March 2026

- **Habit pairs** rebuilt. Previously a separate stats page; now woven into each habit's detail view. Pro feature.
- Web app dark mode (finally).
- Bugfix: streak reset at the wrong time when traveling east across the date line.

## v2.6 — February 2026

- **Less habits** (formerly "inverse habits") promoted out of beta. Flag any habit as a "less" habit and check-ins reverse.
- Apple Watch complication for today's habit count.
- Removed "weekly recap" email — open rates were below 8% and users said it didn't tell them anything they didn't already see in the app.

## v2.5 — January 2026

- **Reminder budget** introduced (3 per habit per day, 12 total per day). See [Reminders](03-reminders.md).
- Streak freezes can now be disabled per habit (previously global).
- Web app keyboard shortcuts: `j`/`k` to move through habits, `space` to toggle today's check-in.

## v2.4 — December 2025

- **Year-in-review** PDF for Pro users.
- New password reset flow (existing one had a known issue with email aliases).

## v2.3 — November 2025

- Android: fixed a crash on launch for users with > 200 archived habits.
- Settings → Account → Export now includes notes (was previously skipping them).

## Roadmap (not committed dates)

- **Pair-tracking** — a fresh take on shared habits with a different social shape than "we both check in". Concept work, not yet engineered.
- **Apple Health / Google Fit** read integration — auto-check-in based on activity data (e.g., a "walk 30 min" habit). Privacy considerations are still being worked through.
- **Apple Watch streak detail view** — currently watch only shows the count, not the streak. The detail page exists in design but the watch screen is small enough that we're not sure it's a good idea.
