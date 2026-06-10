# Incident Response

Acme uses a **three-tier severity system** for production incidents.

- **SEV-1**: customer-impacting outage, data loss, or security breach. Page
  the on-call immediately; declare in #incidents channel; commander assigned
  within 5 minutes. Customer comms within 30 minutes.
- **SEV-2**: degraded service or partial outage. Notify on-call within 15
  minutes; mitigate within 1 hour; written summary in 24 hours.
- **SEV-3**: minor issues with workarounds. Open a ticket; address during
  business hours.

Every SEV-1 and SEV-2 incident requires a **postmortem** within 5 business
days. Postmortems are blameless, public to the company, and follow the
template in the Eng Handbook.

The on-call rotation is owned by the platform team. Page via PagerDuty.
After-hours pages carry a $200 per-page stipend.
