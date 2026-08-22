# Poisson profiler copy-elision audit

Status: `pass`

- Runtime calls/kernel tasks: `1170`
- Memcpy before: `1235` / `1594076` bytes
- Memcpy after: `35` / `13472` bytes
- Eliminated: `1200` copies
- Max absolute numerical difference: `2.2351741790771484e-08`
- Performance published: `false`
