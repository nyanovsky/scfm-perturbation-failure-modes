# Jobs

There is no scheduler here — runs 1–4 used Silico's on-demand H100 jobs, and this machine has
one RTX 3090 and a shell. So a "job" is a script run in the foreground, or under `nohup` when
it is long enough to outlive a session.

Each model runs in **its own conda env** (see `../envs/README.md`). They are not
interchangeable: the three checkpoints pin three different torch builds, and the whole point
of subclassing run 2's adapters rather than rewriting them is that a number produced here is
produced by the same code that produced run 2's.

Order matters. `smoke.sh` first, always: it is what turns C1–C3 from claims into checks, and
a scPRINT that silently kept a randomly initialised gene embedding will import, load and
produce beautifully-shaped noise. Run the smoke, read the printed numbers, and only then
spend hours on the sweep.
