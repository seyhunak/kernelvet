# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report them
privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
("Report a vulnerability" on the repository's Security tab).

Please include:

* the affected version (`kernelvet --version`),
* the steps to reproduce,
* the impact you expect.

We aim to acknowledge reports within 7 days.

## Supported versions

Only the latest released version receives security fixes.

## Security model

KernelVet runs external tools and builds and runs other people's code. It
is designed as follows:

* It never requires root and never calls `sudo`. It never changes drivers,
  kernel modules or system configuration.
* External commands are run as argv lists (`shell=False`), each with a timeout.
  The environment is passed through a scrubber that drops variables whose names
  look like secrets.
* Command output, logs and reports are redacted for tokens, keys, passwords and
  credentials embedded in URLs. The host name is stored only as a hash.
* Output paths are validated. A migration never writes into, over, or around
  the original CUDA project. Symlinks are not followed out of the project tree.
* `migrate build` and `migrate test` compile and **run code from the project
  you point them at**. Treat untrusted projects accordingly, and use
  `isolation = "docker"`. The container has:
  * no network,
  * every Linux capability dropped and `no-new-privileges`,
  * PID and memory limits,
  * GPU device nodes only for GPU runs.

Issues that break any of the guarantees above are security issues.
