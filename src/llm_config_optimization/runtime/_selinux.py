"""Detection of SELinux enforcement for container runs.

The probe reads the kernel's SELinux enforce flag and is deliberately
injectable: the flag's path is a parameter, so callers and tests can point it at
a temporary file instead of the real host. It imports no container backend, so
the check stays usable by any runtime.
"""

from pathlib import Path

#: The kernel flag that holds ``1`` while SELinux is enforcing and ``0`` while
#: it is permissive. It is absent entirely when SELinux is disabled.
_ENFORCE_FLAG = Path("/sys/fs/selinux/enforce")


def selinux_enforcing(enforce_flag: Path = _ENFORCE_FLAG) -> bool:
    """Report whether the host is running SELinux in enforcing mode.

    The flag holds ``1`` when enforcing and ``0`` when permissive; a missing or
    unreadable flag means SELinux is disabled or unavailable, which this probe
    reports as not enforcing. That makes the check safe to call on any host and
    lets callers treat it as a plain boolean capability.
    """
    try:
        return enforce_flag.read_text().strip() == "1"
    except OSError:
        return False
