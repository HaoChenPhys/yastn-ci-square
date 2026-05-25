"""Run a heavy function in a child process and kill it before it starves RAM.

The modular-commutator replica contractions can blow up peak memory and crash
the machine (this box has only ~8 GB). `run_with_memory_cap` runs the work in a
forked child and watches two signals via psutil:

  * **system available memory** — the real machine-protection signal. macOS
    compresses idle pages so per-process RSS badly undercounts; watching
    `virtual_memory().available` and killing before it hits a floor reliably
    prevents swap-death regardless of accounting quirks.
  * **child RSS budget** — a secondary per-process cap.

If either trips, the child is killed and `MemoryBudgetExceeded` is raised, so an
over-large (n, L, χ, slice) setting fails gracefully and the benchmark loop can
back off (smaller slices / window) instead of taking the whole box down.

Usage:
    from _memguard import run_with_memory_cap, MemoryBudgetExceeded
    try:
        result, info = run_with_memory_cap(fn, *args, min_avail_gb=0.8,
                                           mem_gb=4.0, **kwargs)
    except MemoryBudgetExceeded as e:
        ...  # shrink slice size and retry

Uses the "fork" start method so the child inherits already-loaded objects
(e.g. the CTM environment) with no pickling; only the return value crosses the
Queue, so it must be picklable (a number / small dict / ndarray is fine).
"""
import multiprocessing as mp
import time
import traceback

try:
    import psutil
except ModuleNotFoundError as _e:  # fail LOUD — never silently bypass the guard
    raise ModuleNotFoundError(
        "_memguard requires psutil, which lives in the 'torch_peps' conda env. "
        "Run with the explicit interpreter "
        "/opt/homebrew/Caskroom/miniforge/base/envs/torch_peps/bin/python3 "
        "(the base env / bare `python3` has no psutil, which would silently "
        "disable memory protection)."
    ) from _e


class MemoryBudgetExceeded(MemoryError):
    pass


def _child(func, args, kwargs, q):
    try:
        q.put(("ok", func(*args, **kwargs)))
    except Exception as e:  # noqa: BLE001
        q.put(("err", f"{e!r}\n{traceback.format_exc()}"))


def _proc_rss(proc):
    """RSS of `proc` and its children, in bytes (0 if it vanished)."""
    try:
        total = proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return 0


def run_with_memory_cap(func, *args, min_avail_gb=0.8, mem_gb=None,
                        poll=0.05, **kwargs):
    """Run ``func`` in a child; kill it if system RAM runs low or RSS > budget.

    Args:
        min_avail_gb: kill if system available memory drops below this (machine
            protection — the primary, reliable signal on macOS).
        mem_gb: optional hard per-child RSS budget in GB (secondary).
        poll: sampling interval (s).

    Returns ``(result, info)`` with ``info = {peak_rss_gb, min_avail_gb}``.
    Raises :class:`MemoryBudgetExceeded` if a limit tripped, or ``RuntimeError``
    if the child raised / died.
    """
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(func, args, kwargs, q))
    p.start()
    proc = psutil.Process(p.pid)
    avail_floor = min_avail_gb * 2**30
    rss_cap = mem_gb * 2**30 if mem_gb else float("inf")
    peak_rss = 0
    min_avail = float("inf")
    reason = None
    try:
        while p.is_alive():
            avail = psutil.virtual_memory().available
            rss = _proc_rss(proc)
            peak_rss = max(peak_rss, rss)
            min_avail = min(min_avail, avail)
            if avail < avail_floor:
                reason = f"system available RAM {avail/2**30:.2f} GB < {min_avail_gb} GB"
            elif rss > rss_cap:
                reason = f"child RSS {rss/2**30:.2f} GB > {mem_gb} GB"
            if reason:
                p.terminate()
                break
            time.sleep(poll)
    finally:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join()
    info = {"peak_rss_gb": peak_rss / 2**30,
            "min_avail_gb": (0.0 if min_avail == float("inf") else min_avail / 2**30)}
    if reason:
        raise MemoryBudgetExceeded(f"killed: {reason} (info={info})")
    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            return payload, info
        raise RuntimeError(f"child raised:\n{payload}")
    raise RuntimeError(f"child exited without a result (exitcode={p.exitcode}); "
                       f"likely OOM-killed by the OS (info={info})")


if __name__ == "__main__":
    def _light():
        return 6 * 7

    def _active_hog(target_mb):
        import numpy as np
        c = []
        for _ in range(target_mb):
            a = np.ones(1024 * 1024 // 8)
            a += 1.0
            c.append(a)
        return len(c)

    print("light:", run_with_memory_cap(_light, min_avail_gb=0.5))
    try:
        # grab up to 16 GB on an 8 GB box -> must be killed on the avail floor
        run_with_memory_cap(_active_hog, 16384, min_avail_gb=1.0, poll=0.02)
        print("ERROR: hog was not killed")
    except MemoryBudgetExceeded as e:
        print("OK killed:", e)
