"""Opt-in startup memory observations and a separately gated cache release."""
import json
import os
import time
from pathlib import Path


def observe_startup_memory(logger, stage):
    if os.environ.get('DSV41_MEMORY_DIAGNOSTICS') != '1':
        return
    import torch

    try:
        stats = torch.cuda.memory_stats()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        proc_keys = {'VmRSS', 'VmSwap', 'VmLck', 'VmPin', 'RssAnon'}
        mem_keys = {'MemAvailable', 'MemFree', 'Cached', 'SwapCached', 'SwapFree'}
        def values(path, keys):
            return {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                    for line in Path(path).read_text().splitlines()
                    if line.split(':', 1)[0] in keys}
        row = {
            'stage': stage, 'pid': os.getpid(), 'unix_time': time.time(),
            'cuda_allocator_backend': torch.cuda.memory.get_allocator_backend(),
            'cuda_allocated_bytes': stats.get('allocated_bytes.all.current'),
            'cuda_active_bytes': stats.get('active_bytes.all.current'),
            'cuda_reserved_bytes': stats.get('reserved_bytes.all.current'),
            'cuda_inactive_split_bytes': stats.get('inactive_split_bytes.all.current'),
            'cuda_allocation_retries': stats.get('num_alloc_retries'),
            'cuda_oom_count': stats.get('num_ooms'),
            'cuda_segments': stats.get('segment.all.current'),
            'cuda_oversize_segments': stats.get('oversize_segments.current'),
            'private_pool_reserved_current_bytes': {
                key: value for key, value in stats.items()
                if key.startswith('reserved_bytes_by_private_pools.')
                and key.endswith('.all.current')
            },
            'global_cuda_free_bytes': free_bytes, 'global_cuda_total_bytes': total_bytes,
            'process': values('/proc/self/status', proc_keys),
            'host': values('/proc/meminfo', mem_keys),
        }
        logger.info('Startup memory observation: %s', json.dumps(row, sort_keys=True))
    except Exception:
        logger.warning('Startup memory observation failed at %s', stage, exc_info=True)


def trim_startup_cache(logger):
    """Release unused allocator cache once, after capture and calibration."""
    if os.environ.get('DSV41_TRIM_STARTUP_CACHE') != '1':
        return
    import torch

    if torch.cuda.memory.get_allocator_backend() != 'native':
        logger.warning('Startup cache trim skipped: allocator backend is not native')
        return
    torch.cuda.synchronize()
    observe_startup_memory(logger, 'before_startup_cache_trim')
    started = time.perf_counter()
    torch.cuda.empty_cache()
    observe_startup_memory(logger, 'after_startup_cache_trim')
    logger.info('Startup cache trim finished in %.3f ms',
                1000.0 * (time.perf_counter() - started))
