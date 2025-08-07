###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import gc
import gzip
import json
import os
import queue
import math
import threading
import time
from contextlib import contextmanager
from typing import Any, List
import psutil
import torch
import uuid
from habana_frameworks.torch import torch

from vllm_hpu_extension.utils import is_fake_hpu
from .logger import logger

class FileWriter(threading.Thread):

    def __init__(self, filename, event_queue):
        super().__init__()
        self.filename = filename
        self.event_queue = event_queue
        self.daemon = True
        self.timer_event = threading.Event()

    def _drain_event_queue(self):
        content = ''
        while True:
            try:
                element = self.event_queue.get_nowait()
                content += element
            except queue.Empty:
                break
        return content

    def run(self):
        # don't check the queue too often
        while not self.timer_event.wait(1):
            # Block and wait for the next item in the queue
            content = self.event_queue.get()
            # Collect any other items in the queue
            content += self._drain_event_queue()

            with open(self.filename, 'a') as outfile:
                outfile.write(content)


class HabanaProfilerCounterHelper:

    def __init__(self, is_v1 = True):
        self.niter = 0
        self.average_real_throughput = None
        self.logged_once = False
        self.prompt_real_seq_lens = []
        self.decode_real_seq_lens = []
        self.is_v1 = is_v1
        self.real_seq_lens = []
        self.prompt_seq_lens = []
    
    def capture_seq_group_metadata_stats(self, seq_group_metadata_list):
        self.real_seq_lens = [
            len(seq_data.prompt_token_ids) + len(seq_data.output_token_ids)
            for seq_group_metadata in seq_group_metadata_list
            for seq_data in seq_group_metadata.seq_data.values()
        ]
        self.prompt_seq_lens = [
            len(seq_data.prompt_token_ids)
            for seq_group_metadata in seq_group_metadata_list
            for seq_data in seq_group_metadata.seq_data.values()
        ]

    def capture_decode_seq_stats(self, real_seq_lens):
        self.decode_real_seq_lens = real_seq_lens

    def capture_prompt_seq_stats(self, real_seq_lens):
        self.prompt_real_seq_lens.append(real_seq_lens)

    def reset_prompt_seq_stats(self):
        self.prompt_real_seq_lens = []

    def get_counter_dict(self, cache_config, duration, seq_len,
                         batch_size_padded, real_batch_size, prompt_batch_idx,
                         is_prompt):
        throughput = batch_size_padded / (duration / 1e6)
        throughput_effective = real_batch_size / (duration / 1e6)
        if self.is_v1:
            if is_prompt:
                real_max_seq_len = max(self.prompt_real_seq_lens[prompt_batch_idx])
                real_num_tokens = sum(self.prompt_real_seq_lens[prompt_batch_idx])
            else:
                real_max_seq_len = max(self.decode_real_seq_lens)
                real_num_tokens = sum(self.decode_real_seq_lens)
        else:
            real_max_seq_len = max(self.real_seq_lens)
            real_num_tokens = sum(self.real_seq_lens)
        padded_num_tokens = batch_size_padded * seq_len
        batch_token_utilization = real_num_tokens / padded_num_tokens
        if self.average_real_throughput is None:
            self.average_real_throughput = throughput_effective
        else:  # https://www.heikohoffmann.de/htmlthesis/node134.html
            self.average_real_throughput = self.average_real_throughput + 1 / (
                self.niter + 1) * (throughput_effective -
                                   self.average_real_throughput)
        phase = "prompt" if is_prompt else "decode"
        counters = {
            f'{phase}_bucket_batch_size': batch_size_padded,
            f'{phase}_batch_size': real_batch_size,
            f'{phase}_bucket_seq_len': seq_len,
            f'{phase}_seq_len': real_max_seq_len,
            f'{phase}_bucket_gen_throughput': throughput,
            f'{phase}_real_gen_throughput': throughput_effective,
            f'{phase}_batch_token_utilization': batch_token_utilization,
            'average_real_throughput': self.average_real_throughput,
            'engine_iteration': self.niter,
        }
        self.niter += 1
        if is_prompt:
            prompt_bucket_in_throughput = (seq_len * batch_size_padded) / (
                duration / 1e6)
            if self.is_v1:
                prompt_real_in_throughput = sum(
                    self.prompt_real_seq_lens[prompt_batch_idx]) / (duration / 1e6)
            else:
                prompt_real_in_throughput = sum(self.real_seq_lens) / (duration / 1e6)
            counters[
                f'{phase}_bucket_in_throughput'] = prompt_bucket_in_throughput
            counters[f'{phase}_real_in_throughput'] = prompt_real_in_throughput

        # KV cache might not be created yet (e.g. for profiling run)
        if cache_config.num_gpu_blocks is not None and \
            cache_config.num_gpu_blocks != 0:
            if self.is_v1:
                seq_lens = self.prompt_real_seq_lens[prompt_batch_idx] \
                    if is_prompt \
                    else self.decode_real_seq_lens
                cache_num_blocks_used = [
                    math.ceil(sl / cache_config.block_size) for sl in seq_lens
                ]
            else:
                cache_num_blocks_used = [
                    math.ceil(sl / cache_config.block_size)
                    for sl in self.real_seq_lens
                ]
            cache_total_num_blocks_used = sum(cache_num_blocks_used)
            num_cache_blocks = cache_config.num_gpu_blocks
            cache_total_num_free_blocks = \
                num_cache_blocks - cache_total_num_blocks_used
            cache_computed_utilization = \
                cache_total_num_blocks_used / num_cache_blocks
            max_blocks_per_seq = math.ceil(seq_len / cache_config.block_size)
            batch_block_utilization = cache_total_num_blocks_used / (
                batch_size_padded * max_blocks_per_seq)
            counters['cache_num_blocks_used'] = cache_total_num_blocks_used
            counters['cache_num_free_blocks'] = cache_total_num_free_blocks
            counters['cache_computed_utilization'] = cache_computed_utilization
            counters[
                f'{phase}_batch_block_utilization'] = batch_block_utilization
        if not self.logged_once:
            counters['const_cache_num_blocks'] = cache_config.num_gpu_blocks
            counters[
                'const_gpu_memory_utilization'] = \
                    cache_config.gpu_memory_utilization
            counters['const_block_size'] = cache_config.block_size
            self.logged_once = True

        return counters


class HabanaHighLevelProfiler:
    profiling_trace_events: queue.Queue = queue.Queue()
    event_tid = {'counter': 1, 'external': 2, 'internal': 3}
    event_cache: List[Any] = []

    def __init__(self, vllm_instance_id = None):
        self.enabled = os.getenv('VLLM_PROFILER_ENABLED',
                                 'false').lower() == 'true' and int(
                                     os.getenv('RANK', '0')) == 0
        self.pid = os.getpid()
        if self.enabled:
            self.vllm_instance_id = vllm_instance_id if vllm_instance_id is not None \
                else f"vllm-instance-{self.pid}-{str(uuid.uuid4().hex)}"
            msg = f'Profiler enabled for: {self.vllm_instance_id}'
            logger().info(msg)
            self.filename = f'server_events_{self.vllm_instance_id}.json'
            # initialize the trace file (JSON Array Format)
            with open(self.filename, 'w') as outfile:
                outfile.write('[')
            file_writer = FileWriter(self.filename,
                                     self.profiling_trace_events)
            file_writer.start()
        if os.getenv('VLLM_PROFILER_ENABLED') == 'full':
            self.enabled = True # don't save separate high-level traces

    def _dump_with_sep(self, entry):
        entry = json.dumps(entry) + ','
        self.profiling_trace_events.put(entry)

    def get_timestamp_us(self):
        return time.time() * 1000000.0

    def record_counter(self, ts, counter):
        if self.enabled:
            self._dump_with_sep({
                'pid': self.pid,
                'tid': self.event_tid['counter'],
                'ph': 'C',
                'name': 'utils',
                'ts': ts,
                'args': counter
            })

    def start(self, type, name, args=None):
        if self.enabled:
            ts = self.get_timestamp_us()
            if args is not None and 'counter' in args:
                self.record_counter(ts, args['counter'])
                del args['counter']
            event = {
                'pid': self.pid,
                'tid': self.event_tid[type],
                'ph': 'X',
                'name': name,
                'ts': ts,
                'dur': None,
                'args': args
            }
            self.event_cache.append(event)

    def end(self):
        if self.enabled:
            ts = self.get_timestamp_us()
            if not self.event_cache:
                logger().warning(
                    'Profiler: end() call does not have matching start() call. '
                    'Disabling profiler.')
                self.enabled = False
                return
            event = self.event_cache.pop()
            event['dur'] = ts - event['ts']
            self._dump_with_sep(event)
 

    def full_trace_handler(self, dir_name, use_gzip=False):

        def handler_fn(prof) -> None:
            if not os.path.isdir(dir_name):
                try:
                    os.makedirs(dir_name, exist_ok=True)
                except Exception as e:
                    raise RuntimeError("Can't create directory: " +
                                       dir_name) from e
            file_name = f"vllm.{time.time_ns()}.pt.trace.json"
            file_path = os.path.join(dir_name, file_name)
            prof.export_chrome_trace(file_path)
            with open(file_path) as f:
                pytorch_trace = json.load(f)
            os.remove(file_path)
            base = pytorch_trace['baseTimeNanoseconds'] / 1000
            events = self.profiling_trace_events
            while True:
                try:
                    event_str = events.get_nowait()
                    event = json.loads(event_str[:-1])
                    event['ts'] = event['ts'] - base
                    pytorch_trace['traceEvents'].append(event)
                except queue.Empty:
                    break

            pytorch_trace['traceEvents'].append({
                "args": {
                    "name": "vLLM"
                },
                "name": "process_name",
                "ph": "M",
                "pid": 1,
                "tid": 0,
                "ts": 0.0
            })
            if use_gzip:
                file_path = file_path + ".gz"
                with gzip.open(file_path, 'wt', encoding="ascii") as zipfile:
                    json.dump(pytorch_trace, zipfile)
            else:
                with open(file_path, "w") as outfile:
                    outfile.write(json.dumps(pytorch_trace))
            logger().info("Saved full profiling to %s", file_path)

        return handler_fn


    @contextmanager
    def record_event(self, type, name, args=None):
        if self.enabled:
            self.start(type, name, args)
            yield
            self.end()
        else:
            yield


# Adapted from https://stackoverflow.com/a/49361727
def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti'}
    while abs(size) > power:
        size /= power
        n += 1
    return f'{size:.4g} {power_labels[n]+"B"}'


class HabanaMemoryProfiler:

    def __init__(self, device=None):
        self.device = device

    @staticmethod
    def current_device_memory_usage() -> float:
        if is_fake_hpu():
            return 0
        # Return the device memory usage in bytes.
        free_hpu_memory, total_hpu_memory = torch.hpu.mem_get_info()
        return total_hpu_memory - free_hpu_memory

    @staticmethod
    def current_free_device_memory() -> float:
        if is_fake_hpu():
            return 0
        # Return the device memory usage in bytes.
        free_hpu_memory, _ = torch.hpu.mem_get_info()
        return free_hpu_memory

    @staticmethod
    def total_device_memory() -> float:
        if is_fake_hpu():
            return 0
        # Return the device memory usage in bytes.
        _, total_hpu_memory = torch.hpu.mem_get_info()
        return total_hpu_memory

    @staticmethod
    def current_host_memory_usage() -> float:
        # Return the host memory usage in bytes.
        return HabanaMemoryProfiler.total_host_memory(
        ) - HabanaMemoryProfiler.current_free_host_memory()

    @staticmethod
    def current_free_host_memory() -> float:
        # Return the host memory usage in bytes.
        return psutil.virtual_memory().available

    @staticmethod
    def total_host_memory() -> float:
        # Return the host memory usage in bytes.
        return psutil.virtual_memory().total

    def get_summary_string(self):
        if getattr(self, 'final_device_memory', None) is None or getattr(
                self, 'final_host_memory', None) is None:
            raise RuntimeError(
                "HabanaMemoryProfiler.get_summary_string() can only be called "
                "after closing context manager")
        return (
            f"{format_bytes(self.consumed_device_memory)} of device memory "
            f"({format_bytes(self.final_device_memory)}/"
            f"{format_bytes(HabanaMemoryProfiler.total_device_memory())} used)"
            f" and {format_bytes(self.consumed_host_memory)} of host memory "
            f"({format_bytes(self.final_host_memory)}/"
            f"{format_bytes(HabanaMemoryProfiler.total_host_memory())} used)")

    def __enter__(self):
        # Force garbage collection
        gc.collect()
        self.initial_device_memory = \
            HabanaMemoryProfiler.current_device_memory_usage()
        self.initial_host_memory = \
            HabanaMemoryProfiler.current_host_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Force garbage collection
        gc.collect()
        self.final_device_memory = \
            HabanaMemoryProfiler.current_device_memory_usage(
        )
        self.final_host_memory = HabanaMemoryProfiler.current_host_memory_usage(
        )
        self.consumed_device_memory = \
            self.final_device_memory - self.initial_device_memory
        self.consumed_host_memory = \
            self.final_host_memory - self.initial_host_memory


def setup_profiler(warmup, active):
    schedule = torch.profiler.schedule(wait=0,
                                       warmup=warmup,
                                       active=active,
                                       repeat=1)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.HPU
    ]
    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.',
                                                                use_gzip=True),
        record_shapes=False,
        with_stack=True)
    return profiler
