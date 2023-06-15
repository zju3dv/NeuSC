# This file should serve as a drop in replacement for log_utils.py
import os
import sys
import time
import rich
import builtins
import warnings
import readline  # if you need to update stuff from input
import numpy as np

from typing import List, Optional, IO
from collections import deque

from rich import traceback, pretty
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.control import Control
from rich.console import Console
from rich.progress import Progress
from rich.pretty import Pretty, pretty_repr
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn, filesize, ProgressColumn

from tqdm.std import tqdm as std_tqdm
from tqdm.rich import tqdm_rich, FractionColumn, RateColumn

from lib.utils.base_utils import default_dotdict, dotdict

warnings.filterwarnings("ignore")  # ignore disturbing warnings

console = Console()  # shared
progress = Progress(console=console)  # destroyed
live = Live(console=console)  # destroyed
traceback.install(console=console)  # for colorful tracebacks
pretty.install(console=console)

NoneType = type(None)

# NOTE: we use console.log for general purpose logging
# Need to check its reliability and integratability
# Since the string markup might mess things up
# TODO: maybe make the console object confiugrable?


def stop_live():
    global live
    if live is None: return
    live.stop()
    live = None


def start_live():
    global live
    if live is not None: return
    live = Live(console=console)
    live.start()


def stop_progress():
    global progress
    if progress is None: return
    progress.stop()
    progress = None


def start_progress():
    global progress
    if progress is not None: return
    progress = Progress(console=console)


def stacktrace():
    # print colorful stacktrace
    console.print_exception()  # messy formats


def debugger():
    stop_live()
    stop_progress()
    import ipdb
    ipdb.set_trace(sys._getframe(1))


def post_mortem():
    stop_live()
    stop_progress()
    import ipdb
    ipdb.post_mortem()  # break on the last exception's stack for inpection


def line(obj):
    return pretty_repr(obj)


def log(*stuff, back=1, **kwargs):
    frame = sys._getframe(back)  # with another offset
    func = frame.f_code.co_name
    module = frame.f_globals['__name__'] if frame is not None else ''
    prefix = blue(module) + " -> " + green(func) + ":"
    console.log(prefix, *stuff, _stack_offset=2, **kwargs)


def run(cmd, quite=False, dry_run=False):
    if isinstance(cmd, list):
        cmd = ' '.join(list(map(str, cmd)))
    func = sys._getframe(1).f_code.co_name
    if not quite:
        cmd_color = 'cyan' if not cmd.startswith('rm') else 'red'
        cmd_color = 'green' if dry_run else cmd_color
        dry_msg = magenta('[dry_run]: ') if dry_run else ''
        print(yellow(func) + ":", dry_msg + color(cmd, cmd_color), soft_wrap=True)
    if not dry_run:
        code = os.system(cmd)
    else:
        code = 0
    if code != 0:
        print(red(code), "<-", yellow(func) + ":", red(cmd), soft_wrap=True)
        raise RuntimeError(f'{code} <- {func}: {cmd}')


def run_if_not_exists(cmd, outname, *args, **kwargs):
    # whether a file exists, whether a directory has more than 3 elements
    # if (os.path.exists(outname) and os.path.isfile(outname)) or (os.path.isdir(outname) and len(os.listdir(outname)) >= 3):
    if os.path.exists(outname):
        log(yellow('Skipping:'), cyan(cmd))
    else:
        run(cmd, *args, **kwargs)


def print(*stuff,
          sep: str = " ",
          end: str = "\n",
          file: Optional[IO[str]] = None,
          flush: bool = False,
          **kwargs,
          ):
    r"""Print object(s) supplied via positional arguments.
    This function has an identical signature to the built-in print.
    For more advanced features, see the :class:`~rich.console.Console` class.

    Args:
        sep (str, optional): Separator between printed objects. Defaults to " ".
        end (str, optional): Character to write at end of output. Defaults to "\\n".
        file (IO[str], optional): File to write to, or None for stdout. Defaults to None.
        flush (bool, optional): Has no effect as Rich always flushes output. Defaults to False.

    """
    write_console = console if file is None else Console(file=file)
    write_console.print(*stuff, sep=sep, end=end, **kwargs)


def red(string: str) -> str: return f'[red bold]{string}[/]'
def blue(string: str) -> str: return f'[blue bold]{string}[/]'
def cyan(string: str) -> str: return f'[cyan bold]{string}[/]'
def green(string: str) -> str: return f'[green bold]{string}[/]'
def yellow(string: str) -> str: return f'[yellow bold]{string}[/]'
def magenta(string: str) -> str: return f'[magenta bold]{string}[/]'
def color(string: str, color: str): return f'[{color} bold]{string}[/]'


def markup_to_ansi(string: str) -> str:
    """Convert rich-style markup to ANSI sequences for command-line formatting.

    Args:
        string: Text with rich-style markup.

    Returns:
        Text formatted via ANSI sequences.
    """
    with console.capture() as out:
        console.print(string, soft_wrap=True)
    return out.get()

# https://github.com/tqdm/tqdm/blob/master/tqdm/rich.py
# this is really nice
# if we want to integrate this into our own system, just import the tqdm from here


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit="", unit_scale=False, unit_divisor=1000):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f"? {self.unit}/s", style="progress.data.speed")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        # precision = 3 if unit == 1 else 6
        ratio = speed / unit

        precision = 3 - int(np.log(ratio) / np.log(10))
        precision = max(0, precision)
        return Text(f"{ratio:,.{precision}f} {suffix}{self.unit}/s",
                    style="progress.data.speed")
        # ratio = speed / unit
        # if ratio > 1:
        #     return Text(f"{ratio:,.{precision}f} {suffix}{self.unit}/s",
        #                 style="progress.data.speed")
        # else:
        #     return Text(f"{1/ratio:,.{precision}f} {suffix}s/{self.unit}",
        #                 style="progress.data.speed")


class tqdm_rich(std_tqdm):
    def __init__(self, *args, **kwargs):
        # Thanks! tqdm!
        super().__init__(*args, **kwargs)
        if self.disable: return

        # Whatever for now
        stop_live()
        start_progress()

        # Use the predefined progress object
        d = self.format_dict
        self._prog = kwargs.pop('progress', None) or progress
        self._prog.columns = (
            "[progress.description]{task.description}"
            "[progress.percentage]{task.percentage:>4.0f}%",
            BarColumn(bar_width=None),
            FractionColumn(
                unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
            "[", TimeElapsedColumn(), "<", TimeRemainingColumn(),
            ",", RateColumn(unit=d['unit'], unit_scale=d['unit_scale'],
                            unit_divisor=d['unit_divisor']), "]"
        )
        self._prog.start()
        self._task_id = self._prog.add_task(self.desc or "", **d)

    def close(self):
        if self.disable: return
        self.disable = True  # prevent multiple closures

        self.display(refresh=True)
        if self._prog.finished:
            self._prog.stop()
            for task_id in self._prog.task_ids:
                self._prog.remove_task(task_id)

            # Whatever for now
            stop_progress()

    def clear(self, *_, **__):
        pass

    def display(self, refresh=True, *_, **__):
        if not hasattr(self, '_prog'):
            return
        if self._task_id not in self._prog.task_ids:
            return
        self._prog.update(self._task_id, completed=self.n, description=self.desc, refresh=refresh)

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_prog'):
            self._prog.reset(self._task_id, total=total)
        super(tqdm_rich, self).reset(total=total)


tqdm = tqdm_rich


def time_function(func):
    """Decorator: time a function call"""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        end = time.perf_counter()

        log(func.__name__, f"{(end - start) * 1000:.3f} ms")
        return ret

    return wrapper


class time_block:
    def __init__(self, name='', disabled: bool = False, sync_cuda: bool = False):
        self.sync_cuda = sync_cuda
        self.disabled = disabled
        self.name = name
        self.start()  # you can always restart multiple times to reuse this timer

    def __enter__(self):
        self.start()

    def start(self):
        if self.disabled: return self
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.stop()

    def stop(self, print=True, back=2):
        if self.disabled: return
        if self.sync_cuda:
            import torch  # don't want to place this outside
            torch.cuda.synchronize()
        start = self.start_time
        end = time.perf_counter()
        if print: log(f"{(end - start) * 1000:7.3f} ms", self.name, back=back)  # 3 decimals, 3 digits
        return end - start  # return the difference

    def record(self, event: str=''):
        self.name = event
        diff = self.stop(print=bool(event), back=3)
        self.start()
        return diff


def temporal_timer(*args, **kargs):
    time_block_timer = time_block(*args, **kargs)
    time_block_timer.start()
    return time_block_timer


timer = time_block()  # shared global Timer

# @time_function
dict_table_rows = None


def display_dict_table(states: dotdict,
                       styles: default_dotdict[str, NoneType] = default_dotdict(
                           NoneType,
                           {
                               'eta': 'cyan',
                               'epoch': 'cyan',
                               'img_loss': 'magenta',
                               'psnr': 'magenta',
                               'loss': 'magenta',
                               'data': 'blue',
                               'batch': 'blue',
                           }
                       ),
                       maxlen=5,
                       ):

    def create_table(columns: List[str],
                     rows: List[List[str]] = [],
                     styles: default_dotdict[str, NoneType] = default_dotdict(NoneType),
                     ):
        try:
            from easyvv.engine import cfg
            title = cfg.exp_name  # MARK: global config & circular imports
        except Exception as e:
            title = None
        table = Table(title=title, show_footer=True, show_header=False)
        for col in columns:
            table.add_column(footer=Text(col, styles[col]), style=styles[col], justify="center")
        for row in rows:
            table.add_row(*row)
        return table

    keys = list(states.keys())
    values = list(map(str, states.values()))
    width, height = os.get_terminal_size()
    maxlen = max(min(height - 8, maxlen), 1)  # 5 would fill the terminal

    global dict_table_rows
    if dict_table_rows is None:
        dict_table_rows = deque(maxlen=maxlen)
    if dict_table_rows.maxlen != maxlen:
        dict_table_rows = deque(list(dict_table_rows)[-maxlen + 1:], maxlen=maxlen)  # save space for header and footer
    dict_table_rows.append(values)

    # MARK: check performance hit of these calls
    start_live()
    live.update(create_table(keys, dict_table_rows, styles))  # disabled autorefresh
