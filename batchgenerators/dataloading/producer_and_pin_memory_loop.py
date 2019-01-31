import traceback
import numpy as np
from queue import Empty, Full


def producer(queue, data_loader, transform, thread_id, seed, abort_event):
    try:
        np.random.seed(seed)
        data_loader.set_thread_id(thread_id)
        item = None

        while True:
            # check if abort event was set
            if not abort_event.is_set():

                if item is None:

                    try:
                        item = next(data_loader)
                        if transform is not None:
                            item = transform(**item)
                    except StopIteration:
                        item = "end"

                try:
                    queue.put(item, timeout=2)
                    item = None
                except Full:
                    # queue was full because items in it were not consumed. Try again.
                    pass
            else:
                # abort_event was set. Drain queue, then give 'end'
                break

    except KeyboardInterrupt:
        # drain queue, then give 'end', set abort flag and reraise KeyboardInterrupt
        abort_event.set()

        raise KeyboardInterrupt

    except Exception:
        print("Exception in worker", thread_id)
        traceback.print_exc()
        # drain queue, give 'end', send abort_event so that other workers know to exit

        abort_event.set()


def pin_memory_loop(in_queues, out_queue, abort_event):
    import torch
    queue_ctr = 0
    item = None
    while True:
        try:
            if not abort_event.is_set():
                if item is None:
                    item = in_queues[queue_ctr % len(in_queues)].get(timeout=1)
                    if isinstance(item, dict):
                        for k in item.keys():
                            if isinstance(item[k], torch.Tensor):
                                item[k] = item[k].pin_memory()
                    queue_ctr += 1
                out_queue.put(item, timeout=1)
                item = None
            else:
                print('pin_memory_loop exiting...')
                break
        except Empty:
            pass
        except Full:
            pass
