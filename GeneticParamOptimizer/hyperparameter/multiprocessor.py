import torch.multiprocessing
import torch.multiprocessing.pool

class NoDaemonProcess(torch.multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NonDaemonPool(torch.multiprocessing.pool.Pool):
    Process = NoDaemonProcess