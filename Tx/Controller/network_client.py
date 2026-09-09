"""DLC GUI networking adapted to latest-frame M1-1 preview."""
import socket
import threading
import queue
import time
import json
from common.protocol import messages,receive_frame,send_json

class Network:
    def __init__(self,host):
        self.host=host;self.lock=threading.Lock();self.output_lock=threading.Lock()
        self.events=queue.Queue();self.latest=None;self.count=0;self.ctrl=None
        self.sockets=set();self.stop=threading.Event()
        self.threads=[threading.Thread(target=self.worker,args=(p,),daemon=True) for p in (7600,7601)]
        for t in self.threads:t.start()
    def worker(self,port):
        while not self.stop.is_set():
            sock=None
            try:
                sock=socket.create_connection((self.host,port),timeout=3);sock.settimeout(None)
                with self.lock:
                    self.sockets.add(sock)
                    if port==7600:self.ctrl=sock
                self.events.put(dict(event='network',port=port,state='connected'))
                if port==7600:
                    for event in messages(sock):self.events.put(event)
                else:
                    while not self.stop.is_set():
                        name,data=receive_frame(sock)
                        meta=json.loads(name.removeprefix('_preview_'))
                        with self.lock:
                            self.latest=(data,meta,time.monotonic(),time.time_ns())
                            self.count+=1
            except (OSError,EOFError,ValueError) as exc:
                if not self.stop.is_set():self.events.put(dict(event='network',port=port,state='disconnected',message=str(exc)))
            finally:
                if sock:
                    with self.lock:
                        self.sockets.discard(sock)
                        if self.ctrl is sock:self.ctrl=None
                    sock.close()
            self.stop.wait(1)
    def send(self,obj):
        with self.output_lock:
            with self.lock:sock=self.ctrl
            if sock is None:raise ConnectionError('Control channel not connected')
            send_json(sock,obj)
    def pop(self):
        with self.lock:
            frame=self.latest;self.latest=None
            return frame,self.count
    def close(self):
        self.stop.set()
        with self.lock:sockets=list(self.sockets)
        for s in sockets:
            try:s.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            s.close()
        for t in self.threads:t.join(timeout=.5)
