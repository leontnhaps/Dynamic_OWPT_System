"""Adapted from DLC_system_model relay: same 7500/7501/7600/7601 roles.
One controlling GUI, one Pi. Latest-frame slot decouples acquisition and GUI send.
"""
import socket
import socketserver
import threading
import time
import select
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.protocol import messages, send_json, receive_frame, send_frame

lock = threading.RLock()
agent = None
gui = None
condition = threading.Condition()
latest = None
version = 0

class Control(socketserver.BaseRequestHandler):
    def setup(self):
        self.output_lock = threading.Lock()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    def send(self, obj):
        with self.output_lock:
            send_json(self.request, obj)

class PiControl(Control):
    def handle(self):
        global agent
        with lock:
            if agent:
                self.send({'event':'error','message':'Pi already connected'})
                return
            agent = self
        notify({'event':'agent','state':'connected'})
        try:
            for event in messages(self.request):
                notify(event)
        except (OSError, ValueError):
            pass
        finally:
            with lock:
                if agent is self:
                    agent = None
            notify({'event':'agent','state':'disconnected'})

def notify(event):
    with lock:
        target = gui
    if target:
        try:
            target.send(event)
        except OSError:
            pass

class GuiControl(Control):
    def handle(self):
        global gui
        with lock:
            if gui:
                self.send({'event':'error','message':'One controlling GUI only'})
                return
            gui = self
            connected = agent is not None
        try:
            self.send({'event':'hello','agent_state':'connected' if connected else 'disconnected'})
            for cmd in messages(self.request):
                with lock:
                    target = agent
                if target:
                    target.send(cmd)
                else:
                    self.send({'event':'error','message':'Pi not connected'})
        except (OSError, ValueError):
            pass
        finally:
            with lock:
                if gui is self:
                    gui = None
            # Stop acquisition on GUI control disconnect.
            with lock:
                target = agent
            if target:
                try:
                    target.send({'cmd':'preview','enable':False})
                except OSError:
                    pass

class PiImages(socketserver.BaseRequestHandler):
    def handle(self):
        global latest, version
        try:
            while True:
                frame = receive_frame(self.request)
                with condition:
                    latest = frame
                    version += 1
                    condition.notify_all()
        except (OSError, EOFError, ValueError):
            pass

class GuiImages(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(2)
        with condition:
            seen = version
        try:
            while True:
                with condition:
                    condition.wait_for(lambda: version != seen, timeout=1)
                    if version == seen:
                        if select.select([self.request], [], [], 0)[0] and not self.request.recv(1):
                            return
                        continue
                    seen = version
                    frame = latest
                send_frame(self.request, *frame)
        except OSError:
            pass

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

def main():
    servers = []
    try:
        for port, handler in [(7500,PiControl),(7501,PiImages),(7600,GuiControl),(7601,GuiImages)]:
            srv = Server(('0.0.0.0',port), handler)
            servers.append(srv)
            threading.Thread(target=srv.serve_forever,daemon=True).start()
        print('Relay ready: Pi=7500/7501 GUI=7600/7601',flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            srv.shutdown()
            srv.server_close()
if __name__ == '__main__':
    main()
