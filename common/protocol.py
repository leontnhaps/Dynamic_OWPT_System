"""DLC image framing: uint16 name length, UTF-8 name, uint32 JPEG length, JPEG.
M1-1 revision: the name contains _preview_ followed by JSON frame information.
"""
import json
import struct

def exact(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size-len(data))
        if not part:
            raise EOFError('Disconnected')
        data.extend(part)
    return bytes(data)

def receive_frame(sock):
    n = struct.unpack('<H', exact(sock, 2))[0]
    name = exact(sock, n).decode()
    n = struct.unpack('<I', exact(sock, 4))[0]
    if not 0 < n <= 32*1024*1024:
        raise ValueError('Invalid JPEG size')
    return name, exact(sock, n)

def send_frame(sock, name, jpeg):
    raw = name.encode()
    sock.sendall(struct.pack('<H', len(raw))+raw+struct.pack('<I', len(jpeg))+jpeg)

def send_json(sock, obj):
    sock.sendall((json.dumps(obj)+'\n').encode())

def messages(sock):
    buf = b''
    while True:
        part = sock.recv(4096)
        if not part:
            return
        buf += part
        if len(buf) > 65536:
            raise ValueError('Oversized command')
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            if line:
                yield json.loads(line)
