import numpy as np
import json
import socket
import threading
import time
import sys
import queue

np.set_printoptions(precision=4)
STANDARD_DT=1.0/120




def parse_message(input_bytes):
    """ decode byte into utf-8 string until 0x00 is found"""
    n_bytes = len(input_bytes)
    start_offset = 0
    end_offset = 0
    msg_str = ""
    while start_offset < n_bytes and start_offset < n_bytes:
        while end_offset < n_bytes and input_bytes[end_offset] != 0x00:
            end_offset += 1
        msg_str += bytes.decode(input_bytes[start_offset:end_offset], "utf-8")
        start_offset = end_offset + 1
    return msg_str



def find_header_of_message(conn):
    LEN = 0
    data = b''
    header_received = False
    while not header_received:
        len_msg = conn.recv(1)
        data += len_msg
        if len(data) == 4:
            LEN = int.from_bytes(data, 'big')
            # print("Length: " + str(LEN))
            data = b''
            header_received = True

    while len(data) < LEN*2:
        byte = conn.recv(1)
        data += byte
    return data



def parse_client_message(server, client_msg_str):
    try:
        if client_msg_str.lower().startswith("ok"):
            return
    except Exception as e:
        print("Exception:",e.args)
        sys.exit(0)
    try:
        server.rev_data.update(json.loads(client_msg_str))
    except Exception as e:
        print("Recieve Data Exception: ", e.args)


def read_client_message_with_header(server, conn):
    input_bytes = conn.recv(server.buffer_size)
    while len(input_bytes) < 2:
        input_bytes += conn.recv(server.buffer_size)
    input_str = bytes.decode(input_bytes, "utf-8")
    if input_str[:2] == "m:":
        end_of_number = input_str[2:].find(':')
        message_size_str = input_str[2:2+end_of_number]
        message_size = int(message_size_str)
        message = input_str[2+end_of_number+1:]
        reached_end = False
        while len(message) < message_size and not reached_end:
            _input_bytes = conn.recv(server.buffer_size)
            n_bytes = len(_input_bytes)
            end_offset = 0
            while end_offset < n_bytes and _input_bytes[end_offset] != 0x00:
                end_offset += 1
            message += bytes.decode(_input_bytes[0:end_offset], "utf-8")
            reached_end = _input_bytes[end_offset-1] == 0x00
        parse_client_message(server, message.rstrip("\0"))# remove trailing null if present
        print(" reading", message)
    else:
        print("Error reading", input_str)


def receive_client_message(server, conn):
    if server.search_message_header:
        read_client_message_with_header(server, conn)
    else:
        input_bytes = conn.recv(server.buffer_size)
        client_msg_str = parse_message(input_bytes)
        parse_client_message(server, client_msg_str)


def send_message(server, conn, msg):
    if server.search_message_header:
        msg = "m:" + str(len(msg)) + ":" + msg
    msg = msg.encode("utf-8")
    msg += b'\x00'
    conn.sendall(msg)

def on_new_client(server, conn, addr):
    #client_msg = conn.recv(1024)
    print("welcome",addr)
    receive_client_message(server, conn)
    skel_dict = server.get_skeleton_dict()
    print(skel_dict)
    server_msg = json.dumps(skel_dict)
    #print("send", len(server_msg), server_msg)
    send_message(server, conn, server_msg)
    print("wait for initial answer")
    #client_msg = conn.recv(server.buffer_size)
    receive_client_message(server, conn)
    print("received answer")
    while True:
        try:
            frame = server.get_frame()
            if frame is not None:
                server_msg = json.dumps(frame)
            else:
                server_msg = ""
            send_message(server, conn, server_msg)
            time.sleep(server.get_frame_time())
            receive_client_message(server, conn)

        except socket.error as error:
            print("connection was closed", error.args)
            conn.close()
            return
    conn.close()


def server_thread(server, s):
    print("server started")
    while server.run:
        c, addr = s.accept()
        t = threading.Thread(target=on_new_client, name="addr", args=(server, c, addr))
        t.start()
        server.connections[addr] = t
    print("server stopped")
    s.close()


class AnimationTCPServer(object):
    """ TCP server that sends and receives a single message
        https://pymotw.com/2/socket/tcp.html
    """
    BUFFER_SIZE = 4092*10000#10485760

    def __init__(self, port, animation_src, buffer_size=BUFFER_SIZE):
        self.address = ("", port)
        self.buffer_size = buffer_size
        self.connections = dict()
        self.run = True
        self.input_key = ""
        self.animation_src = animation_src
        self.search_message_header = False
        self.rev_data = {}

    def start(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(self.address)
        except socket.error:
            print("Binding failed")
            return

        s.listen(10)
        t = threading.Thread(target=server_thread, name="c", args=(self, s))
        t.start()
        print("started server")

    def close(self):
        self.run = False

    def get_frame(self):
        return self.animation_src.frame_buffer

    def get_skeleton_dict(self):
        return self.animation_src.skeleton_dict

    def get_frame_time(self):
        return self.animation_src.frame_time
