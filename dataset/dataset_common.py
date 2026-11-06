import os
import numpy as np
import binascii
import scapy.all as scapy
from scapy.all import raw
from scapy.packet import Raw
from scapy.layers.inet import IP, TCP, UDP
from scapy.all import rdpcap
from scapy.layers.dns import DNS, DNSQR, DNSRR
import datetime

import json


def find_files(data_path, extension=".pcap"):
    pcap_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(extension):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def raw_packet_to_string(packet, remove_ip=True, keep_payload=True):
    ip = packet["IP"]
    if remove_ip:
        PAD_IP_ADDR = "0.0.0.0"
        ip.src, ip.dst = PAD_IP_ADDR, PAD_IP_ADDR
    header = (binascii.hexlify(bytes(ip))).decode()
    if keep_payload:
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
    else:
        payload = ''
    header = header[:160] if len(header) > 160 else header + '0' * (160 - len(header))
    payload = payload[:480] if len(payload) > 480 else payload + '0' * (480 - len(payload))
    return header, payload

def string_to_hex_array(flow_string):
    return np.array([int(flow_string[i:i + 2], 16) for i in range(0, len(flow_string), 2)])

# def read_5hp_list(pcap_filename, if_augment=False, remove_ip=True, keep_payload=True):
#     packets = scapy.rdpcap(pcap_filename)
#     data = []
#     flow_string_length = 3200
#     flow_packet_num = 5
#     end = len(packets) if if_augment else flow_packet_num
#     for packet in packets[:end]:
#         try:
#             # ip = packet['IP']
#             header, payload = raw_packet_to_string(packet, remove_ip=remove_ip, keep_payload=keep_payload)
#         except:
#             # continue
#             header, payload = '0' * 160, '0' * 480
#         data.append(header + payload)
#
#     if not if_augment or len(data) <= flow_packet_num:
#         flow_string = ''.join(data)
#         flow_string += '0' * (flow_string_length - len(flow_string))
#         flow_array = string_to_hex_array(flow_string)
#         return [{
#             "data": flow_array,
#         }]
#     else:
#         assert len(data) > flow_packet_num
#         flow_array_list = []
#         for i in range(len(data) - flow_packet_num + 1):
#             flow_string = ''.join(data[i:i + flow_packet_num])
#             flow_array_list.append(string_to_hex_array(flow_string))
#         return [{
#             "data": flow_array,
#         } for flow_array in
#         zip(flow_array_list)]



# 提取应用层协议
def get_application_layer(packet):
    if Raw in packet:
        return packet[Raw].load.decode(errors="ignore")  # 解码为可读格式
    return None

def extract_application_protocol(raw_data):
    try:
        # 检查是否包含 HTTP/1.x 协议标志
        if raw_data.startswith("HTTP/1."):
            return "HTTP (likely used for UPnP)"
        # 检查是否包含 UPnP 服务类型标志
        if "ST: upnp:" in raw_data:
            return "UPnP over HTTP"
        # 如果没有明确标志，返回未知
    except Exception as e:
        print(e)
    return None

def is_dns_packet(packet):
    # 检查是否是 UDP 数据包并且有负载
    if packet.haslayer(UDP):
        udp_payload = packet[UDP].payload
        # 检查是否有 DNS 层
        if packet.haslayer(DNS):
            dns_layer = packet[DNS]
            if dns_layer.qr == 0:  # qr=0 表示查询
                return "DNS Query"
            elif dns_layer.qr == 1:  # qr=1 表示响应
                return "DNS Response"
    return None

def extract_flow_data(packets):
    """
    从数据包列表中提取流量信息
    Args:
        packets: Scapy 读取的数据包列表
    Returns:
        dict: 包含流量信息的字典
    """
    flow = {
        "start": None,
        "pkts_toserver": 0,
        "pkts_toclient": 0,
        "bytes_toserver": 0,
        "bytes_toclient": 0,
    }

    server_ip = None  # 用于区分服务端与客户端的IP（假设第一个包中IP为服务端IP）

    for packet in packets:


        # 提取时间戳
        if flow["start"] is None:
            flow["start"] = datetime.datetime.fromtimestamp(float(packet.time)).strftime('%Y-%m-%d %H:%M:%S.%f')

        # 提取 IP 地址
        src_ip = packet['IP'].src
        dst_ip = packet['IP'].dst
        length = len(packet)

        # 假设服务端IP为第一个包的目的地
        if server_ip is None:
            server_ip = dst_ip

        # 判断方向并累加
        if dst_ip == server_ip:
            flow["pkts_toserver"] += 1
            flow["bytes_toserver"] += length
        else:
            flow["pkts_toclient"] += 1
            flow["bytes_toclient"] += length


    return flow

def read_5hp_list(pcap_filename, if_augment=False, remove_ip=True, keep_payload=True):
    packets = scapy.rdpcap(pcap_filename)
    data = []
    flow_details = []  # 用于存储详细信息
    flow_string_length = 3200
    flow_packet_num = 5
    end = len(packets) if if_augment else flow_packet_num
    flow_info = extract_flow_data(packets)

    for packet in packets[:end]:
        try:
            time = float(packet.time)
            timestamp = datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S.%f')
            src_ip = packet['IP'].src
            dst_ip = packet['IP'].dst
            src_port = packet['TCP'].sport if 'TCP' in packet else (
                    packet['UDP'].sport if 'UDP' in packet else None)
            dst_port = packet['TCP'].dport if 'TCP' in packet else (
                    packet['UDP'].dport if 'UDP' in packet else None)
            protocol = packet['IP'].proto
            flow_id = "{}-{}-{}-{}-{}".format(src_ip, dst_ip, src_port, dst_port, protocol)
            App_protocol = None
            # App_protocol_detail = get_application_layer(packet)
            # print(App_protocol_detail)
            # App_protocol = extract_application_protocol(App_protocol_detail)
            # # print(App_protocol)

            if App_protocol is None:
                App_protocol = is_dns_packet(packet)
                print(App_protocol)
                if App_protocol is None:
                    if packet.haslayer(TCP) or packet.haslayer(UDP):
                        if src_port == 80 or dst_port ==80:
                            App_protocol = "HTTP"
                        elif src_port == 137 or dst_port == 137:
                            App_protocol = "NBNS"
                        elif src_port == 53 or dst_port == 53:
                            App_protocol = "DNS"
                        elif src_port == 21 or dst_port == 21:
                            App_protocol = "FTP"
                        elif src_port == 443 or dst_port == 443:
                            App_protocol = "HTTPS"
                        else:
                            App_protocol = "None"

            print("APP_p"+App_protocol)

            # 提取详细信息
            details = {
                "Timestamp": timestamp,
                "flow_id": flow_id,
                "src_ip": src_ip if 'IP' in packet else None,
                "dst_ip": dst_ip if 'IP' in packet else None,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol if 'IP' in packet else None,
                "App_protocol": App_protocol,
                "raw_bytes": raw(packet).hex() if raw else None
            }
            flow_details.append(details)
            print(details)

            # 提取包头和有效载荷
            header, payload = raw_packet_to_string(packet, remove_ip=remove_ip, keep_payload=keep_payload)
            print(header)
            print(payload)
        except Exception as e:
            # 处理解析失败的包
            print(f"Error reading: {e}")
            continue

            # header, payload = '0' * 160, '0' * 480
            # details = {
            #     "src_ip": None,
            #     "dst_ip": None,
            #     "src_port": None,
            #     "dst_port": None,
            #     "protocol": None,
            #     "raw_bytes": None
            # }
            #
            # flow_details.append(details)

        data.append(header + payload)


    # 拼接成 flow_string 并转换为数组
    if not if_augment or len(data) <= flow_packet_num:
        flow_string = ''.join(data)
        flow_string += '0' * (flow_string_length - len(flow_string))
        flow_array = string_to_hex_array(flow_string)
        return [{
            "data": flow_array,
            "details": flow_details,  # 将详细信息返回
            "flow": flow_info
        }]
    else:
        # 增强数据
        assert len(data) > flow_packet_num
        flow_array_list = []
        for i in range(len(data) - flow_packet_num + 1):
            flow_string = ''.join(data[i:i + flow_packet_num])
            flow_array_list.append(string_to_hex_array(flow_string))
        return [{
            "data": flow_array,
            "details": flow_details[i:i + flow_packet_num]  # 对应增强的流量数据
        } for i, flow_array in enumerate(flow_array_list)]
