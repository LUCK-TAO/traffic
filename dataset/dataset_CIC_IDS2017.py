# import os
# from tqdm import tqdm
# import subprocess
#
#
# #/media/wang/文档资料/BaiduNetdiskDownload/CICIDS2017/PCAPs
# def split_pcap_files():
#     splitter = "/home/wang/下载/NetMamba/ShieldGPT/pcap_tool/splitter"
#     flow_packet_num_dict = {
#         "BENIGN": 60,
#         "DoS GoldenEye": 100,
#         "DoS Hulk": 100,
#         "DoS Slowhttptest": 200,
#         "DoS slowloris": 300,
#     }
#     for label in flow_packet_num_dict.keys():
#         filenames = list(filter(lambda x: x.endswith(".pcap") and x.startswith(label),
#                         os.listdir("/media/wang/文档资料/PycharmProjects/analysis-classification-master/del_repeat_pcap"))) # only consider Tor traffic, ignore NonTor
#         filenames = [f"/media/wang/文档资料/PycharmProjects/analysis-classification-master/del_repeat_pcap/{x}" for x in filenames]
#         print(filenames)
#         for filename in tqdm(filenames):
#             dir_name = os.path.join("/home/wang/下载/NetMamba/mnt/ssd1/CICIDS2017/flows", label)
#             os.makedirs(dir_name, exist_ok=True)
#             flow_prefix = os.path.basename(filename)[:-len(".pcap")]
#             with open(os.path.join("/home/wang/下载/NetMamba/mnt/ssd1/CICIDS2017/flows", f"{flow_prefix}.log"), "w") as f:
#                 subprocess.run(f"{splitter} -i '{filename}' -o '{dir_name}' -p {flow_prefix}- -f five_tuple -l {flow_packet_num_dict[label]}",
#                             shell=True, stdout=f, stderr=subprocess.STDOUT)
#
# if __name__ == "__main__":
#     split_pcap_files()

import os
from tqdm import tqdm
import subprocess

def rename_pcap_files():
    filenames = os.listdir("/home/wang/下载/NetMamba/dataset/CICIDS2017")
    for filename in tqdm(filenames):
        new_filename = filename.strip(" ")
        os.rename(os.path.join("/home/wang/下载/NetMamba//dataset/CICIDS2017", filename), os.path.join("/home/wang/下载/NetMamba//dataset/CICIDS2017", new_filename))

def split_pcap_files():
    splitter = "/home/wang/下载/NetMamba/ShieldGPT/pcap_tool/splitter"
    label_dict = {
        "Bot_attack": ["Bot"],
        "DDoS_attack": ["DDoS"],
        "DoS_attack": ["DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"],
        "Patator_attack": ["FTP-Patator", "SSH-Patator"],
        "Infiltration_attack": ["Infiltration"],
        "PortScan_attack": ["PortScan"],
        "Web_attack": ["Web-Attack"]
    }
    # label_dict = {
    #     "Bot_attack": ["Bot"],
    #     "Patator_attack": ["FTP-Patator"],
    #     "Infiltration_attack": ["Infiltration"],
    #     "PortScan_attack": ["PortScan"],
    #     "Web_attack": ["Web-Attack-XSS"]
    # }

    for label, value_list in label_dict.items():
        filenames = []
        for value in value_list:
            filenames += list(filter(lambda x: x.endswith("pcap") and x.startswith(value),
                            os.listdir("/home/wang/下载/NetMamba//dataset/CICIDS2017"))) # only consider VPN traffic, ignore NonVPN
        filenames = [f"/home/wang/下载/NetMamba//dataset/CICIDS2017/{x}" for x in filenames]
        for filename in tqdm(filenames):
            dir_name = os.path.join("/home/wang/下载/NetMamba/mnt/ssd1/CICIDS2017/flows", label)
            os.makedirs(dir_name, exist_ok=True)
            flow_prefix = os.path.basename(filename)[:-len(".pcap")]
            with open(os.path.join("/home/wang/下载/NetMamba/mnt/ssd1/CICIDS2017/flows", f"{flow_prefix}.log"), "w") as f:
                subprocess.run(f"{splitter} -i '{filename}' -o '{dir_name}' -p {flow_prefix}- -f five_tuple",
                            shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    # rename_pcap_files()
    split_pcap_files()