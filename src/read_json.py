import json

# 加载原始 JSON 数据
with open('test.json', 'r') as file:
    data = json.load(file)

# 提取所需的信息
extracted_data = []
for entry in data:
    layers = entry["_source"]["layers"]

    # 提取 IP 信息
    src_ip = layers["ip"]["ip.src"]
    dst_ip = layers["ip"]["ip.dst"]

    # 检查是否有 TCP 层，并提取端口
    if "tcp" in layers:
        src_port = layers["tcp"]["tcp.srcport"]
        dst_port = layers["tcp"]["tcp.dstport"]
    else:
        # 如果没有 TCP 层，跳过此条数据或设定默认值
        continue  # 跳过没有 TCP 层的数据，或者可以设为默认值，如：src_port = dst_port = 0

    # 将数据添加到列表，class 标记为 BENIGN
    extracted_data.append({
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": int(src_port),
        "dst_port": int(dst_port),
        "class": "BENIGN"
    })

# 将提取的数据写入到新的 JSON 文件
with open('extracted_data.json', 'w') as outfile:
    json.dump({"Extracted Data": extracted_data}, outfile, ensure_ascii=False, indent=4)

print("提取完成，数据已保存到 'extracted_data.json'。")
