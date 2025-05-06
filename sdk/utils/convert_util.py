def decimal_convert_to_little_endian_list(data):
    """
    10进制数据转小端字节数组
    :param data: 10进制数据
    :return: 小端字节数组
    """
    convert_data = int(data)
    low_byte = convert_data & 0xFF  # 低字节
    high_byte = (convert_data >> 8) & 0xFF  # 高字节
    return [low_byte, high_byte]


def decimal_convert_to_big_endian_list(data):
    """
    10进制数据转大端字节数组
    :param data: 10进制数据
    :return: 大端字节数组
    """
    convert_data = int(data)
    low_byte = convert_data & 0xFF  # 低字节
    high_byte = (convert_data >> 8) & 0xFF  # 高字节
    return [high_byte, low_byte]


def little_endian_list_convert_to_decimal(data):
    """
    小端字节数组转10进制数据
    :param data: 小端字节数组
    :return: 10进制数据
    """
    big_endian_data = data[::-1]
    decimal_number = big_endian_convert_to_decimal(big_endian_data)
    return decimal_number


def big_endian_convert_to_decimal(data):
    """
    大端字节数组转10进制数据
    :param data: 大端字节数组
    :return: 10进制数据
    """
    decimal_number = int.from_bytes(data, byteorder='big')
    return decimal_number