def decode(minifloat):
    sign = (minifloat >> 7) & 1
    exponent = (minifloat >> 4) & 0b111
    mantissa = minifloat & 0b1111
    if exponent == 0 and mantissa == 0:
        return 0.0
    bias = 3
    e = exponent - bias
    m = 1 + mantissa / 16.0  # 1.M
    val = m * (2 ** e)
    return -val if sign else val

def encode(value):
    if value == 0:
        return 0
    sign = 0
    if value < 0:
        sign = 1
        value = -value
    exponent = int(value).bit_length() - 1
    bias = 3
    exp_bits = exponent + bias
    if exp_bits < 0 or exp_bits > 7:
        return 0xFF  # Overflow
    base = 2 ** exponent
    mantissa = round((value / base - 1) * 16)
    if mantissa < 0: mantissa = 0
    if mantissa > 15: mantissa = 15
    return (sign << 7) | (exp_bits << 4) | mantissa

def generate_add_mul_tables():
    ADD_TABLE = [[0 for _ in range(256)] for _ in range(256)]
    MUL_TABLE = [[0 for _ in range(256)] for _ in range(256)]
    for a in range(256):
        for b in range(256):
            va = decode(a)
            vb = decode(b)
            ADD_TABLE[a][b] = encode(va + vb)
            MUL_TABLE[a][b] = encode(va * vb)
    return ADD_TABLE, MUL_TABLE

def write_table_to_rust(table, name, filename):
    with open(filename, 'w') as f:
        f.write(f"pub const {name}: [u8; 65536] = [\n")
        for row in table:
            row_str = ', '.join(f"0x{val:02x}" for val in row)
            f.write(f"    {row_str},\n")
        f.write("];\n")

# === MAIN ===
ADD_TABLE, MUL_TABLE = generate_add_mul_tables()
write_table_to_rust(ADD_TABLE, "ADD_TABLE", "float_add_table.rs")
write_table_to_rust(MUL_TABLE, "MUL_TABLE", "float_mul_table.rs")