import math

def generate_exp_table_pos_16bit(bits=16, scale=None, x_min=0.0, x_max=16.0):
    n = 2 ** bits
    table = []
    if scale is None:
        scale = (2**bits - 1) / math.exp(x_max)  # 
    for i in range(n):
        x = x_min + (x_max - x_min) * i / (n - 1)
        val = math.exp(x)
        fixed_val = int(round(val * scale))
        fixed_val = min(fixed_val, 2 ** bits - 1)
        table.append(fixed_val)
    return table

def write_table_rust(table, filename="exp_table16.rs", varname="EXP_TABLE"):
    with open(filename, "w") as f:
        f.write(f"pub const {varname}: [u8; {len(table) * 2}] = [\n")
        for i in range(0, len(table), 8):
            line = ", ".join(f"0x{(v & 0xff):02x}, 0x{(v >> 8):02x}" for v in table[i:i+8])
            f.write("    " + line + ",\n")
        f.write("];\n")

#
table = generate_exp_table_pos_16bit()
write_table_rust(table)