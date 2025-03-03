#    This file is part of chipwhisperer.
#
#    chipwhisperer is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    chipwhisperer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with chipwhisperer.  If not, see <http://www.gnu.org/licenses/>.
#
#   Inspiration taken from https://github.com/Vipul97/des/blob/master/des/des.py
#
#=================================================
from collections import OrderedDict
import inspect
import numpy as np
from chipwhisperer.common.utils.util import binarylist2bytearray, bytearray2binarylist
try:
    from .base import ModelsBase
    debug = False
except ImportError:
    from base import ModelsBase
    debug = True
global attack_round

from textwrap import wrap

BLOCK_SIZE = 64

# number left rotations of pc1
KEY_SCHEDULE = [0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

PC1 = [
    56, 48, 40, 32, 24, 16,  8,
     0, 57, 49, 41, 33, 25, 17,
     9,  1, 58, 50, 42, 34, 26,
    18, 10,  2, 59, 51, 43, 35,
    62, 54, 46, 38, 30, 22, 14,
     6, 61, 53, 45, 37, 29, 21,
    13,  5, 60, 52, 44, 36, 28,
    20, 12,  4, 27, 19, 11,  3
]

PC1_INV = [
    7, 15, 23, 55, 51, 43, 35, None,
    6, 14, 22, 54, 50, 42, 34, None,
    5, 13, 21, 53, 49, 41, 33, None,
    4, 12, 20, 52, 48, 40, 32, None,
    3, 11, 19, 27, 47, 39, 31, None,
    2, 10, 18, 26, 46, 38, 30, None,
    1,  9, 17, 25, 45, 37, 29, None,
    0,  8, 16, 24, 44, 36, 28, None
]

IP = [
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7,
    56, 48, 40, 32, 24, 16, 8, 0,
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6
]

P_BOX = [
    15, 6, 19, 20, 28, 11, 27, 16,
    0, 14, 22, 25, 4, 17, 30, 9,
    1, 7, 23, 13, 31, 26, 2, 8,
    18, 12, 29, 5, 21, 10, 3, 24
]

PC2 = [
    13, 16, 10, 23,  0,  4,
     2, 27, 14,  5, 20,  9,
    22, 18, 11,  3, 25,  7,
    15,  6, 26, 19, 12,  1,
    40, 51, 30, 36, 46, 54,
    29, 39, 50, 44, 32, 47,
    43, 48, 38, 55, 33, 52,
    45, 41, 49, 35, 28, 31
]

PC2_INV = [
     4,   23,     6,   15,     5,    9,   19, 
    17, None,    11,    2,    14,   22,    0,
     8,   18,     1, None,    13,   21,   10,
  None,   12,     3, None,    16,   20,    7,
    46,   30,    26,   47,    34,   40, None,
    45,   27,  None,   38,    31,   24,   43,
  None,   36,    33,   42,    28,   35,   37,
    44,   32,    25,   41,  None,   29,   39
]

EXP = [
    31, 0, 1, 2, 3, 4,
    3, 4, 5, 6, 7, 8,
    7, 8, 9, 10, 11, 12,
    11, 12, 13, 14, 15, 16,
    15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24,
    23, 24, 25, 26, 27, 28,
    27, 28, 29, 30, 31, 0
]

S_BOX_TABLE = [
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]


class DESLeakageHelper(object):
    #Name of DES Model
    name = 'DES Leakage Model (unnamed)'
    #c model enumeration value, if a C model exists for this device
    c_model_enum_value = None
    c_model_enum_name = None

    def pad(self, bin_str):
        padding_length = (BLOCK_SIZE - len(bin_str) % BLOCK_SIZE) % BLOCK_SIZE
        return bin_str + '0' * padding_length

    def hex_to_bin(self, hex_str):
        return f'{int(hex_str, 16):0{len(hex_str) * 4}b}'

    def gen_subkeys(self, key):
        key_permutation = self.permute(key, PC1)
        lk, rk = self.split_block(key_permutation)

        self.fprint('KEY', key, 8)
        self.fprint('KEY PERMUTATION', key_permutation, 7)
        subkeys = []

        for n_shifts in KEY_SCHEDULE:
            lk, rk = self.left_rotate([lk, rk], n_shifts)
            compression_permutation = self.permute(lk + rk, PC2)
            subkeys.append(compression_permutation)

        return subkeys

    def get_round_key(self, inputkey, inputround, desiredround, returnSubkeys=True):
        if inputround == 0:
            inputkey = [str(v) for v in bytearray2binarylist(inputkey, 8)]
            key = [int(v) for v in self.permute(inputkey, PC1)]
        else:
            inputkey = [str(v) for v in bytearray2binarylist(inputkey, 6)]
            key = [int(v) if v != '?' else v for v in self.permute(inputkey, PC2_INV)]
        i = inputround
        L = key[:28]
        R = key[28:]
        while i < desiredround:
            i += 1
            j = 0
            # Perform circular left shifts
            while j < KEY_SCHEDULE[i]:
                L.append(L[0])
                del L[0]

                R.append(R[0])
                del R[0]

                j += 1
        while i > desiredround:
            # Perform circular right shifts
            j = 0
            while j < KEY_SCHEDULE[i]:
                L.insert(0,L[27])
                del L[28]

                R.insert(0,R[27])
                del R[28]

                j += 1
            i -= 1

        lr = [str(v) for v in (L+R)]

        if desiredround==0:
            key = [int(v) if v != '?' else v for v in self.permute(lr, PC1_INV)]
            return binarylist2bytearray(key, 8) if returnSubkeys else key
        else:
            key = [int(v) if v != '?' else v for v in self.permute(lr, PC2)]
            return binarylist2bytearray(key, 6) if returnSubkeys else key

    def fprint(self, text, value, nbits, type=16):
        if not debug:
            return
        
        binlist = wrap(value, nbits)
        print(f'{text:>22}: {value}')
        print(f'{"":>24}',end='')
        for bin in binlist:
            if type == 10:
                print(int(bin,2), end = ' ')
            else:
                print(f'{int(bin,2):02X}', end = ' ')
        print()


    def split_block(self, block):
        mid = len(block) // 2
        return block[:mid], block[mid:]


    def left_rotate(self, blocks, n_shifts):
        return [block[n_shifts:] + block[:n_shifts] for block in blocks]


    def permute(self, block, table):
        return ''.join(block[i] if i is not None else '?' for i in table)


    def s_box(self, block):
        output = ''
        for i in range(8):
            sub_str = block[i * 6:i * 6 + 6]
            row = int(sub_str[0] + sub_str[-1], 2)
            column = int(sub_str[1:5], 2)
            output += f'{S_BOX_TABLE[i][row][column]:04b}'
        return output


    def xor(self, block_1, block_2):
        return f'{int(block_1, 2) ^ int(block_2, 2):0{len(block_1)}b}'


    def round(self, input_block, subkey):
        l, r                  = self.split_block(input_block)
        expansion_permutation = self.permute(r, EXP)
        xor_with_subkey       = self.xor(expansion_permutation, subkey)
        s_box_output          = self.s_box(xor_with_subkey)
        p_box_output          = self.permute(s_box_output, P_BOX)
        xor_with_left         = self.xor(p_box_output, l)
        output                = r + xor_with_left

        self.fprint('INPUT', f'{l} {r}', 8)
        self.fprint('SUBKEY', subkey, 6)
        # self.fprint('EXPANSION PERMUTATION', expansion_permutation, 6)
        # self.fprint('XOR', xor_with_subkey,)
        self.fprint('S-BOX OUTPUT', s_box_output, 4, 10)
        # self.fprint('P-BOX PERMUTATION', p_box_output, 4)
        # self.fprint('XOR', xor_with_left, 4)
        # self.fprint('SWAP', f'{r} {xor_with_left}', 8)
        self.fprint('OUTPUT', output, 8)

        return output, s_box_output


    def get_round_1_output(self, input, subkey):       
        # IP
        ip_input = self.permute(input, IP)

        # Run round 1 and return results
        return self.round(ip_input, subkey)

    def get_round_2_output(self, input, subkey):
        # Run round 1 and return results
        return self.round(input, subkey)


class SBox_1_output(DESLeakageHelper):
    name = 'HW: SBoxes Output, First Round'
    c_model_enum_value = 0
    c_model_enum_name = 'LEAK_HW_SBOXOUT_FIRSTROUND'

    def leakage(self, pt, ct, key, bnum):
        # Get round 1 output
        pt    = self.hex_to_bin(''.join(f'{i:02X}' for i in pt))
        key    = f'{key[bnum] << ((7-bnum) * 6):048b}'

        output, s_box_output = self.get_round_1_output(pt, key)

        # Return the requested sbox output byte
        s_box_output = int(wrap(s_box_output, 4)[bnum],2)
        return s_box_output


class SBox_2_output(DESLeakageHelper):
    name = 'HW: SBoxes Output, Second Round'
    c_model_enum_value = 0
    c_model_enum_name = 'LEAK_HW_SBOXOUT_SECONDROUND'

    def leakage(self, pt, ct, key, bnum):
        # Get round 1 output
        pt    = self.hex_to_bin(''.join(f'{i:02X}' for i in pt))
        key    = f'{key[bnum] << ((7-bnum) * 6):048b}'

        output, s_box_output = self.get_round_2_output(pt, key)

        # Return the requested sbox output byte
        s_box_output = int(wrap(s_box_output, 4)[bnum],2)
        return s_box_output

enc_list = [SBox_1_output, SBox_2_output]
dec_list = []


class DES(ModelsBase):
    _name = 'DES'

    hwModels = OrderedDict((mod.name, mod) for mod in (enc_list + dec_list))

    _has_prev = False


    def __init__(self, model=SBox_1_output, bitmask=0xFF):
        ModelsBase.__init__(self, 8, 64, model=model)
        self.numRoundKeys = 16
        self._mask = bitmask


    def _updateHwModel(self):
        """" Re-implement this to update leakage model """
        self.modelobj = None

        #Check if they passed an object...
        if isinstance(self.model, DESLeakageHelper):
            self.modelobj = self.model

        #Check if they passed a class...
        elif inspect.isclass(self.model) and issubclass(self.model, DESLeakageHelper):
            self.modelobj = self.model()

        #Otherwise it's probably one of these older keys (kept for backwards-compatability)
        else:
            for mod in self.hwModels:
                if (mod.c_model_enum_value == self.model) or (mod.name == self.model):
                    self.modelobj = mod()
                    break

        if self.modelobj is None:
            raise ValueError("Invalid model: %s" % str(self.model))


    def process_known_key(self, inpkey):
        return self.modelobj.get_round_key(inpkey, 0, 1)


    def leakage(self, pt, ct, guess, bnum, state):
        try:
            # Make a copy so we don't screw with anything...
            key = list(state['knownkey'])
        except:
            # We don't log due to time-sensitive nature... but if state doesn't have "knownkey" will result in
            # unknown knownkey which causes some attacks to fail. Possibly should make this some sort of
            # flag to indicate we want to ignore the problem?
            key = [None] * 8

        # Guess can be 'none' if we want to use original key as-is
        if guess is not None:
            key[bnum] = guess

        # Get intermediate value
        intermediate_value = self.modelobj.leakage(pt, ct, key, bnum)

        # For bit-wise attacks, mask off specific bit value
        intermediate_value = self._mask & intermediate_value

        # Return HW of guess
        return self.HW[intermediate_value]


def main():
    global attack_round
    attack_round = 2
    des = DESLeakageHelper()
    # input = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]
    # input = [0x0a, 0x8c, 0x29, 0x7a, 0xf3, 0x61, 0x18, 0x15]
    # key = [0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6]
    input = des.hex_to_bin('DE55A6780229FD4F')
    key = '221030213238073F'

    key = '010110011010000001001010111101011110011111101010'

    # output, _ = des.get_round_1_output(input, key)
    des.get_round_2_output(input, key)

if __name__ == '__main__':
    main()