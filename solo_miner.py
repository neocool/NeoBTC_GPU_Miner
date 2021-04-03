from bitcoin_api.transactions import Transaction, int_to_varbyteint
import hashlib
import codecs
import struct
import time
import requests, json
from numba import njit, int64, cuda 
import numpy as np
import math

############### CUDA Functions #######################
@cuda.jit(device=True)
def rightrotate_numba(x, c):
    """ Right rotate the number x by c bytes."""
    x &= 0xFFFFFFFF
    return ((x >> c) | (x << (32 - c))) & 0xFFFFFFFF

@cuda.jit(device=True)
def Rightshift_numba(x, c):
    """ Right shift the number x by c bytes."""
    return x >> c

@cuda.jit(device=True)
def leftrotate_numba(x, c):
    """ Left rotate the number x by c bytes."""
    x &= 0xFFFFFFFF
    return ((x << c) | (x >> (32 - c))) & 0xFFFFFFFF

@cuda.jit(device=True)
def reverse_endian_number(num):    
    num =   (num &  0x000000FF)   << 24 | ((num &  0x0000FF00) >>8 ) << 16 | ((num &  0x00FF0000) >>16) << 8 | ((num &  0xFF000000) >>24)
    return num

@cuda.jit(device=True)
def first_stage(w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15):
    s0 = (rightrotate_numba(w1 , 7) ^ rightrotate_numba(w1, 18) ^ Rightshift_numba(w1, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w14 , 17) ^ rightrotate_numba(w14, 19) ^ Rightshift_numba(w14, 10)) & 0xFFFFFFFF 
    w16 = (w0  + s0 + w9+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w2 , 7) ^ rightrotate_numba(w2, 18) ^ Rightshift_numba(w2, 3)) & 0xFFFFFFFF      
    s1 = (rightrotate_numba(w15 , 17) ^ rightrotate_numba(w15, 19) ^ Rightshift_numba(w15, 10)) & 0xFFFFFFFF 
    w17 = (w1  + s0 + w10+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w3 , 7) ^ rightrotate_numba(w3, 18) ^ Rightshift_numba(w3, 3)) & 0xFFFFFFFF      
    s1 = (rightrotate_numba(w16 , 17) ^ rightrotate_numba(w16, 19) ^ Rightshift_numba(w16, 10)) & 0xFFFFFFFF 
    w18 = (w2  + s0 + w11+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w4 , 7) ^ rightrotate_numba(w4, 18) ^ Rightshift_numba(w4, 3)) & 0xFFFFFFFF      
    s1 = (rightrotate_numba(w17 , 17) ^ rightrotate_numba(w17, 19) ^ Rightshift_numba(w17, 10)) & 0xFFFFFFFF 
    w19 = (w3  + s0 + w12+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w5 , 7) ^ rightrotate_numba(w5, 18) ^ Rightshift_numba(w5, 3)) & 0xFFFFFFFF      
    s1 = (rightrotate_numba(w18 , 17) ^ rightrotate_numba(w18, 19) ^ Rightshift_numba(w18, 10)) & 0xFFFFFFFF 
    w20 = (w4  + s0 + w13+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w6 , 7) ^ rightrotate_numba(w6, 18) ^ Rightshift_numba(w6, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w19 , 17) ^ rightrotate_numba(w19, 19) ^ Rightshift_numba(w19, 10)) & 0xFFFFFFFF
    w21 = (w5  + s0 + w14+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w7 , 7) ^ rightrotate_numba(w7, 18) ^ Rightshift_numba(w7, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w20 , 17) ^ rightrotate_numba(w20, 19) ^ Rightshift_numba(w20, 10)) & 0xFFFFFFFF
    w22 = (w6  + s0 + w15+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w8 , 7) ^ rightrotate_numba(w8, 18) ^ Rightshift_numba(w8, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w21 , 17) ^ rightrotate_numba(w21, 19) ^ Rightshift_numba(w21, 10)) & 0xFFFFFFFF
    w23 = (w7  + s0 + w16+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w9 , 7) ^ rightrotate_numba(w9, 18) ^ Rightshift_numba(w9, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w22 , 17) ^ rightrotate_numba(w22, 19) ^ Rightshift_numba(w22, 10)) & 0xFFFFFFFF
    w24 = (w8  + s0 + w17+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w10 , 7) ^ rightrotate_numba(w10, 18) ^ Rightshift_numba(w10, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w23 , 17) ^ rightrotate_numba(w23, 19) ^ Rightshift_numba(w23, 10)) & 0xFFFFFFFF
    w25 = (w9  + s0 + w18+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w11 , 7) ^ rightrotate_numba(w11, 18) ^ Rightshift_numba(w11, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w24 , 17) ^ rightrotate_numba(w24, 19) ^ Rightshift_numba(w24, 10)) & 0xFFFFFFFF
    w26 = (w10  + s0 + w19+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w12 , 7) ^ rightrotate_numba(w12, 18) ^ Rightshift_numba(w12, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w25 , 17) ^ rightrotate_numba(w25, 19) ^ Rightshift_numba(w25, 10)) & 0xFFFFFFFF
    w27 = (w11  + s0 + w20+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w13 , 7) ^ rightrotate_numba(w13, 18) ^ Rightshift_numba(w13, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w26 , 17) ^ rightrotate_numba(w26, 19) ^ Rightshift_numba(w26, 10)) & 0xFFFFFFFF
    w28 = (w12  + s0 + w21+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w14 , 7) ^ rightrotate_numba(w14, 18) ^ Rightshift_numba(w14, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w27 , 17) ^ rightrotate_numba(w27, 19) ^ Rightshift_numba(w27, 10)) & 0xFFFFFFFF
    w29 = (w13  + s0 + w22+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w15 , 7) ^ rightrotate_numba(w15, 18) ^ Rightshift_numba(w15, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w28 , 17) ^ rightrotate_numba(w28, 19) ^ Rightshift_numba(w28, 10)) & 0xFFFFFFFF
    w30 = (w14  + s0 + w23+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w16 , 7) ^ rightrotate_numba(w16, 18) ^ Rightshift_numba(w16, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w29 , 17) ^ rightrotate_numba(w29, 19) ^ Rightshift_numba(w29, 10)) & 0xFFFFFFFF
    w31 = (w15  + s0 + w24+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w17 , 7) ^ rightrotate_numba(w17, 18) ^ Rightshift_numba(w17, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w30 , 17) ^ rightrotate_numba(w30, 19) ^ Rightshift_numba(w30, 10)) & 0xFFFFFFFF
    w32 = (w16  + s0 + w25+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w18 , 7) ^ rightrotate_numba(w18, 18) ^ Rightshift_numba(w18, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w31 , 17) ^ rightrotate_numba(w31, 19) ^ Rightshift_numba(w31, 10)) & 0xFFFFFFFF
    w33 = (w17  + s0 + w26+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w19 , 7) ^ rightrotate_numba(w19, 18) ^ Rightshift_numba(w19, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w32 , 17) ^ rightrotate_numba(w32, 19) ^ Rightshift_numba(w32, 10)) & 0xFFFFFFFF
    w34 = (w18  + s0 + w27+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w20 , 7) ^ rightrotate_numba(w20, 18) ^ Rightshift_numba(w20, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w33 , 17) ^ rightrotate_numba(w33, 19) ^ Rightshift_numba(w33, 10)) & 0xFFFFFFFF
    w35 = (w19  + s0 + w28+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w21 , 7) ^ rightrotate_numba(w21, 18) ^ Rightshift_numba(w21, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w34 , 17) ^ rightrotate_numba(w34, 19) ^ Rightshift_numba(w34, 10)) & 0xFFFFFFFF
    w36 = (w20  + s0 + w29+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w22 , 7) ^ rightrotate_numba(w22, 18) ^ Rightshift_numba(w22, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w35 , 17) ^ rightrotate_numba(w35, 19) ^ Rightshift_numba(w35, 10)) & 0xFFFFFFFF
    w37 = (w21  + s0 + w30+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w23 , 7) ^ rightrotate_numba(w23, 18) ^ Rightshift_numba(w23, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w36 , 17) ^ rightrotate_numba(w36, 19) ^ Rightshift_numba(w36, 10)) & 0xFFFFFFFF
    w38 = (w22  + s0 + w31+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w24 , 7) ^ rightrotate_numba(w24, 18) ^ Rightshift_numba(w24, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w37 , 17) ^ rightrotate_numba(w37, 19) ^ Rightshift_numba(w37, 10)) & 0xFFFFFFFF
    w39 = (w23  + s0 + w32+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w25 , 7) ^ rightrotate_numba(w25, 18) ^ Rightshift_numba(w25, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w38 , 17) ^ rightrotate_numba(w38, 19) ^ Rightshift_numba(w38, 10)) & 0xFFFFFFFF
    w40 = (w24  + s0 + w33+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w26 , 7) ^ rightrotate_numba(w26, 18) ^ Rightshift_numba(w26, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w39 , 17) ^ rightrotate_numba(w39, 19) ^ Rightshift_numba(w39, 10)) & 0xFFFFFFFF
    w41 = (w25  + s0 + w34+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w27 , 7) ^ rightrotate_numba(w27, 18) ^ Rightshift_numba(w27, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w40 , 17) ^ rightrotate_numba(w40, 19) ^ Rightshift_numba(w40, 10)) & 0xFFFFFFFF
    w42 = (w26  + s0 + w35+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w28 , 7) ^ rightrotate_numba(w28, 18) ^ Rightshift_numba(w28, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w41 , 17) ^ rightrotate_numba(w41, 19) ^ Rightshift_numba(w41, 10)) & 0xFFFFFFFF
    w43 = (w27  + s0 + w36+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w29 , 7) ^ rightrotate_numba(w29, 18) ^ Rightshift_numba(w29, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w42 , 17) ^ rightrotate_numba(w42, 19) ^ Rightshift_numba(w42, 10)) & 0xFFFFFFFF
    w44 = (w28  + s0 + w37+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w30 , 7) ^ rightrotate_numba(w30, 18) ^ Rightshift_numba(w30, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w43 , 17) ^ rightrotate_numba(w43, 19) ^ Rightshift_numba(w43, 10)) & 0xFFFFFFFF
    w45 = (w29  + s0 + w38+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w31 , 7) ^ rightrotate_numba(w31, 18) ^ Rightshift_numba(w31, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w44 , 17) ^ rightrotate_numba(w44, 19) ^ Rightshift_numba(w44, 10)) & 0xFFFFFFFF
    w46 = (w30  + s0 + w39+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w32 , 7) ^ rightrotate_numba(w32, 18) ^ Rightshift_numba(w32, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w45 , 17) ^ rightrotate_numba(w45, 19) ^ Rightshift_numba(w45, 10)) & 0xFFFFFFFF
    w47 = (w31  + s0 + w40+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w33 , 7) ^ rightrotate_numba(w33, 18) ^ Rightshift_numba(w33, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w46 , 17) ^ rightrotate_numba(w46, 19) ^ Rightshift_numba(w46, 10)) & 0xFFFFFFFF
    w48 = (w32  + s0 + w41+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w34 , 7) ^ rightrotate_numba(w34, 18) ^ Rightshift_numba(w34, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w47 , 17) ^ rightrotate_numba(w47, 19) ^ Rightshift_numba(w47, 10)) & 0xFFFFFFFF
    w49 = (w33  + s0 + w42+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w35 , 7) ^ rightrotate_numba(w35, 18) ^ Rightshift_numba(w35, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w48 , 17) ^ rightrotate_numba(w48, 19) ^ Rightshift_numba(w48, 10)) & 0xFFFFFFFF
    w50 = (w34  + s0 + w43+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w36 , 7) ^ rightrotate_numba(w36, 18) ^ Rightshift_numba(w36, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w49 , 17) ^ rightrotate_numba(w49, 19) ^ Rightshift_numba(w49, 10)) & 0xFFFFFFFF
    w51 = (w35  + s0 + w44+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w37 , 7) ^ rightrotate_numba(w37, 18) ^ Rightshift_numba(w37, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w50 , 17) ^ rightrotate_numba(w50, 19) ^ Rightshift_numba(w50, 10)) & 0xFFFFFFFF
    w52 = (w36  + s0 + w45+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w38 , 7) ^ rightrotate_numba(w38, 18) ^ Rightshift_numba(w38, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w51 , 17) ^ rightrotate_numba(w51, 19) ^ Rightshift_numba(w51, 10)) & 0xFFFFFFFF
    w53 = (w37  + s0 + w46+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w39 , 7) ^ rightrotate_numba(w39, 18) ^ Rightshift_numba(w39, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w52 , 17) ^ rightrotate_numba(w52, 19) ^ Rightshift_numba(w52, 10)) & 0xFFFFFFFF
    w54 = (w38  + s0 + w47+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w40 , 7) ^ rightrotate_numba(w40, 18) ^ Rightshift_numba(w40, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w53 , 17) ^ rightrotate_numba(w53, 19) ^ Rightshift_numba(w53, 10)) & 0xFFFFFFFF
    w55 = (w39  + s0 + w48+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w41 , 7) ^ rightrotate_numba(w41, 18) ^ Rightshift_numba(w41, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w54 , 17) ^ rightrotate_numba(w54, 19) ^ Rightshift_numba(w54, 10)) & 0xFFFFFFFF
    w56 = (w40  + s0 + w49+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w42 , 7) ^ rightrotate_numba(w42, 18) ^ Rightshift_numba(w42, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w55 , 17) ^ rightrotate_numba(w55, 19) ^ Rightshift_numba(w55, 10)) & 0xFFFFFFFF
    w57 = (w41  + s0 + w50+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w43 , 7) ^ rightrotate_numba(w43, 18) ^ Rightshift_numba(w43, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w56 , 17) ^ rightrotate_numba(w56, 19) ^ Rightshift_numba(w56, 10)) & 0xFFFFFFFF
    w58 = (w42  + s0 + w51+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w44 , 7) ^ rightrotate_numba(w44, 18) ^ Rightshift_numba(w44, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w57 , 17) ^ rightrotate_numba(w57, 19) ^ Rightshift_numba(w57, 10)) & 0xFFFFFFFF
    w59 = (w43  + s0 + w52+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w45 , 7) ^ rightrotate_numba(w45, 18) ^ Rightshift_numba(w45, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w58 , 17) ^ rightrotate_numba(w58, 19) ^ Rightshift_numba(w58, 10)) & 0xFFFFFFFF
    w60 = (w44  + s0 + w53+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w46 , 7) ^ rightrotate_numba(w46, 18) ^ Rightshift_numba(w46, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w59 , 17) ^ rightrotate_numba(w59, 19) ^ Rightshift_numba(w59, 10)) & 0xFFFFFFFF
    w61 = (w45  + s0 + w54+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w47 , 7) ^ rightrotate_numba(w47, 18) ^ Rightshift_numba(w47, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w60 , 17) ^ rightrotate_numba(w60, 19) ^ Rightshift_numba(w60, 10)) & 0xFFFFFFFF
    w62 = (w46  + s0 + w55+ s1) & 0xFFFFFFFF
    s0 = (rightrotate_numba(w48 , 7) ^ rightrotate_numba(w48, 18) ^ Rightshift_numba(w48, 3)) & 0xFFFFFFFF
    s1 = (rightrotate_numba(w61 , 17) ^ rightrotate_numba(w61, 19) ^ Rightshift_numba(w61, 10)) & 0xFFFFFFFF
    w63 = (w47  + s0 + w56+ s1) & 0xFFFFFFFF
    return w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35,w36,w37,w38,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48,w49,w50,w51,w52,w53,w54,w55,w56,w57,w58,w59,w60,w61,w62,w63

@cuda.jit(device=True)
def second_stage(h0, h1, h2, h3, h4, h5, h6, h7,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35,w36,w37,w38,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48,w49,w50,w51,w52,w53,w54,w55,w56,w57,w58,w59,w60,w61,w62,w63):
    a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1116352408 + w0) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1899447441 + w1) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3049323471 + w2) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3921009573 + w3) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +961987163 + w4) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1508970993 + w5) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2453635748 + w6) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2870763221 + w7) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3624381080 + w8) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +310598401 + w9) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +607225278 + w10) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1426881987 + w11) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1925078388 + w12) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2162078206 + w13) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2614888103 + w14) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3248222580 + w15) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3835390401 + w16) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +4022224774 + w17) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +264347078 + w18) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +604807628 + w19) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +770255983 + w20) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1249150122 + w21) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1555081692 + w22) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1996064986 + w23) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2554220882 + w24) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2821834349 + w25) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2952996808 + w26) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3210313671 + w27) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3336571891 + w28) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3584528711 + w29) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +113926993 + w30) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +338241895 + w31) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +666307205 + w32) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +773529912 + w33) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1294757372 + w34) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1396182291 + w35) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1695183700 + w36) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1986661051 + w37) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2177026350 + w38) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2456956037 + w39) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2730485921 + w40) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2820302411 + w41) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3259730800 + w42) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3345764771 + w43) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3516065817 + w44) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3600352804 + w45) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +4094571909 + w46) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +275423344 + w47) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +430227734 + w48) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +506948616 + w49) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +659060556 + w50) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF 
    temp1 = (h + S1 + ch +883997877 + w51) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +958139571 + w52) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1322822218 + w53) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1537002063 + w54) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1747873779 + w55) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +1955562222 + w56) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2024104815 + w57) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2227730452 + w58) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2361852424 + w59) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2428436474 + w60) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +2756734187 + w61) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3204031479 + w62) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF
    S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
    ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
    temp1 = (h + S1 + ch +3329325298 + w63) & 0xFFFFFFFF
    S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
    maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
    temp2 = (S0 + maj) & 0xFFFFFFFF
    h,g,f,e,d,c,b,a  = g,f,e,(d + temp1) & 0xFFFFFFFF,c,b,a,(temp1 + temp2) & 0xFFFFFFFF

    # Add this chunk's hash to result so far:
    h0 = (h0 + a) & 0xFFFFFFFF        
    h1 = (h1 + b) & 0xFFFFFFFF
    h2 = (h2 + c) & 0xFFFFFFFF
    h3 = (h3 + d) & 0xFFFFFFFF
    h4 = (h4 + e) & 0xFFFFFFFF
    h5 = (h5 + f) & 0xFFFFFFFF
    h6 = (h6 + g) & 0xFFFFFFFF
    h7 = (h7 + h) & 0xFFFFFFFF   
    return h0, h1, h2, h3, h4, h5, h6, h7

@cuda.jit
def cuda_miner(d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,nonce_array):
    pos = cuda.grid(1)
    start_nonce =  pos * 1024
    end_nonce = start_nonce+ 1024    
    if end_nonce > 4294967295:
        end_nonce = 4294967295 
    
    for nonce in range(start_nonce,end_nonce):
        h0, h1, h2, h3, h4, h5, h6, h7 = 0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19       
                
        w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15 = d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15

        w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35,w36,w37,w38,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48,w49,w50,w51,w52,w53,w54,w55,w56,w57,w58,w59,w60,w61,w62,w63 = first_stage(w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15)

        # 2.e. Main loop, cf. https://tools.ietf.org/html/rfc6234
        h0, h1, h2, h3, h4, h5, h6, h7 = second_stage(h0, h1, h2, h3, h4, h5, h6, h7,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35,w36,w37,w38,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48,w49,w50,w51,w52,w53,w54,w55,w56,w57,w58,w59,w60,w61,w62,w63)

        ######################### Second Loop ########################################################################## 
        w0,w1,w2,w4,w13,w14,w15 = d16,d17,d18, 2147483648,0,0,640
        w3,w5,w6,w7,w8,w9,w10,w11,w12 = reverse_endian_number(nonce),0,0,0,0,0,0,0,0 
        w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35,w36,w37,w38,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48,w49,w50,w51,w52,w53,w54,w55,w56,w57,w58,w59,w60,w61,w62,w63 = first_stage(w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15)

        # 2.e. Main loop, cf. https://tools.ietf.org/html/rfc6234
        h0, h1, h2, h3, h4, h5, h6, h7 = second_stage(h0, h1, h2, h3, h4, h5, h6, h7,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35,w36,w37,w38,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48,w49,w50,w51,w52,w53,w54,w55,w56,w57,w58,w59,w60,w61,w62,w63)

        ##############################  Reversed #############################       
        
        w9,w10,w11,w12 = 0,0,0,0
        w0,w1,w2,w3,w4,w5,w6,w7 = h0 ,h1 , h2 , h3 , h4 ,h5 ,h6 , h7
        w8 = 2147483648
        w15 = 256
        w14 = 0
        h0, h1, h2, h3, h4, h5, h6, h7 = 0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
        w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35,w36,w37,w38,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48,w49,w50,w51,w52,w53,w54,w55,w56,w57,w58,w59,w60,w61,w62,w63 = first_stage(w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15)

        # 2.d. Initialize hash value for this chunk
        
        # 2.e. Main loop, cf. https://tools.ietf.org/html/rfc6234
        h0, h1, h2, h3, h4, h5, h6, h7 = second_stage(h0, h1, h2, h3, h4, h5, h6, h7,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35,w36,w37,w38,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48,w49,w50,w51,w52,w53,w54,w55,w56,w57,w58,w59,w60,w61,w62,w63)
        if h7 == 0:                       
            if h6 == 0:                            
                nonce_array[0]  =  nonce

@njit
def hex_to_int(hexadecimal ):    
    conversion_table = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10 , 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15}
    power = 7    
    decimal = 0
    for digit in hexadecimal:
        decimal += conversion_table[digit]*16**power
        power -= 1
    return decimal

@njit
def int_to_hex(decimal):
    conversion_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a' , 'b', 'c', 'd', 'e', 'f']
    hexadecimal = ''
    while(decimal>0):
        remainder = decimal%16
        hexadecimal = conversion_table[remainder]+ hexadecimal
        decimal = decimal//16
    
    value = ""    
    for i in range(8 - len(hexadecimal)):
        value += "0"
    value += hexadecimal    

    return value

############# CPU Functions #####################
def get_cuda_device_info():
    gpu = cuda.get_current_device()
    print("name = %s" % gpu.name)
    print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
    print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
    print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
    print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
    print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
    print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
    print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
    print("maxSharedMemoryPerBlock = %s" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
    print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
    print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
    print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
    print("warpSize = %s" % str(gpu.WARP_SIZE))
    print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
    print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
    print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))

def construct_block_header(version, prevHash, merkleRoot, t, bits, nonce):
    # construct new header
    header = (struct.pack("<L", version) +
              codecs.decode(prevHash, "hex")[::-1] +
              codecs.decode(merkleRoot, "hex")[::-1] +
              struct.pack("<L", t) +
              struct.pack("<L", bits) +
              struct.pack("<L", nonce))    
    return header

def calculate_difficulty(bits):
    # calculate difficulty from bits
    exponent = bits >> 24
    mantissa = bits & 0x00FFFFFF
    diff = mantissa << (8 * (exponent - 3))
    #print(reversedDigest < diff.to_bytes(32, 'big'))
    diffInBytes = diff.to_bytes(32, 'big')
    return diffInBytes

def gen_block(version,prevHash,merkleRoot,t,bits,nonce, transaction_data_list):  
    magic_num = bytes.fromhex("f9beb4d9")
    bits = int(bits,16) 

    header = (struct.pack("<L", version) +
              codecs.decode(prevHash, "hex")[::-1] +
              codecs.decode(merkleRoot, "hex")[::-1] +
              struct.pack("<L", t) +
              struct.pack("<L", bits) +
              struct.pack("<L", nonce)
              )    
    tx_count = int_to_varbyteint(len(transaction_data_list))    
    txs= ""
    for i in range(len(transaction_data_list)):
        txs += transaction_data_list[i]        
    tx_data = (tx_count+
              codecs.decode(txs, "hex")           
              )    
    block_data = header + tx_data
    block_size = bytearray(len(block_data).to_bytes(4, 'little'))
    block = block_data
    return block

def read_cfg():
    with open("confg.cfg","r") as file_bject:
        lines = file_bject.readlines()
        btcNode_ip = lines[0].split(":")[1].strip("\n")
        btcNode_port = lines[1].split(":")[1].strip("\n")
        btcNode_user = lines[2].split(":")[1].strip("\n")
        btcNode_pass = lines[3].split(":")[1].strip("\n")
        btc_public_address = lines[4].split(":")[1].strip("\n")
        miner_id = int(lines[5].split(":")[1].strip("\n"))
        debug_level = int(lines[6].split(":")[1].strip("\n"))
        

    return btcNode_ip, btcNode_port,btcNode_user,btcNode_pass,btc_public_address,miner_id,debug_level

def rpc_call(method, parameter,btcNode_ip, btcNode_port,btcNode_user,btcNode_pass):

    serverURL = 'http://' + btcNode_ip+ ':' + btcNode_port
    basicAuthCredentials = ( btcNode_user , btcNode_pass)
    
    headers = {'content-type': 'application/json'}    
    if len(parameter) >1:
        payload = json.dumps({"method": method, "params": parameter, "jsonrpc": "2.0"})
    elif len(parameter) ==1:
        payload = json.dumps({"method": method, "params": parameter, "jsonrpc": "2.0"})
    
    response = requests.post(serverURL, headers=headers, auth=basicAuthCredentials, data=payload)
    return response

def get_block_info(btcNode_ip, btcNode_port,btcNode_user,btcNode_pass):

    serverURL = 'http://' + btcNode_ip+ ':' + btcNode_port
    basicAuthCredentials = ( btcNode_user , btcNode_pass)
    
    headers = {'content-type': 'application/json'}
    payload = json.dumps({"method": 'getblocktemplate', "params": [{"rules": ["segwit"]}], "jsonrpc": "2.0"})
    response = requests.post(serverURL, headers=headers, auth=basicAuthCredentials, data=payload)
 
    return response.json()['result']

def get_tx_hashlist(coinbase_transaction, transactions):
    hashlist = []
    hashlist.append(coinbase_transaction)
    for i in range(len(transactions)):
        hashlist.append(transactions[i]['txid'])
    return hashlist

def prepare_mining_data(mining_block_data,merkleRoot, nonce):    
    version = mining_block_data['version']
    prevHash = mining_block_data['previousblockhash']
    t = mining_block_data['curtime']    
    bits = int(mining_block_data['bits'],16)  
    
    header = construct_block_header(version, prevHash, merkleRoot, t, bits, nonce)    
    digest = hashlib.sha256(header).digest()
    reversedDigest = hashlib.sha256(digest).digest()[::-1]   

    #print("version,     prevHash,   merkleRoot,     time,   bits,    nonce")
    #print(str([version, prevHash, merkleRoot, t, bits, nonce]))
    
    return reversedDigest

def merkle(hashList):
    if len(hashList) == 1:
        return hashList[0]
    newHashList = []
    # Process pairs. For odd length, the last is skipped
    for i in range(0, len(hashList)-1, 2):
        newHashList.append(hash2(hashList[i], hashList[i+1]))
    if len(hashList) % 2 == 1: # odd, hash last item twice
        newHashList.append(hash2(hashList[-1], hashList[-1]))
    return merkle(newHashList)

def hash2(a, b):
    # Reverse inputs before and after hashing
    # due to big-endian / little-endian nonsense
    
    a1 = bytearray.fromhex(a)
    a1.reverse()
    b1 = bytearray.fromhex(b)
    b1.reverse()
    a1 = ''.join(format(x, '02x') for x in a1 )
    b1 = ''.join(format(x, '02x') for x in b1 )
    
    #Concatenate these values and calculate a sha256 digest from the binary data
    contcat = a1+b1     

    concatb = bytearray.fromhex(contcat)  
    h = hashlib.sha256(concatb).hexdigest()    
        
    #Convert this value to binary and perform another sha256 operation on the output
    h = bytearray.fromhex(h)  
    h = hashlib.sha256(h).hexdigest()    

    #Finally, reverse the order from little to big endian.
    h = bytearray.fromhex(h)
    h.reverse()
    h = ''.join(format(x, '02x') for x in h )    
    
    return h

def gen_coinbase_transaction(coinbasevalue,height,btc_public_address,miner_id=1):       
    #raw_transaction = "020000000001010000000000000000000000000000000000000000000000000000000000000000ffffffff200331410a04c5ee3660626a2f42696e616e63652f062188f300003575e8df3600ffffffff0239fe5729000000001976a914887d65fdc11cd8151c92530f323aada252792dc888ac0000000000000000266a24aa21a9ed2c425421aa881a2f92152f38f4cc4413837e6f3ee38a8e44f3620c91ce117cc80120000000000000000000000000000000000000000000000000000000000000000000000000"
    #print(data)
    #parameter1 = [{"txid":"0000000000000000000000000000000000000000000000000000000000000000","vout":0}]    
    #parameter2 = [{"34Gs8jhfjnpULE1b6pNfYqZGzT7xcBaaRB":coinbasevalue_inbtc} ]       
    #parameter = [parameter1, parameter2]    
     
    t = Transaction(coinbase=True, block_height=height,version=1,miner_id=miner_id)      
    t.add_input(prev_txid ="0000000000000000000000000000000000000000000000000000000000000000",output_n=0)
    t.add_output(value=coinbasevalue,address=btc_public_address,output_n=1)    
    t.sign_and_update_coinbase()
    
    #Decode Transaction
    coinbase_transaction = t.raw_coinbase().hex()
    #print(coinbase_transaction)
    t.sign_and_update_coinbase()
    #parameter = [coinbase_transaction]    
    #response = rpc_call('decoderawtransaction', parameter)   
    #print(response)
    #print(response.text)
    #print(response.json()['result'])
    return t.txid, coinbase_transaction

def submit_work(block):
    print("submitting work")
    #input()       
    with open("submitted_blocks.txt","a") as file_object:
        file_object.write(block.hex())
    mydata = block.hex()
    results = rpc_call("submitblock",[mydata])
    print(results)
    print(results.text)

def ml_header(mining_block_data,merkleRoot):    
    version = mining_block_data['version']
    prevHash = mining_block_data['previousblockhash']
    t = mining_block_data['curtime']    
    bits = int(mining_block_data['bits'],16)
    
    # construct new header
    header = (struct.pack("<L", version) +
              codecs.decode(prevHash, "hex")[::-1] +
              codecs.decode(merkleRoot, "hex")[::-1] +
              struct.pack("<L", t) +
              struct.pack("<L", bits)
              )
    #digest = hashlib.sha256(header).digest()
    #reversedDigest = hashlib.sha256(digest).digest()[::-1]
    return header.hex()

def check_winner(nonce,header,bits):
    #1302976280
    #416786765
    print("Checking nonce: " + str(nonce))    
    #nonce = int.from_bytes(int(nonce).to_bytes(4,"little"),"big")
    nonce = struct.pack("<L", nonce).hex()
    diff = calculate_difficulty(bits)
    header = header + nonce
    digest = hashlib.sha256(bytearray.fromhex(header)).digest()
    reversedDigest = hashlib.sha256(digest).digest()[::-1]     
    if  reversedDigest < diff:        
        return True        
    else:        
        return False

def prepare_nonce_array(nonce_array, nonce_start, nonce_end ):
    for nonce in range (nonce_start,nonce_end):
        nonce = int(nonce.to_bytes(4, byteorder="little").hex(),16)  
        nonce_array = np.append(nonce_array,[[nonce]],axis=0)        
    nonce_array = np.delete(nonce_array, [0] ,axis=0)
    filePath = "nonce_arrays/nonce_"+str(nonce_start)+".txt"
    #nonce_array.tofile(filePath)    overwrite exsisting files!!!!!!!!!!
    return nonce_array

def prepare_w1(dataarray, init_w, diff):    
    k = [   0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
    
    diff = diff[16:24]
    diff = hex_to_int(diff)    
    w = init_w
    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h4 = 0x510e527f
    h5 = 0x9b05688c
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19         
    for x in range(16):            
        w[x] = dataarray[x]

    for i in range(16, 64):                      
        s0 = (rightrotate_numba(w[i-15] , 7) ^ rightrotate_numba(w[i-15] , 18) ^ rightshift_numba(w[i-15] , 3)) & 0xFFFFFFFF            
        s1 = (rightrotate_numba(w[i-2], 17) ^ rightrotate_numba(w[i-2], 19) ^ rightshift_numba(w[i-2], 10)) & 0xFFFFFFFF            
        w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF

    # 2.d. Initialize hash value for this chunk
    a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7
    # 2.e. Main loop, cf. https://tools.ietf.org/html/rfc6234
    for i in range(64):
        S1 = (rightrotate_numba(e, 6) ^ rightrotate_numba(e, 11) ^ rightrotate_numba(e, 25)) & 0xFFFFFFFF
        ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF            
        temp1 = (h + S1 + ch + k[i] + w[i]) & 0xFFFFFFFF                
        S0 = (rightrotate_numba(a, 2) ^ rightrotate_numba(a, 13) ^ rightrotate_numba(a, 22)) & 0xFFFFFFFF
        maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
        temp2 = (S0 + maj) & 0xFFFFFFFF            
        h = g
        g = f
        f = e
        e = (d + temp1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xFFFFFFFF
    # Add this chunk's hash to result so far:
    h0 = (h0 + a) & 0xFFFFFFFF        
    h1 = (h1 + b) & 0xFFFFFFFF
    h2 = (h2 + c) & 0xFFFFFFFF
    h3 = (h3 + d) & 0xFFFFFFFF
    h4 = (h4 + e) & 0xFFFFFFFF
    h5 = (h5 + f) & 0xFFFFFFFF
    h6 = (h6 + g) & 0xFFFFFFFF
    h7 = (h7 + h) & 0xFFFFFFFF      
    
    for z in range(16):
        w[z] = dataarray[z+16]           
          
    w[5],w[6],w[7],w[8],w[9],w[10],w[11],w[12] =h0, h1, h2, h3, h4, h5, h6, h7          
    #w = np.append(w,[diff]) 
    w = np.append(w, diff)
    return w

def mid_state(header,diff):
    dataarray = np.empty(19, dtype=np.uint32)
    dataarray.fill(0)
    for  i in range(19):        
        dataarray[i] = hex_to_int(header[i*8:(i*8)+8])       
    return dataarray.copy()

def copy_to_memory():     
    MAXARRSIZE = 800000000
    MAX_NONCE = 4294967295    
    res_nonce = np.zeros(1,dtype=np.int64)
    res_nonce[0] = -1
    res_nonce = cuda.to_device(res_nonce)    
    #print("Copied Nonce array to memory!")    
    return res_nonce

def get_darray(mining_block_data,merkle_root,x):    
    bits = int(mining_block_data['bits'],16)  
    diff = calculate_difficulty(bits).hex()    
    header = ml_header(mining_block_data,merkle_root) 
    
    #bits = int("170d1f8c",16) 
    #header = "00000020964eb8a03dee5fc64d2c1b66103787c374d7ca903b18040000000000000000007ba6c46f242655c2084449cdcc60a23e1c50ee730c4e8b220b3df5e2f9429d32d2d943608c1f0d17"    
    #diff = "0000000000000000000d1f8c0000000000000000000000000000000000000000"  
    #wining nonce for test: 1302976280 
    
    darray = mid_state(header,diff)      
    return darray, header,bits, diff

def mine_numba(block_height, mining_block_data,merkle_root,nonce_array,debug_level):    
    if debug_level <=1:
        print("Block:" + str(block_height))   
    #GPU mining loop       
    
    # Set the number of threads in a block
    threadsperblock = 128
    blockspergrid = math.ceil(4294967295/ (1024*threadsperblock) ) 
    #Cuda Kernel
    
    darray, header,bits, diff = get_darray(mining_block_data,merkle_root,0)
    #res_array = cuda.device_array_like(nonce_array)    
    d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18 = darray[0:19]
    cuda_miner[blockspergrid, threadsperblock](d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,nonce_array)   
    nonce = nonce_array.copy_to_host()[0]

    #nonce = gpu_mine_thread(work_load_que[0],header,bits,darray)        
    if nonce > -1:
        #nonce = int.from_bytes(int(nonce).to_bytes(4,"little"),"big")
        if check_winner(nonce,header,bits):
            print("Winner Winner Chicken Dinner!  " + str(nonce))                                
            return nonce
    if debug_level <=0:
        print("Exausted all values of nonce!")  
    return -1

def main():
    potential_nonce = -1
    nonce_array = copy_to_memory()
    btcNode_ip, btcNode_port,btcNode_user,btcNode_pass,btc_public_address,miner_id,debug_level = read_cfg()
    if debug_level <=3:
        print("Mining Starts Now!")
    while potential_nonce < 0:
        if debug_level <=2:
            startTime = time.time()        
        #Get mining template
        try:
            mining_block_info = get_block_info(btcNode_ip, btcNode_port,btcNode_user,btcNode_pass)
        except:
            continue        

        #generate coinbase transaction and get raw transaction and transaction hash (id)
        coinbase_transaction_txid, coinbase_transaction = gen_coinbase_transaction(mining_block_info['coinbasevalue'],mining_block_info['height'],btc_public_address,miner_id)             
        #add the coinbase transaction id and transactions from the mining templete to a list 
        hashList = get_tx_hashlist(coinbase_transaction_txid,mining_block_info['transactions'])
        #Calculate the merkle root 
        merkle_root = merkle(hashList)
        #get all the raw transaction data from the mining templete and coinbase transaction
        tansaction_data_list = []
        tansaction_data_list.append(coinbase_transaction)
        for i in range(len(mining_block_info['transactions'])):
            tansaction_data_list.append(mining_block_info['transactions'][i]['data'])

        #get header info from mining templete
        version,previous_hash,t,bits,block_height = mining_block_info['version'], mining_block_info['previousblockhash'],mining_block_info['curtime'], mining_block_info['bits'],mining_block_info['height']
        mining_block_data = {'version': version,'previousblockhash': previous_hash, 'curtime': t, 'bits': bits }
        
        #Finally mine for the correct nonce        
        potential_nonce = mine_numba(block_height,mining_block_data,merkle_root, nonce_array,debug_level)   
        
        if debug_level <=2:
            currTime = time.time()
            timeDiff = currTime - startTime
            print('hashes completed: ' +str(4294967295) +  " Time: "+ str(round(timeDiff)) + " , hashes rate: " + str(int(4294967295/timeDiff)))
        
    
    #set correct nonce. I like to make things very obvious :)
    correct_nonce = potential_nonce
    #if lucky generate a block and submit it to the network
    block= gen_block(version,previous_hash,merkle_root,t,bits,correct_nonce, tansaction_data_list)
    submit_work(block)

if __name__ == '__main__':    
    main()
