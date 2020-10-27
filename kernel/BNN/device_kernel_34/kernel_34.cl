#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel int32_t conv0_pad_pipe_1 __attribute__((depth(3468)));
channel int32_t conv0_pipe_2 __attribute__((depth(16384)));
channel int32_t bn1_pipe_3 __attribute__((depth(16384)));
channel int32_t bn1_pipe_115 __attribute__((depth(16384)));
channel uint16_t layer1_0_rsign1_pipe_4 __attribute__((depth(1024)));
channel uint16_t layer1_0_conv1_pad_pipe_5 __attribute__((depth(1156)));
channel int8_t layer1_0_conv1_pipe_6 __attribute__((depth(16384)));
channel int32_t layer1_0_bn1_pipe_7 __attribute__((depth(16384)));
channel int32_t layer1_0_residual1_pipe_8 __attribute__((depth(16384)));
channel int32_t layer1_0_rprelu1_pipe_9 __attribute__((depth(16384)));
channel int32_t layer1_0_rprelu1_pipe_116 __attribute__((depth(16384)));
channel uint16_t layer1_0_rsign2_pipe_10 __attribute__((depth(1024)));
channel uint16_t layer1_0_conv2_pad_pipe_11 __attribute__((depth(1156)));
channel int8_t layer1_0_conv2_pipe_12 __attribute__((depth(16384)));
channel int32_t layer1_0_bn2_pipe_13 __attribute__((depth(16384)));
channel int32_t layer1_0_residual2_pipe_14 __attribute__((depth(16384)));
channel int32_t layer1_0_rprelu2_pipe_15 __attribute__((depth(16384)));
channel int32_t layer1_0_rprelu2_pipe_117 __attribute__((depth(16384)));
channel uint16_t layer1_1_rsign1_pipe_16 __attribute__((depth(1024)));
channel uint16_t layer1_1_conv1_pad_pipe_17 __attribute__((depth(1156)));
channel int8_t layer1_1_conv1_pipe_18 __attribute__((depth(16384)));
channel int32_t layer1_1_bn1_pipe_19 __attribute__((depth(16384)));
channel int32_t layer1_1_residual1_pipe_20 __attribute__((depth(16384)));
channel int32_t layer1_1_rprelu1_pipe_21 __attribute__((depth(16384)));
channel int32_t layer1_1_rprelu1_pipe_118 __attribute__((depth(16384)));
channel uint16_t layer1_1_rsign2_pipe_22 __attribute__((depth(1024)));
channel uint16_t layer1_1_conv2_pad_pipe_23 __attribute__((depth(1156)));
channel int8_t layer1_1_conv2_pipe_24 __attribute__((depth(16384)));
channel int32_t layer1_1_bn2_pipe_25 __attribute__((depth(16384)));
channel int32_t layer1_1_residual2_pipe_26 __attribute__((depth(16384)));
channel int32_t layer1_1_rprelu2_pipe_27 __attribute__((depth(16384)));
channel int32_t layer1_1_rprelu2_pipe_119 __attribute__((depth(16384)));
channel uint16_t layer1_2_rsign1_pipe_28 __attribute__((depth(1024)));
channel uint16_t layer1_2_conv1_pad_pipe_29 __attribute__((depth(1156)));
channel int8_t layer1_2_conv1_pipe_30 __attribute__((depth(16384)));
channel int32_t layer1_2_bn1_pipe_31 __attribute__((depth(16384)));
channel int32_t layer1_2_residual1_pipe_32 __attribute__((depth(16384)));
channel int32_t layer1_2_rprelu1_pipe_33 __attribute__((depth(16384)));
channel int32_t layer1_2_rprelu1_pipe_120 __attribute__((depth(16384)));
channel uint16_t layer1_2_rsign2_pipe_34 __attribute__((depth(1024)));
channel uint16_t layer1_2_conv2_pad_pipe_35 __attribute__((depth(1156)));
channel int8_t layer1_2_conv2_pipe_36 __attribute__((depth(16384)));
channel int32_t layer1_2_bn2_pipe_37 __attribute__((depth(16384)));
channel int32_t layer1_2_residual2_pipe_38 __attribute__((depth(16384)));
channel int32_t layer1_2_rprelu2_pipe_39 __attribute__((depth(16384)));
channel int32_t layer1_2_rprelu2_pipe_121 __attribute__((depth(16384)));
channel uint16_t layer2_0_rsign1_pipe_40 __attribute__((depth(1024)));
channel uint16_t layer2_0_conv1_pad_pipe_41 __attribute__((depth(1156)));
channel int8_t layer2_0_conv1_pipe_42 __attribute__((depth(8192)));
channel int32_t layer2_0_bn1_pipe_43 __attribute__((depth(8192)));
channel int32_t layer2_0_avgpool_res_pipe_122 __attribute__((depth(4096)));
channel int32_t layer2_0_concat_pipe_123 __attribute__((depth(8192)));
channel int32_t layer2_0_residual1_pipe_44 __attribute__((depth(8192)));
channel int32_t layer2_0_rprelu1_pipe_45 __attribute__((depth(8192)));
channel int32_t layer2_0_rprelu1_pipe_124 __attribute__((depth(8192)));
channel uint32_t layer2_0_rsign2_pipe_46 __attribute__((depth(256)));
channel uint32_t layer2_0_conv2_pad_pipe_47 __attribute__((depth(324)));
channel int8_t layer2_0_conv2_pipe_48 __attribute__((depth(8192)));
channel int32_t layer2_0_bn2_pipe_49 __attribute__((depth(8192)));
channel int32_t layer2_0_residual2_pipe_50 __attribute__((depth(8192)));
channel int32_t layer2_0_rprelu2_pipe_51 __attribute__((depth(8192)));
channel int32_t layer2_0_rprelu2_pipe_125 __attribute__((depth(8192)));
channel uint32_t layer2_1_rsign1_pipe_52 __attribute__((depth(256)));
channel uint32_t layer2_1_conv1_pad_pipe_53 __attribute__((depth(324)));
channel int8_t layer2_1_conv1_pipe_54 __attribute__((depth(8192)));
channel int32_t layer2_1_bn1_pipe_55 __attribute__((depth(8192)));
channel int32_t layer2_1_residual1_pipe_56 __attribute__((depth(8192)));
channel int32_t layer2_1_rprelu1_pipe_57 __attribute__((depth(8192)));
channel int32_t layer2_1_rprelu1_pipe_126 __attribute__((depth(8192)));
channel uint32_t layer2_1_rsign2_pipe_58 __attribute__((depth(256)));
channel uint32_t layer2_1_conv2_pad_pipe_59 __attribute__((depth(324)));
channel int8_t layer2_1_conv2_pipe_60 __attribute__((depth(8192)));
channel int32_t layer2_1_bn2_pipe_61 __attribute__((depth(8192)));
channel int32_t layer2_1_residual2_pipe_62 __attribute__((depth(8192)));
channel int32_t layer2_1_rprelu2_pipe_63 __attribute__((depth(8192)));
channel int32_t layer2_1_rprelu2_pipe_127 __attribute__((depth(8192)));
channel uint32_t layer2_2_rsign1_pipe_64 __attribute__((depth(256)));
channel uint32_t layer2_2_conv1_pad_pipe_65 __attribute__((depth(324)));
channel int8_t layer2_2_conv1_pipe_66 __attribute__((depth(8192)));
channel int32_t layer2_2_bn1_pipe_67 __attribute__((depth(8192)));
channel int32_t layer2_2_residual1_pipe_68 __attribute__((depth(8192)));
channel int32_t layer2_2_rprelu1_pipe_69 __attribute__((depth(8192)));
channel int32_t layer2_2_rprelu1_pipe_128 __attribute__((depth(8192)));
channel uint32_t layer2_2_rsign2_pipe_70 __attribute__((depth(256)));
channel uint32_t layer2_2_conv2_pad_pipe_71 __attribute__((depth(324)));
channel int8_t layer2_2_conv2_pipe_72 __attribute__((depth(8192)));
channel int32_t layer2_2_bn2_pipe_73 __attribute__((depth(8192)));
channel int32_t layer2_2_residual2_pipe_74 __attribute__((depth(8192)));
channel int32_t layer2_2_rprelu2_pipe_75 __attribute__((depth(8192)));
channel int32_t layer2_2_rprelu2_pipe_129 __attribute__((depth(8192)));
channel uint32_t layer3_0_rsign1_pipe_76 __attribute__((depth(256)));
channel uint32_t layer3_0_conv1_pad_pipe_77 __attribute__((depth(324)));
channel int8_t layer3_0_conv1_pipe_78 __attribute__((depth(4096)));
channel int32_t layer3_0_bn1_pipe_79 __attribute__((depth(4096)));
channel int32_t layer3_0_avgpool_res_pipe_130 __attribute__((depth(2048)));
channel int32_t layer3_0_concat_pipe_131 __attribute__((depth(4096)));
channel int32_t layer3_0_residual1_pipe_80 __attribute__((depth(4096)));
channel int32_t layer3_0_rprelu1_pipe_81 __attribute__((depth(4096)));
channel int32_t layer3_0_rprelu1_pipe_132 __attribute__((depth(4096)));
channel uint32_t layer3_0_rsign2_pipe_82 __attribute__((depth(128)));
channel uint32_t layer3_0_conv2_pad_pipe_83 __attribute__((depth(200)));
channel int8_t layer3_0_conv2_pipe_84 __attribute__((depth(4096)));
channel int32_t layer3_0_bn2_pipe_85 __attribute__((depth(4096)));
channel int32_t layer3_0_residual2_pipe_86 __attribute__((depth(4096)));
channel int32_t layer3_0_rprelu2_pipe_87 __attribute__((depth(4096)));
channel int32_t layer3_0_rprelu2_pipe_133 __attribute__((depth(4096)));
channel uint32_t layer3_1_rsign1_pipe_88 __attribute__((depth(128)));
channel uint32_t layer3_1_conv1_pad_pipe_89 __attribute__((depth(200)));
channel int8_t layer3_1_conv1_pipe_90 __attribute__((depth(4096)));
channel int32_t layer3_1_bn1_pipe_91 __attribute__((depth(4096)));
channel int32_t layer3_1_residual1_pipe_92 __attribute__((depth(4096)));
channel int32_t layer3_1_rprelu1_pipe_93 __attribute__((depth(4096)));
channel int32_t layer3_1_rprelu1_pipe_134 __attribute__((depth(4096)));
channel uint32_t layer3_1_rsign2_pipe_94 __attribute__((depth(128)));
channel uint32_t layer3_1_conv2_pad_pipe_95 __attribute__((depth(200)));
channel int8_t layer3_1_conv2_pipe_96 __attribute__((depth(4096)));
channel int32_t layer3_1_bn2_pipe_97 __attribute__((depth(4096)));
channel int32_t layer3_1_residual2_pipe_98 __attribute__((depth(4096)));
channel int32_t layer3_1_rprelu2_pipe_99 __attribute__((depth(4096)));
channel int32_t layer3_1_rprelu2_pipe_135 __attribute__((depth(4096)));
channel uint32_t layer3_2_rsign1_pipe_100 __attribute__((depth(128)));
channel uint32_t layer3_2_conv1_pad_pipe_101 __attribute__((depth(200)));
channel int8_t layer3_2_conv1_pipe_102 __attribute__((depth(4096)));
channel int32_t layer3_2_bn1_pipe_103 __attribute__((depth(4096)));
channel int32_t layer3_2_residual1_pipe_104 __attribute__((depth(4096)));
channel int32_t layer3_2_rprelu1_pipe_105 __attribute__((depth(4096)));
channel int32_t layer3_2_rprelu1_pipe_136 __attribute__((depth(4096)));
channel uint32_t layer3_2_rsign2_pipe_106 __attribute__((depth(128)));
channel uint32_t layer3_2_conv2_pad_pipe_107 __attribute__((depth(200)));
channel int8_t layer3_2_conv2_pipe_108 __attribute__((depth(4096)));
channel int32_t layer3_2_bn2_pipe_109 __attribute__((depth(4096)));
channel int32_t layer3_2_residual2_pipe_110 __attribute__((depth(4096)));
channel int32_t layer3_2_rprelu2_pipe_111 __attribute__((depth(4096)));
channel int32_t avgpool_res_pipe_112 __attribute__((depth(64)));
channel int32_t flatten_pipe_113 __attribute__((depth(64)));
channel int32_t fc_matmul_pipe_114 __attribute__((depth(10)));
__kernel void test(__global int32_t* restrict input_image, __global int32_t* restrict fc) {
    int32_t _top;
    int32_t w_conv1[432];
    int32_t w_bn1_0[16];
    int32_t w_bn1_1[16];
    int32_t w_fc_167[640];
    int32_t w_fc_168[10];
    int32_t w_layer1_0_rprelu1_0[16];
    int32_t w_layer1_0_rprelu1_1[16];
    int32_t w_layer1_0_rprelu1_2[16];
    int32_t w_layer1_0_rprelu2_3[16];
    int32_t w_layer1_0_rprelu2_4[16];
    int32_t w_layer1_0_rprelu2_5[16];
    int32_t w_layer1_0_rsign1[16];
    int32_t w_layer1_0_rsign2[16];
    uint16_t w_layer1_0_conv1[144];
    int32_t w_layer1_0_bn1_9[16];
    int32_t w_layer1_0_bn1_10[16];
    uint16_t w_layer1_0_conv2[144];
    int32_t w_layer1_0_bn2_14[16];
    int32_t w_layer1_0_bn2_15[16];
    int32_t w_layer1_1_rprelu1_0[16];
    int32_t w_layer1_1_rprelu1_1[16];
    int32_t w_layer1_1_rprelu1_2[16];
    int32_t w_layer1_1_rprelu2_3[16];
    int32_t w_layer1_1_rprelu2_4[16];
    int32_t w_layer1_1_rprelu2_5[16];
    int32_t w_layer1_1_rsign1[16];
    int32_t w_layer1_1_rsign2[16];
    uint16_t w_layer1_1_conv1[144];
    int32_t w_layer1_1_bn1_9[16];
    int32_t w_layer1_1_bn1_10[16];
    uint16_t w_layer1_1_conv2[144];
    int32_t w_layer1_1_bn2_14[16];
    int32_t w_layer1_1_bn2_15[16];
    int32_t w_layer1_2_rprelu1_0[16];
    int32_t w_layer1_2_rprelu1_1[16];
    int32_t w_layer1_2_rprelu1_2[16];
    int32_t w_layer1_2_rprelu2_3[16];
    int32_t w_layer1_2_rprelu2_4[16];
    int32_t w_layer1_2_rprelu2_5[16];
    int32_t w_layer1_2_rsign1[16];
    int32_t w_layer1_2_rsign2[16];
    uint16_t w_layer1_2_conv1[144];
    int32_t w_layer1_2_bn1_9[16];
    int32_t w_layer1_2_bn1_10[16];
    uint16_t w_layer1_2_conv2[144];
    int32_t w_layer1_2_bn2_14[16];
    int32_t w_layer1_2_bn2_15[16];
    int32_t w_layer2_0_rprelu1_0[32];
    int32_t w_layer2_0_rprelu1_1[32];
    int32_t w_layer2_0_rprelu1_2[32];
    int32_t w_layer2_0_rprelu2_3[32];
    int32_t w_layer2_0_rprelu2_4[32];
    int32_t w_layer2_0_rprelu2_5[32];
    int32_t w_layer2_0_rsign1[16];
    int32_t w_layer2_0_rsign2[32];
    uint32_t w_layer2_0_conv1[32][1][3][3];
    int32_t w_layer2_0_bn1_9[32];
    int32_t w_layer2_0_bn1_10[32];
    uint32_t w_layer2_0_conv2[288];
    int32_t w_layer2_0_bn2_14[32];
    int32_t w_layer2_0_bn2_15[32];
    int32_t w_layer2_1_rprelu1_0[32];
    int32_t w_layer2_1_rprelu1_1[32];
    int32_t w_layer2_1_rprelu1_2[32];
    int32_t w_layer2_1_rprelu2_3[32];
    int32_t w_layer2_1_rprelu2_4[32];
    int32_t w_layer2_1_rprelu2_5[32];
    int32_t w_layer2_1_rsign1[32];
    int32_t w_layer2_1_rsign2[32];
    uint32_t w_layer2_1_conv1[288];
    int32_t w_layer2_1_bn1_9[32];
    int32_t w_layer2_1_bn1_10[32];
    uint32_t w_layer2_1_conv2[288];
    int32_t w_layer2_1_bn2_14[32];
    int32_t w_layer2_1_bn2_15[32];
    int32_t w_layer2_2_rprelu1_0[32];
    int32_t w_layer2_2_rprelu1_1[32];
    int32_t w_layer2_2_rprelu1_2[32];
    int32_t w_layer2_2_rprelu2_3[32];
    int32_t w_layer2_2_rprelu2_4[32];
    int32_t w_layer2_2_rprelu2_5[32];
    int32_t w_layer2_2_rsign1[32];
    int32_t w_layer2_2_rsign2[32];
    uint32_t w_layer2_2_conv1[288];
    int32_t w_layer2_2_bn1_9[32];
    int32_t w_layer2_2_bn1_10[32];
    uint32_t w_layer2_2_conv2[288];
    int32_t w_layer2_2_bn2_14[32];
    int32_t w_layer2_2_bn2_15[32];
    int32_t w_layer3_0_rprelu1_0[64];
    int32_t w_layer3_0_rprelu1_1[64];
    int32_t w_layer3_0_rprelu1_2[64];
    int32_t w_layer3_0_rprelu2_3[64];
    int32_t w_layer3_0_rprelu2_4[64];
    int32_t w_layer3_0_rprelu2_5[64];
    int32_t w_layer3_0_rsign1[32];
    int32_t w_layer3_0_rsign2[64];
    uint32_t w_layer3_0_conv1[64][1][3][3];
    int32_t w_layer3_0_bn1_9[64];
    int32_t w_layer3_0_bn1_10[64];
    uint32_t w_layer3_0_conv2[1152];
    int32_t w_layer3_0_bn2_14[64];
    int32_t w_layer3_0_bn2_15[64];
    int32_t w_layer3_1_rprelu1_0[64];
    int32_t w_layer3_1_rprelu1_1[64];
    int32_t w_layer3_1_rprelu1_2[64];
    int32_t w_layer3_1_rprelu2_3[64];
    int32_t w_layer3_1_rprelu2_4[64];
    int32_t w_layer3_1_rprelu2_5[64];
    int32_t w_layer3_1_rsign1[64];
    int32_t w_layer3_1_rsign2[64];
    uint32_t w_layer3_1_conv1[1152];
    int32_t w_layer3_1_bn1_9[64];
    int32_t w_layer3_1_bn1_10[64];
    uint32_t w_layer3_1_conv2[1152];
    int32_t w_layer3_1_bn2_14[64];
    int32_t w_layer3_1_bn2_15[64];
    int32_t w_layer3_2_rprelu1_0[64];
    int32_t w_layer3_2_rprelu1_1[64];
    int32_t w_layer3_2_rprelu1_2[64];
    int32_t w_layer3_2_rprelu2_3[64];
    int32_t w_layer3_2_rprelu2_4[64];
    int32_t w_layer3_2_rprelu2_5[64];
    int32_t w_layer3_2_rsign1[64];
    int32_t w_layer3_2_rsign2[64];
    uint32_t w_layer3_2_conv1[1152];
    int32_t w_layer3_2_bn1_9[64];
    int32_t w_layer3_2_bn1_10[64];
    uint32_t w_layer3_2_conv2[1152];
    int32_t w_layer3_2_bn2_14[64];
    int32_t w_layer3_2_bn2_15[64];
    int32_t conv0_pad[3468];
        for (int32_t indices = 0; indices < 1; ++indices) {
      for (int32_t not_zero = 0; not_zero < 3; ++not_zero) {
        for (int32_t index_tuple = 0; index_tuple < 34; ++index_tuple) {
          for (int32_t i = 0; i < 34; ++i) {
            int32_t conv0_pad_temp;
            conv0_pad_temp = (int32_t)(((((1 <= index_tuple) && (index_tuple < 33)) && (1 <= i)) && (i < 33)) ? ((int32_t)input_image[((((i + (index_tuple * 32)) + (not_zero * 1024)) + (indices * 3072)) + -33)]) : ((int32_t)0));
            write_channel_intel(conv0_pad_pipe_1, conv0_pad_temp);
            conv0_pad[(((i + (index_tuple * 34)) + (not_zero * 1156)) + (indices * 3468))] = conv0_pad_temp;
          }
        }
      }
    }
    int32_t conv0[16384];
    int32_t conv0_LB[306];
    int32_t conv0_WB[27];
        for (int32_t nn = 0; nn < 1; ++nn) {
      for (int32_t yy_reuse = 0; yy_reuse < 34; ++yy_reuse) {
        for (int32_t xx_reuse = 0; xx_reuse < 34; ++xx_reuse) {
          for (int32_t conv0_pad_2 = 0; conv0_pad_2 < 3; ++conv0_pad_2) {
            for (int32_t conv0_pad_1 = 0; conv0_pad_1 < 2; ++conv0_pad_1) {
              conv0_LB[((xx_reuse + (conv0_pad_1 * 34)) + (conv0_pad_2 * 102))] = conv0_LB[(((xx_reuse + (conv0_pad_1 * 34)) + (conv0_pad_2 * 102)) + 34)];
            }
            int32_t conv0_pad_temp1;
            conv0_pad_temp1 = read_channel_intel(conv0_pad_pipe_1);
            conv0_LB[((xx_reuse + (conv0_pad_2 * 102)) + 68)] = conv0_pad_temp1;
          }
          if (2 <= yy_reuse) {
            for (int32_t conv0_LB_1 = 0; conv0_LB_1 < 3; ++conv0_LB_1) {
              for (int32_t conv0_LB_2 = 0; conv0_LB_2 < 3; ++conv0_LB_2) {
                for (int32_t conv0_LB_0 = 0; conv0_LB_0 < 2; ++conv0_LB_0) {
                  conv0_WB[((conv0_LB_0 + (conv0_LB_1 * 3)) + (conv0_LB_2 * 9))] = conv0_WB[(((conv0_LB_0 + (conv0_LB_1 * 3)) + (conv0_LB_2 * 9)) + 1)];
                }
                conv0_WB[(((conv0_LB_1 + (conv0_LB_2 * 3)) * 3) + 2)] = conv0_LB[((xx_reuse + (conv0_LB_1 * 34)) + (conv0_LB_2 * 102))];
              }
            }
            for (int32_t ff = 0; ff < 16; ++ff) {
              if (2 <= xx_reuse) {
                int32_t sum;
                for (int32_t rc = 0; rc < 3; ++rc) {
                  for (int32_t ry = 0; ry < 3; ++ry) {
                    for (int32_t rx = 0; rx < 3; ++rx) {
                      sum = ((int32_t)(((int64_t)(((int64_t)conv0_WB[((rx + (ry * 3)) + (rc * 9))]) * ((int64_t)w_conv1[(((rx + (ry * 3)) + (rc * 9)) + (ff * 27))]))) + ((int64_t)sum)));
                    }
                  }
                }
                int32_t conv0_temp;
                conv0_temp = sum;
                write_channel_intel(conv0_pipe_2, conv0_temp);
              }
            }
          }
        }
      }
    }
    int32_t bn1[16384];
    for (int32_t n = 0; n < 1; ++n) {
      for (int32_t c = 0; c < 16; ++c) {
        for (int32_t h = 0; h < 32; ++h) {
          for (int32_t w = 0; w < 32; ++w) {
            int32_t conv0_temp1;
            printf("5");
            conv0_temp1 = read_channel_intel(conv0_pipe_2);
            int32_t bn1_temp;
            bn1_temp = ((int32_t)(((int64_t)(((int64_t)conv0_temp1) * ((int64_t)w_bn1_0[c]))) + ((int64_t)w_bn1_1[c])));
            write_channel_intel(bn1_pipe_115, bn1_temp);
            write_channel_intel(bn1_pipe_3, bn1_temp);
          }
        }
      }
    }
    uint16_t layer1_0_rsign1[1024];
        for (int32_t nn1 = 0; nn1 < 1; ++nn1) {
      for (int32_t cc = 0; cc < 1; ++cc) {
        for (int32_t hh = 0; hh < 32; ++hh) {
          for (int32_t ww = 0; ww < 32; ++ww) {
            uint16_t layer1_0_rsign1_pack;
            for (int32_t i1 = 0; i1 < 16; ++i1) {
              int32_t bn1_temp1;
              bn1_temp1 = read_channel_intel(bn1_pipe_3);
              layer1_0_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)bn1_temp1) + ((int33_t)w_layer1_0_rsign1[((cc * 16) + i1)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint16_t layer1_0_rsign1_temp;
            layer1_0_rsign1_temp = layer1_0_rsign1_pack;
            write_channel_intel(layer1_0_rsign1_pipe_4, layer1_0_rsign1_temp);
          }
        }
      }
    }
    uint16_t layer1_0_conv1_pad[1156];
        for (int32_t ii = 0; ii < 1; ++ii) {
      for (int32_t cc1 = 0; cc1 < 1; ++cc1) {
        for (int32_t hh1 = 0; hh1 < 34; ++hh1) {
          for (int32_t ww1 = 0; ww1 < 34; ++ww1) {
            uint16_t layer1_0_rsign1_temp1 = 0;
            bool cond = ((((1 <= ww1) && (ww1 < 33)) && (1 <= hh1)) && (hh1 < 33));
            if (cond) {
              layer1_0_rsign1_temp1 = read_channel_intel(layer1_0_rsign1_pipe_4); 
            }
            uint16_t layer1_0_conv1_pad_temp;
            layer1_0_conv1_pad_temp = ((uint16_t)(uint32_t)(cond ? (((uint32_t)layer1_0_rsign1_temp1)) : ((uint32_t)0U)));
            write_channel_intel(layer1_0_conv1_pad_pipe_5, layer1_0_conv1_pad_temp);
            layer1_0_conv1_pad[(((ww1 + (hh1 * 34)) + (cc1 * 1156)) + (ii * 1156))] = layer1_0_conv1_pad_temp;
          }
        }
      }
    }
    int8_t layer1_0_conv1[16384];
    uint16_t layer1_0_conv1_LB[102];
    uint16_t layer1_0_conv1_WB[9];
        for (int32_t nn2 = 0; nn2 < 1; ++nn2) {
      for (int32_t yy_reuse1 = 0; yy_reuse1 < 34; ++yy_reuse1) {
        for (int32_t xx_reuse1 = 0; xx_reuse1 < 34; ++xx_reuse1) {
          for (int32_t layer1_0_conv1_pad_1 = 0; layer1_0_conv1_pad_1 < 2; ++layer1_0_conv1_pad_1) {
            layer1_0_conv1_LB[(xx_reuse1 + (layer1_0_conv1_pad_1 * 34))] = layer1_0_conv1_LB[((xx_reuse1 + (layer1_0_conv1_pad_1 * 34)) + 34)];
          }
          uint16_t layer1_0_conv1_pad_temp1;
          layer1_0_conv1_pad_temp1 = read_channel_intel(layer1_0_conv1_pad_pipe_5);
          layer1_0_conv1_LB[(xx_reuse1 + 68)] = layer1_0_conv1_pad_temp1;
          if (2 <= yy_reuse1) {
            for (int32_t layer1_0_conv1_LB_1 = 0; layer1_0_conv1_LB_1 < 3; ++layer1_0_conv1_LB_1) {
              for (int32_t layer1_0_conv1_LB_0 = 0; layer1_0_conv1_LB_0 < 2; ++layer1_0_conv1_LB_0) {
                layer1_0_conv1_WB[(layer1_0_conv1_LB_0 + (layer1_0_conv1_LB_1 * 3))] = layer1_0_conv1_WB[((layer1_0_conv1_LB_0 + (layer1_0_conv1_LB_1 * 3)) + 1)];
              }
              layer1_0_conv1_WB[((layer1_0_conv1_LB_1 * 3) + 2)] = layer1_0_conv1_LB[(xx_reuse1 + (layer1_0_conv1_LB_1 * 34))];
            }
            for (int32_t ff1 = 0; ff1 < 16; ++ff1) {
              if (2 <= xx_reuse1) {
                int8_t layer1_0_conv1_sum;
                for (int32_t layer1_0_conv1_rc = 0; layer1_0_conv1_rc < 1; ++layer1_0_conv1_rc) {
                  for (int32_t layer1_0_conv1_ry = 0; layer1_0_conv1_ry < 3; ++layer1_0_conv1_ry) {
                    for (int32_t layer1_0_conv1_rx = 0; layer1_0_conv1_rx < 3; ++layer1_0_conv1_rx) {
                      for (int32_t layer1_0_conv1_rb = 0; layer1_0_conv1_rb < 16; ++layer1_0_conv1_rb) {
                        layer1_0_conv1_sum = ((int8_t)(((int32_t)(((layer1_0_conv1_WB[(layer1_0_conv1_rx + (layer1_0_conv1_ry * 3))] ^ w_layer1_0_conv1[((layer1_0_conv1_rx + (layer1_0_conv1_ry * 3)) + (ff1 * 9))]) & (1L << layer1_0_conv1_rb)) >> layer1_0_conv1_rb)) + ((int32_t)layer1_0_conv1_sum)));
                      }
                    }
                  }
                }
                int8_t layer1_0_conv1_temp;
                layer1_0_conv1_temp = ((int8_t)(144 - ((int32_t)(layer1_0_conv1_sum << 1))));
                write_channel_intel(layer1_0_conv1_pipe_6, layer1_0_conv1_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer1_0_bn1[16384];
        for (int32_t n1 = 0; n1 < 1; ++n1) {
      for (int32_t c1 = 0; c1 < 16; ++c1) {
        for (int32_t h1 = 0; h1 < 32; ++h1) {
          for (int32_t w1 = 0; w1 < 32; ++w1) {
            int8_t layer1_0_conv1_temp1;
            layer1_0_conv1_temp1 = read_channel_intel(layer1_0_conv1_pipe_6);printf("2");
            int32_t layer1_0_bn1_temp;
            layer1_0_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer1_0_conv1_temp1) * ((int40_t)w_layer1_0_bn1_9[c1]))) + ((int41_t)w_layer1_0_bn1_10[c1])));
            write_channel_intel(layer1_0_bn1_pipe_7, layer1_0_bn1_temp);
          }
        }
      }
    }
    int32_t layer1_0_residual1[16384];
        for (int32_t nn3 = 0; nn3 < 1; ++nn3) {
      for (int32_t cc2 = 0; cc2 < 16; ++cc2) {
        for (int32_t ww2 = 0; ww2 < 32; ++ww2) {
          for (int32_t hh2 = 0; hh2 < 32; ++hh2) {
            int32_t bn1_temp2;
            bn1_temp2 = read_channel_intel(bn1_pipe_115);
            int32_t layer1_0_bn1_temp1;
            layer1_0_bn1_temp1 = read_channel_intel(layer1_0_bn1_pipe_7);
            int32_t layer1_0_residual1_temp;
            layer1_0_residual1_temp = ((int32_t)(((int33_t)layer1_0_bn1_temp1) + ((int33_t)bn1_temp2)));
            write_channel_intel(layer1_0_residual1_pipe_8, layer1_0_residual1_temp);
          }
        }
      }
    }
    int32_t layer1_0_rprelu1[16384];
            for (int32_t nn4 = 0; nn4 < 1; ++nn4) {
      for (int32_t cc3 = 0; cc3 < 16; ++cc3) {
        for (int32_t ww3 = 0; ww3 < 32; ++ww3) {
          for (int32_t hh3 = 0; hh3 < 32; ++hh3) {
            int32_t layer1_0_residual1_temp1;
            layer1_0_residual1_temp1 = read_channel_intel(layer1_0_residual1_pipe_8);
            int32_t layer1_0_rprelu1_temp;
            layer1_0_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_0_residual1_temp1) + ((int33_t)w_layer1_0_rprelu1_0[cc3])))) ? (((int64_t)(((int33_t)layer1_0_residual1_temp1) + ((int33_t)w_layer1_0_rprelu1_0[cc3])))) : ((int64_t)(((int64_t)w_layer1_0_rprelu1_2[cc3]) * ((int64_t)(((int33_t)layer1_0_residual1_temp1) + ((int33_t)w_layer1_0_rprelu1_0[cc3]))))))) + ((int64_t)w_layer1_0_rprelu1_1[cc3])));
            write_channel_intel(layer1_0_rprelu1_pipe_116, layer1_0_rprelu1_temp);
            write_channel_intel(layer1_0_rprelu1_pipe_9, layer1_0_rprelu1_temp);
          }
        }
      }
    }
    uint16_t layer1_0_rsign2[1024];
        for (int32_t nn5 = 0; nn5 < 1; ++nn5) {
      for (int32_t cc4 = 0; cc4 < 1; ++cc4) {
        for (int32_t hh4 = 0; hh4 < 32; ++hh4) {
          for (int32_t ww4 = 0; ww4 < 32; ++ww4) {
            uint16_t layer1_0_rsign2_pack;
            for (int32_t i2 = 0; i2 < 16; ++i2) {
              int32_t layer1_0_rprelu1_temp1;
              layer1_0_rprelu1_temp1 = read_channel_intel(layer1_0_rprelu1_pipe_9);
              layer1_0_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_0_rprelu1_temp1) + ((int33_t)w_layer1_0_rsign2[((cc4 * 16) + i2)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint16_t layer1_0_rsign2_temp;
            layer1_0_rsign2_temp = layer1_0_rsign2_pack;
            write_channel_intel(layer1_0_rsign2_pipe_10, layer1_0_rsign2_temp);
          }
        }
      }
    }
    uint16_t layer1_0_conv2_pad[1156];
        for (int32_t ii1 = 0; ii1 < 1; ++ii1) {
      for (int32_t cc5 = 0; cc5 < 1; ++cc5) {
        for (int32_t hh5 = 0; hh5 < 34; ++hh5) {
          for (int32_t ww5 = 0; ww5 < 34; ++ww5) {
            bool cond = ((((1 <= ww5) && (ww5 < 33)) && (1 <= hh5)) && (hh5 < 33));
            uint16_t layer1_0_rsign2_temp1 = 0;
            if (cond) {
              layer1_0_rsign2_temp1 = read_channel_intel(layer1_0_rsign2_pipe_10);
            }
            uint16_t layer1_0_conv2_pad_temp;
            layer1_0_conv2_pad_temp = ((uint16_t)(uint32_t)(cond ? (((uint32_t)layer1_0_rsign2_temp1)) : ((uint32_t)0U)));
            write_channel_intel(layer1_0_conv2_pad_pipe_11, layer1_0_conv2_pad_temp);
            layer1_0_conv2_pad[(((ww5 + (hh5 * 34)) + (cc5 * 1156)) + (ii1 * 1156))] = layer1_0_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer1_0_conv2[16384];
    uint16_t layer1_0_conv2_LB[102];
    uint16_t layer1_0_conv2_WB[9];
        for (int32_t nn6 = 0; nn6 < 1; ++nn6) {
      for (int32_t yy_reuse2 = 0; yy_reuse2 < 34; ++yy_reuse2) {
        for (int32_t xx_reuse2 = 0; xx_reuse2 < 34; ++xx_reuse2) {
          for (int32_t layer1_0_conv2_pad_1 = 0; layer1_0_conv2_pad_1 < 2; ++layer1_0_conv2_pad_1) {
            layer1_0_conv2_LB[(xx_reuse2 + (layer1_0_conv2_pad_1 * 34))] = layer1_0_conv2_LB[((xx_reuse2 + (layer1_0_conv2_pad_1 * 34)) + 34)];
          }
          uint16_t layer1_0_conv2_pad_temp1;
          layer1_0_conv2_pad_temp1 = read_channel_intel(layer1_0_conv2_pad_pipe_11);
          layer1_0_conv2_LB[(xx_reuse2 + 68)] = layer1_0_conv2_pad_temp1;
          if (2 <= yy_reuse2) {
            for (int32_t layer1_0_conv2_LB_1 = 0; layer1_0_conv2_LB_1 < 3; ++layer1_0_conv2_LB_1) {
              for (int32_t layer1_0_conv2_LB_0 = 0; layer1_0_conv2_LB_0 < 2; ++layer1_0_conv2_LB_0) {
                layer1_0_conv2_WB[(layer1_0_conv2_LB_0 + (layer1_0_conv2_LB_1 * 3))] = layer1_0_conv2_WB[((layer1_0_conv2_LB_0 + (layer1_0_conv2_LB_1 * 3)) + 1)];
              }
              layer1_0_conv2_WB[((layer1_0_conv2_LB_1 * 3) + 2)] = layer1_0_conv2_LB[(xx_reuse2 + (layer1_0_conv2_LB_1 * 34))];
            }
            for (int32_t ff2 = 0; ff2 < 16; ++ff2) {
              if (2 <= xx_reuse2) {
                int8_t layer1_0_conv2_sum;
                for (int32_t layer1_0_conv2_rc = 0; layer1_0_conv2_rc < 1; ++layer1_0_conv2_rc) {
                  for (int32_t layer1_0_conv2_ry = 0; layer1_0_conv2_ry < 3; ++layer1_0_conv2_ry) {
                    for (int32_t layer1_0_conv2_rx = 0; layer1_0_conv2_rx < 3; ++layer1_0_conv2_rx) {
                      for (int32_t layer1_0_conv2_rb = 0; layer1_0_conv2_rb < 16; ++layer1_0_conv2_rb) {
                        layer1_0_conv2_sum = ((int8_t)(((int32_t)(((layer1_0_conv2_WB[(layer1_0_conv2_rx + (layer1_0_conv2_ry * 3))] ^ w_layer1_0_conv2[((layer1_0_conv2_rx + (layer1_0_conv2_ry * 3)) + (ff2 * 9))]) & (1L << layer1_0_conv2_rb)) >> layer1_0_conv2_rb)) + ((int32_t)layer1_0_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer1_0_conv2_temp;
                layer1_0_conv2_temp = ((int8_t)(144 - ((int32_t)(layer1_0_conv2_sum << 1))));
                write_channel_intel(layer1_0_conv2_pipe_12, layer1_0_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer1_0_bn2[16384];
        for (int32_t n2 = 0; n2 < 1; ++n2) {
      for (int32_t c2 = 0; c2 < 16; ++c2) {
        for (int32_t h2 = 0; h2 < 32; ++h2) {
          for (int32_t w2 = 0; w2 < 32; ++w2) {
            int8_t layer1_0_conv2_temp1;
            layer1_0_conv2_temp1 = read_channel_intel(layer1_0_conv2_pipe_12);
            int32_t layer1_0_bn2_temp;
            layer1_0_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer1_0_conv2_temp1) * ((int40_t)w_layer1_0_bn2_14[c2]))) + ((int41_t)w_layer1_0_bn2_15[c2])));
            write_channel_intel(layer1_0_bn2_pipe_13, layer1_0_bn2_temp);
          }
        }
      }
    }
    int32_t layer1_0_residual2[16384];
        for (int32_t nn7 = 0; nn7 < 1; ++nn7) {
      for (int32_t cc6 = 0; cc6 < 16; ++cc6) {
        for (int32_t ww6 = 0; ww6 < 32; ++ww6) {
          for (int32_t hh6 = 0; hh6 < 32; ++hh6) {
            int32_t layer1_0_bn2_temp1;
            layer1_0_bn2_temp1 = read_channel_intel(layer1_0_bn2_pipe_13);
            int32_t layer1_0_rprelu1_temp2;
            layer1_0_rprelu1_temp2 = read_channel_intel(layer1_0_rprelu1_pipe_116);
            int32_t layer1_0_residual2_temp;
            layer1_0_residual2_temp = ((int32_t)(((int33_t)layer1_0_bn2_temp1) + ((int33_t)layer1_0_rprelu1_temp2)));
            write_channel_intel(layer1_0_residual2_pipe_14, layer1_0_residual2_temp);
          }
        }
      }
    }
    int32_t layer1_0_rprelu2[16384];
            for (int32_t nn8 = 0; nn8 < 1; ++nn8) {
      for (int32_t cc7 = 0; cc7 < 16; ++cc7) {
        for (int32_t ww7 = 0; ww7 < 32; ++ww7) {
          for (int32_t hh7 = 0; hh7 < 32; ++hh7) {
            int32_t layer1_0_residual2_temp1;
            layer1_0_residual2_temp1 = read_channel_intel(layer1_0_residual2_pipe_14);
            int32_t layer1_0_rprelu2_temp;
            layer1_0_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_0_residual2_temp1) + ((int33_t)w_layer1_0_rprelu2_3[cc7])))) ? (((int64_t)(((int33_t)layer1_0_residual2_temp1) + ((int33_t)w_layer1_0_rprelu2_3[cc7])))) : ((int64_t)(((int64_t)w_layer1_0_rprelu2_5[cc7]) * ((int64_t)(((int33_t)layer1_0_residual2_temp1) + ((int33_t)w_layer1_0_rprelu2_3[cc7]))))))) + ((int64_t)w_layer1_0_rprelu2_4[cc7])));
            write_channel_intel(layer1_0_rprelu2_pipe_117, layer1_0_rprelu2_temp);
            write_channel_intel(layer1_0_rprelu2_pipe_15, layer1_0_rprelu2_temp);
          }
        }
      }
    }
    uint16_t layer1_1_rsign1[1024];
        for (int32_t nn9 = 0; nn9 < 1; ++nn9) {
      for (int32_t cc8 = 0; cc8 < 1; ++cc8) {
        for (int32_t hh8 = 0; hh8 < 32; ++hh8) {
          for (int32_t ww8 = 0; ww8 < 32; ++ww8) {
            uint16_t layer1_1_rsign1_pack;
            for (int32_t i3 = 0; i3 < 16; ++i3) {
              int32_t layer1_0_rprelu2_temp1;
              layer1_0_rprelu2_temp1 = read_channel_intel(layer1_0_rprelu2_pipe_15);
              layer1_1_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_0_rprelu2_temp1) + ((int33_t)w_layer1_1_rsign1[((cc8 * 16) + i3)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint16_t layer1_1_rsign1_temp;
            layer1_1_rsign1_temp = layer1_1_rsign1_pack;
            write_channel_intel(layer1_1_rsign1_pipe_16, layer1_1_rsign1_temp);
          }
        }
      }
    }
    uint16_t layer1_1_conv1_pad[1156];
        for (int32_t ii2 = 0; ii2 < 1; ++ii2) {
      for (int32_t cc9 = 0; cc9 < 1; ++cc9) {
        for (int32_t hh9 = 0; hh9 < 34; ++hh9) {
          for (int32_t ww9 = 0; ww9 < 34; ++ww9) {
            bool cond = ((((1 <= ww9) && (ww9 < 33)) && (1 <= hh9)) && (hh9 < 33));
            uint16_t layer1_1_rsign1_temp1 = 0;
            if (cond) {
              layer1_1_rsign1_temp1 = read_channel_intel(layer1_1_rsign1_pipe_16);
            }
            uint16_t layer1_1_conv1_pad_temp;
            layer1_1_conv1_pad_temp = ((uint16_t)(uint32_t)(cond ? (((uint32_t)layer1_1_rsign1_temp1)) : ((uint32_t)0U)));
            write_channel_intel(layer1_1_conv1_pad_pipe_17, layer1_1_conv1_pad_temp);
            layer1_1_conv1_pad[(((ww9 + (hh9 * 34)) + (cc9 * 1156)) + (ii2 * 1156))] = layer1_1_conv1_pad_temp;
          }
        }
      }
    }
    int8_t layer1_1_conv1[16384];
    uint16_t layer1_1_conv1_LB[102];
    uint16_t layer1_1_conv1_WB[9];
        for (int32_t nn10 = 0; nn10 < 1; ++nn10) {
      for (int32_t yy_reuse3 = 0; yy_reuse3 < 34; ++yy_reuse3) {
        for (int32_t xx_reuse3 = 0; xx_reuse3 < 34; ++xx_reuse3) {
          for (int32_t layer1_1_conv1_pad_1 = 0; layer1_1_conv1_pad_1 < 2; ++layer1_1_conv1_pad_1) {
            layer1_1_conv1_LB[(xx_reuse3 + (layer1_1_conv1_pad_1 * 34))] = layer1_1_conv1_LB[((xx_reuse3 + (layer1_1_conv1_pad_1 * 34)) + 34)];
          }
          uint16_t layer1_1_conv1_pad_temp1;
          layer1_1_conv1_pad_temp1 = read_channel_intel(layer1_1_conv1_pad_pipe_17);
          layer1_1_conv1_LB[(xx_reuse3 + 68)] = layer1_1_conv1_pad_temp1;
          if (2 <= yy_reuse3) {
            for (int32_t layer1_1_conv1_LB_1 = 0; layer1_1_conv1_LB_1 < 3; ++layer1_1_conv1_LB_1) {
              for (int32_t layer1_1_conv1_LB_0 = 0; layer1_1_conv1_LB_0 < 2; ++layer1_1_conv1_LB_0) {
                layer1_1_conv1_WB[(layer1_1_conv1_LB_0 + (layer1_1_conv1_LB_1 * 3))] = layer1_1_conv1_WB[((layer1_1_conv1_LB_0 + (layer1_1_conv1_LB_1 * 3)) + 1)];
              }
              layer1_1_conv1_WB[((layer1_1_conv1_LB_1 * 3) + 2)] = layer1_1_conv1_LB[(xx_reuse3 + (layer1_1_conv1_LB_1 * 34))];
            }
            for (int32_t ff3 = 0; ff3 < 16; ++ff3) {
              if (2 <= xx_reuse3) {
                int8_t layer1_1_conv1_sum;
                for (int32_t layer1_1_conv1_rc = 0; layer1_1_conv1_rc < 1; ++layer1_1_conv1_rc) {
                  for (int32_t layer1_1_conv1_ry = 0; layer1_1_conv1_ry < 3; ++layer1_1_conv1_ry) {
                    for (int32_t layer1_1_conv1_rx = 0; layer1_1_conv1_rx < 3; ++layer1_1_conv1_rx) {
                      for (int32_t layer1_1_conv1_rb = 0; layer1_1_conv1_rb < 16; ++layer1_1_conv1_rb) {
                        layer1_1_conv1_sum = ((int8_t)(((int32_t)(((layer1_1_conv1_WB[(layer1_1_conv1_rx + (layer1_1_conv1_ry * 3))] ^ w_layer1_1_conv1[((layer1_1_conv1_rx + (layer1_1_conv1_ry * 3)) + (ff3 * 9))]) & (1L << layer1_1_conv1_rb)) >> layer1_1_conv1_rb)) + ((int32_t)layer1_1_conv1_sum)));
                      }
                    }
                  }
                }
                int8_t layer1_1_conv1_temp;
                layer1_1_conv1_temp = ((int8_t)(144 - ((int32_t)(layer1_1_conv1_sum << 1))));
                write_channel_intel(layer1_1_conv1_pipe_18, layer1_1_conv1_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer1_1_bn1[16384];
        for (int32_t n3 = 0; n3 < 1; ++n3) {
      for (int32_t c3 = 0; c3 < 16; ++c3) {
        for (int32_t h3 = 0; h3 < 32; ++h3) {
          for (int32_t w3 = 0; w3 < 32; ++w3) {
            int8_t layer1_1_conv1_temp1;
            layer1_1_conv1_temp1 = read_channel_intel(layer1_1_conv1_pipe_18);
            int32_t layer1_1_bn1_temp;
            layer1_1_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer1_1_conv1_temp1) * ((int40_t)w_layer1_1_bn1_9[c3]))) + ((int41_t)w_layer1_1_bn1_10[c3])));
            write_channel_intel(layer1_1_bn1_pipe_19, layer1_1_bn1_temp);
          }
        }
      }
    }
    int32_t layer1_1_residual1[16384];
        for (int32_t nn11 = 0; nn11 < 1; ++nn11) {
      for (int32_t cc10 = 0; cc10 < 16; ++cc10) {
        for (int32_t ww10 = 0; ww10 < 32; ++ww10) {
          for (int32_t hh10 = 0; hh10 < 32; ++hh10) {
            int32_t layer1_0_rprelu2_temp2;
            layer1_0_rprelu2_temp2 = read_channel_intel(layer1_0_rprelu2_pipe_117);
            int32_t layer1_1_bn1_temp1;
            layer1_1_bn1_temp1 = read_channel_intel(layer1_1_bn1_pipe_19);
            int32_t layer1_1_residual1_temp;
            layer1_1_residual1_temp = ((int32_t)(((int33_t)layer1_1_bn1_temp1) + ((int33_t)layer1_0_rprelu2_temp2)));
            write_channel_intel(layer1_1_residual1_pipe_20, layer1_1_residual1_temp);
          }
        }
      }
    }
    int32_t layer1_1_rprelu1[16384];
            for (int32_t nn12 = 0; nn12 < 1; ++nn12) {
      for (int32_t cc11 = 0; cc11 < 16; ++cc11) {
        for (int32_t ww11 = 0; ww11 < 32; ++ww11) {
          for (int32_t hh11 = 0; hh11 < 32; ++hh11) {
            int32_t layer1_1_residual1_temp1;
            layer1_1_residual1_temp1 = read_channel_intel(layer1_1_residual1_pipe_20);
            int32_t layer1_1_rprelu1_temp;
            layer1_1_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_1_residual1_temp1) + ((int33_t)w_layer1_1_rprelu1_0[cc11])))) ? (((int64_t)(((int33_t)layer1_1_residual1_temp1) + ((int33_t)w_layer1_1_rprelu1_0[cc11])))) : ((int64_t)(((int64_t)w_layer1_1_rprelu1_2[cc11]) * ((int64_t)(((int33_t)layer1_1_residual1_temp1) + ((int33_t)w_layer1_1_rprelu1_0[cc11]))))))) + ((int64_t)w_layer1_1_rprelu1_1[cc11])));
            write_channel_intel(layer1_1_rprelu1_pipe_118, layer1_1_rprelu1_temp);
            write_channel_intel(layer1_1_rprelu1_pipe_21, layer1_1_rprelu1_temp);
          }
        }
      }
    }
    uint16_t layer1_1_rsign2[1024];
        for (int32_t nn13 = 0; nn13 < 1; ++nn13) {
      for (int32_t cc12 = 0; cc12 < 1; ++cc12) {
        for (int32_t hh12 = 0; hh12 < 32; ++hh12) {
          for (int32_t ww12 = 0; ww12 < 32; ++ww12) {
            uint16_t layer1_1_rsign2_pack;
            for (int32_t i4 = 0; i4 < 16; ++i4) {
              int32_t layer1_1_rprelu1_temp1;
              layer1_1_rprelu1_temp1 = read_channel_intel(layer1_1_rprelu1_pipe_21);
              layer1_1_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_1_rprelu1_temp1) + ((int33_t)w_layer1_1_rsign2[((cc12 * 16) + i4)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint16_t layer1_1_rsign2_temp;
            layer1_1_rsign2_temp = layer1_1_rsign2_pack;
            write_channel_intel(layer1_1_rsign2_pipe_22, layer1_1_rsign2_temp);
          }
        }
      }
    }
    uint16_t layer1_1_conv2_pad[1156];
        for (int32_t ii3 = 0; ii3 < 1; ++ii3) {
      for (int32_t cc13 = 0; cc13 < 1; ++cc13) {
        for (int32_t hh13 = 0; hh13 < 34; ++hh13) {
          for (int32_t ww13 = 0; ww13 < 34; ++ww13) {
            bool cond = ((((1 <= ww13) && (ww13 < 33)) && (1 <= hh13)) && (hh13 < 33));
            uint16_t layer1_1_rsign2_temp1 = 0;
            if (cond) {
              layer1_1_rsign2_temp1 = read_channel_intel(layer1_1_rsign2_pipe_22);
            }
            uint16_t layer1_1_conv2_pad_temp;
            layer1_1_conv2_pad_temp = ((uint16_t)(uint32_t)(cond ? (((uint32_t)layer1_1_rsign2_temp1)) : ((uint32_t)0U)));
            write_channel_intel(layer1_1_conv2_pad_pipe_23, layer1_1_conv2_pad_temp);
            layer1_1_conv2_pad[(((ww13 + (hh13 * 34)) + (cc13 * 1156)) + (ii3 * 1156))] = layer1_1_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer1_1_conv2[16384];
    uint16_t layer1_1_conv2_LB[102];
    uint16_t layer1_1_conv2_WB[9];
        for (int32_t nn14 = 0; nn14 < 1; ++nn14) {
      for (int32_t yy_reuse4 = 0; yy_reuse4 < 34; ++yy_reuse4) {
        for (int32_t xx_reuse4 = 0; xx_reuse4 < 34; ++xx_reuse4) {
          for (int32_t layer1_1_conv2_pad_1 = 0; layer1_1_conv2_pad_1 < 2; ++layer1_1_conv2_pad_1) {
            layer1_1_conv2_LB[(xx_reuse4 + (layer1_1_conv2_pad_1 * 34))] = layer1_1_conv2_LB[((xx_reuse4 + (layer1_1_conv2_pad_1 * 34)) + 34)];
          }
          uint16_t layer1_1_conv2_pad_temp1;
          layer1_1_conv2_pad_temp1 = read_channel_intel(layer1_1_conv2_pad_pipe_23);
          layer1_1_conv2_LB[(xx_reuse4 + 68)] = layer1_1_conv2_pad_temp1;
          if (2 <= yy_reuse4) {
            for (int32_t layer1_1_conv2_LB_1 = 0; layer1_1_conv2_LB_1 < 3; ++layer1_1_conv2_LB_1) {
              for (int32_t layer1_1_conv2_LB_0 = 0; layer1_1_conv2_LB_0 < 2; ++layer1_1_conv2_LB_0) {
                layer1_1_conv2_WB[(layer1_1_conv2_LB_0 + (layer1_1_conv2_LB_1 * 3))] = layer1_1_conv2_WB[((layer1_1_conv2_LB_0 + (layer1_1_conv2_LB_1 * 3)) + 1)];
              }
              layer1_1_conv2_WB[((layer1_1_conv2_LB_1 * 3) + 2)] = layer1_1_conv2_LB[(xx_reuse4 + (layer1_1_conv2_LB_1 * 34))];
            }
            for (int32_t ff4 = 0; ff4 < 16; ++ff4) {
              if (2 <= xx_reuse4) {
                int8_t layer1_1_conv2_sum;
                for (int32_t layer1_1_conv2_rc = 0; layer1_1_conv2_rc < 1; ++layer1_1_conv2_rc) {
                  for (int32_t layer1_1_conv2_ry = 0; layer1_1_conv2_ry < 3; ++layer1_1_conv2_ry) {
                    for (int32_t layer1_1_conv2_rx = 0; layer1_1_conv2_rx < 3; ++layer1_1_conv2_rx) {
                      for (int32_t layer1_1_conv2_rb = 0; layer1_1_conv2_rb < 16; ++layer1_1_conv2_rb) {
                        layer1_1_conv2_sum = ((int8_t)(((int32_t)(((layer1_1_conv2_WB[(layer1_1_conv2_rx + (layer1_1_conv2_ry * 3))] ^ w_layer1_1_conv2[((layer1_1_conv2_rx + (layer1_1_conv2_ry * 3)) + (ff4 * 9))]) & (1L << layer1_1_conv2_rb)) >> layer1_1_conv2_rb)) + ((int32_t)layer1_1_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer1_1_conv2_temp;
                layer1_1_conv2_temp = ((int8_t)(144 - ((int32_t)(layer1_1_conv2_sum << 1))));
                write_channel_intel(layer1_1_conv2_pipe_24, layer1_1_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer1_1_bn2[16384];
        for (int32_t n4 = 0; n4 < 1; ++n4) {
      for (int32_t c4 = 0; c4 < 16; ++c4) {
        for (int32_t h4 = 0; h4 < 32; ++h4) {
          for (int32_t w4 = 0; w4 < 32; ++w4) {
            int8_t layer1_1_conv2_temp1;
            layer1_1_conv2_temp1 = read_channel_intel(layer1_1_conv2_pipe_24);
            int32_t layer1_1_bn2_temp;
            layer1_1_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer1_1_conv2_temp1) * ((int40_t)w_layer1_1_bn2_14[c4]))) + ((int41_t)w_layer1_1_bn2_15[c4])));
            write_channel_intel(layer1_1_bn2_pipe_25, layer1_1_bn2_temp);
          }
        }
      }
    }
    int32_t layer1_1_residual2[16384];
        for (int32_t nn15 = 0; nn15 < 1; ++nn15) {
      for (int32_t cc14 = 0; cc14 < 16; ++cc14) {
        for (int32_t ww14 = 0; ww14 < 32; ++ww14) {
          for (int32_t hh14 = 0; hh14 < 32; ++hh14) {
            int32_t layer1_1_rprelu1_temp2;
            layer1_1_rprelu1_temp2 = read_channel_intel(layer1_1_rprelu1_pipe_118);
            int32_t layer1_1_bn2_temp1;
            layer1_1_bn2_temp1 = read_channel_intel(layer1_1_bn2_pipe_25);
            int32_t layer1_1_residual2_temp;
            layer1_1_residual2_temp = ((int32_t)(((int33_t)layer1_1_bn2_temp1) + ((int33_t)layer1_1_rprelu1_temp2)));
            write_channel_intel(layer1_1_residual2_pipe_26, layer1_1_residual2_temp);
          }
        }
      }
    }
    int32_t layer1_1_rprelu2[16384];
            for (int32_t nn16 = 0; nn16 < 1; ++nn16) {
      for (int32_t cc15 = 0; cc15 < 16; ++cc15) {
        for (int32_t ww15 = 0; ww15 < 32; ++ww15) {
          for (int32_t hh15 = 0; hh15 < 32; ++hh15) {
            int32_t layer1_1_residual2_temp1;
            layer1_1_residual2_temp1 = read_channel_intel(layer1_1_residual2_pipe_26);
            int32_t layer1_1_rprelu2_temp;
            layer1_1_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_1_residual2_temp1) + ((int33_t)w_layer1_1_rprelu2_3[cc15])))) ? (((int64_t)(((int33_t)layer1_1_residual2_temp1) + ((int33_t)w_layer1_1_rprelu2_3[cc15])))) : ((int64_t)(((int64_t)w_layer1_1_rprelu2_5[cc15]) * ((int64_t)(((int33_t)layer1_1_residual2_temp1) + ((int33_t)w_layer1_1_rprelu2_3[cc15]))))))) + ((int64_t)w_layer1_1_rprelu2_4[cc15])));
            write_channel_intel(layer1_1_rprelu2_pipe_119, layer1_1_rprelu2_temp);
            write_channel_intel(layer1_1_rprelu2_pipe_27, layer1_1_rprelu2_temp);
          }
        }
      }
    }
    uint16_t layer1_2_rsign1[1024];
        for (int32_t nn17 = 0; nn17 < 1; ++nn17) {
      for (int32_t cc16 = 0; cc16 < 1; ++cc16) {
        for (int32_t hh16 = 0; hh16 < 32; ++hh16) {
          for (int32_t ww16 = 0; ww16 < 32; ++ww16) {
            uint16_t layer1_2_rsign1_pack;
            for (int32_t i5 = 0; i5 < 16; ++i5) {
              int32_t layer1_1_rprelu2_temp1;
              layer1_1_rprelu2_temp1 = read_channel_intel(layer1_1_rprelu2_pipe_27);
              layer1_2_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_1_rprelu2_temp1) + ((int33_t)w_layer1_2_rsign1[((cc16 * 16) + i5)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint16_t layer1_2_rsign1_temp;
            layer1_2_rsign1_temp = layer1_2_rsign1_pack;
            write_channel_intel(layer1_2_rsign1_pipe_28, layer1_2_rsign1_temp);
          }
        }
      }
    }
    uint16_t layer1_2_conv1_pad[1156];
        for (int32_t ii4 = 0; ii4 < 1; ++ii4) {
      for (int32_t cc17 = 0; cc17 < 1; ++cc17) {
        for (int32_t hh17 = 0; hh17 < 34; ++hh17) {
          for (int32_t ww17 = 0; ww17 < 34; ++ww17) {
            bool cond = ((((1 <= ww17) && (ww17 < 33)) && (1 <= hh17)) && (hh17 < 33));
            uint16_t layer1_2_rsign1_temp1 = 0;
            if (cond) {
              layer1_2_rsign1_temp1 = read_channel_intel(layer1_2_rsign1_pipe_28);
            }
            uint16_t layer1_2_conv1_pad_temp;
            layer1_2_conv1_pad_temp = ((uint16_t)(uint32_t)(cond ? (((uint32_t)layer1_2_rsign1_temp1)) : ((uint32_t)0U)));
            write_channel_intel(layer1_2_conv1_pad_pipe_29, layer1_2_conv1_pad_temp);
            layer1_2_conv1_pad[(((ww17 + (hh17 * 34)) + (cc17 * 1156)) + (ii4 * 1156))] = layer1_2_conv1_pad_temp;
          }
        }
      }
    }
    int8_t layer1_2_conv1[16384];
    uint16_t layer1_2_conv1_LB[102];
    uint16_t layer1_2_conv1_WB[9];
        for (int32_t nn18 = 0; nn18 < 1; ++nn18) {
      for (int32_t yy_reuse5 = 0; yy_reuse5 < 34; ++yy_reuse5) {
        for (int32_t xx_reuse5 = 0; xx_reuse5 < 34; ++xx_reuse5) {
          for (int32_t layer1_2_conv1_pad_1 = 0; layer1_2_conv1_pad_1 < 2; ++layer1_2_conv1_pad_1) {
            layer1_2_conv1_LB[(xx_reuse5 + (layer1_2_conv1_pad_1 * 34))] = layer1_2_conv1_LB[((xx_reuse5 + (layer1_2_conv1_pad_1 * 34)) + 34)];
          }
          uint16_t layer1_2_conv1_pad_temp1;
          layer1_2_conv1_pad_temp1 = read_channel_intel(layer1_2_conv1_pad_pipe_29);
          layer1_2_conv1_LB[(xx_reuse5 + 68)] = layer1_2_conv1_pad_temp1;
          if (2 <= yy_reuse5) {
            for (int32_t layer1_2_conv1_LB_1 = 0; layer1_2_conv1_LB_1 < 3; ++layer1_2_conv1_LB_1) {
              for (int32_t layer1_2_conv1_LB_0 = 0; layer1_2_conv1_LB_0 < 2; ++layer1_2_conv1_LB_0) {
                layer1_2_conv1_WB[(layer1_2_conv1_LB_0 + (layer1_2_conv1_LB_1 * 3))] = layer1_2_conv1_WB[((layer1_2_conv1_LB_0 + (layer1_2_conv1_LB_1 * 3)) + 1)];
              }
              layer1_2_conv1_WB[((layer1_2_conv1_LB_1 * 3) + 2)] = layer1_2_conv1_LB[(xx_reuse5 + (layer1_2_conv1_LB_1 * 34))];
            }
            for (int32_t ff5 = 0; ff5 < 16; ++ff5) {
              if (2 <= xx_reuse5) {
                int8_t layer1_2_conv1_sum;
                for (int32_t layer1_2_conv1_rc = 0; layer1_2_conv1_rc < 1; ++layer1_2_conv1_rc) {
                  for (int32_t layer1_2_conv1_ry = 0; layer1_2_conv1_ry < 3; ++layer1_2_conv1_ry) {
                    for (int32_t layer1_2_conv1_rx = 0; layer1_2_conv1_rx < 3; ++layer1_2_conv1_rx) {
                      for (int32_t layer1_2_conv1_rb = 0; layer1_2_conv1_rb < 16; ++layer1_2_conv1_rb) {
                        layer1_2_conv1_sum = ((int8_t)(((int32_t)(((layer1_2_conv1_WB[(layer1_2_conv1_rx + (layer1_2_conv1_ry * 3))] ^ w_layer1_2_conv1[((layer1_2_conv1_rx + (layer1_2_conv1_ry * 3)) + (ff5 * 9))]) & (1L << layer1_2_conv1_rb)) >> layer1_2_conv1_rb)) + ((int32_t)layer1_2_conv1_sum)));
                      }
                    }
                  }
                }
                int8_t layer1_2_conv1_temp;
                layer1_2_conv1_temp = ((int8_t)(144 - ((int32_t)(layer1_2_conv1_sum << 1))));
                write_channel_intel(layer1_2_conv1_pipe_30, layer1_2_conv1_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer1_2_bn1[16384];
        for (int32_t n5 = 0; n5 < 1; ++n5) {
      for (int32_t c5 = 0; c5 < 16; ++c5) {
        for (int32_t h5 = 0; h5 < 32; ++h5) {
          for (int32_t w5 = 0; w5 < 32; ++w5) {
            int8_t layer1_2_conv1_temp1;
            layer1_2_conv1_temp1 = read_channel_intel(layer1_2_conv1_pipe_30);
            int32_t layer1_2_bn1_temp;
            layer1_2_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer1_2_conv1_temp1) * ((int40_t)w_layer1_2_bn1_9[c5]))) + ((int41_t)w_layer1_2_bn1_10[c5])));
            write_channel_intel(layer1_2_bn1_pipe_31, layer1_2_bn1_temp);
          }
        }
      }
    }
    int32_t layer1_2_residual1[16384];
        for (int32_t nn19 = 0; nn19 < 1; ++nn19) {
      for (int32_t cc18 = 0; cc18 < 16; ++cc18) {
        for (int32_t ww18 = 0; ww18 < 32; ++ww18) {
          for (int32_t hh18 = 0; hh18 < 32; ++hh18) {
            int32_t layer1_2_bn1_temp1;
            layer1_2_bn1_temp1 = read_channel_intel(layer1_2_bn1_pipe_31);
            int32_t layer1_1_rprelu2_temp2;
            layer1_1_rprelu2_temp2 = read_channel_intel(layer1_1_rprelu2_pipe_119);
            int32_t layer1_2_residual1_temp;
            layer1_2_residual1_temp = ((int32_t)(((int33_t)layer1_2_bn1_temp1) + ((int33_t)layer1_1_rprelu2_temp2)));
            write_channel_intel(layer1_2_residual1_pipe_32, layer1_2_residual1_temp);
          }
        }
      }
    }
    int32_t layer1_2_rprelu1[16384];
            for (int32_t nn20 = 0; nn20 < 1; ++nn20) {
      for (int32_t cc19 = 0; cc19 < 16; ++cc19) {
        for (int32_t ww19 = 0; ww19 < 32; ++ww19) {
          for (int32_t hh19 = 0; hh19 < 32; ++hh19) {
            int32_t layer1_2_residual1_temp1;
            layer1_2_residual1_temp1 = read_channel_intel(layer1_2_residual1_pipe_32);
            int32_t layer1_2_rprelu1_temp;
            layer1_2_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_2_residual1_temp1) + ((int33_t)w_layer1_2_rprelu1_0[cc19])))) ? (((int64_t)(((int33_t)layer1_2_residual1_temp1) + ((int33_t)w_layer1_2_rprelu1_0[cc19])))) : ((int64_t)(((int64_t)w_layer1_2_rprelu1_2[cc19]) * ((int64_t)(((int33_t)layer1_2_residual1_temp1) + ((int33_t)w_layer1_2_rprelu1_0[cc19]))))))) + ((int64_t)w_layer1_2_rprelu1_1[cc19])));
            write_channel_intel(layer1_2_rprelu1_pipe_120, layer1_2_rprelu1_temp);
            write_channel_intel(layer1_2_rprelu1_pipe_33, layer1_2_rprelu1_temp);
          }
        }
      }
    }
    uint16_t layer1_2_rsign2[1024];
        for (int32_t nn21 = 0; nn21 < 1; ++nn21) {
      for (int32_t cc20 = 0; cc20 < 1; ++cc20) {
        for (int32_t hh20 = 0; hh20 < 32; ++hh20) {
          for (int32_t ww20 = 0; ww20 < 32; ++ww20) {
            uint16_t layer1_2_rsign2_pack;
            for (int32_t i6 = 0; i6 < 16; ++i6) {
              int32_t layer1_2_rprelu1_temp1;
              layer1_2_rprelu1_temp1 = read_channel_intel(layer1_2_rprelu1_pipe_33);
              layer1_2_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_2_rprelu1_temp1) + ((int33_t)w_layer1_2_rsign2[((cc20 * 16) + i6)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint16_t layer1_2_rsign2_temp;
            layer1_2_rsign2_temp = layer1_2_rsign2_pack;
            write_channel_intel(layer1_2_rsign2_pipe_34, layer1_2_rsign2_temp);
          }
        }
      }
    }
    uint16_t layer1_2_conv2_pad[1156];
        for (int32_t ii5 = 0; ii5 < 1; ++ii5) {
      for (int32_t cc21 = 0; cc21 < 1; ++cc21) {
        for (int32_t hh21 = 0; hh21 < 34; ++hh21) {
          for (int32_t ww21 = 0; ww21 < 34; ++ww21) {
            bool cond = ((((1 <= ww21) && (ww21 < 33)) && (1 <= hh21)) && (hh21 < 33));
            uint16_t layer1_2_rsign2_temp1 = 0;
            if (cond) {
              layer1_2_rsign2_temp1 = read_channel_intel(layer1_2_rsign2_pipe_34);
            }
            uint16_t layer1_2_conv2_pad_temp;
            layer1_2_conv2_pad_temp = ((uint16_t)(uint32_t)(cond ? (((uint32_t)layer1_2_rsign2_temp1)) : ((uint32_t)0U)));
            write_channel_intel(layer1_2_conv2_pad_pipe_35, layer1_2_conv2_pad_temp);
            layer1_2_conv2_pad[(((ww21 + (hh21 * 34)) + (cc21 * 1156)) + (ii5 * 1156))] = layer1_2_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer1_2_conv2[16384];
    uint16_t layer1_2_conv2_LB[102];
    uint16_t layer1_2_conv2_WB[9];
        for (int32_t nn22 = 0; nn22 < 1; ++nn22) {
      for (int32_t yy_reuse6 = 0; yy_reuse6 < 34; ++yy_reuse6) {
        for (int32_t xx_reuse6 = 0; xx_reuse6 < 34; ++xx_reuse6) {
          for (int32_t layer1_2_conv2_pad_1 = 0; layer1_2_conv2_pad_1 < 2; ++layer1_2_conv2_pad_1) {
            layer1_2_conv2_LB[(xx_reuse6 + (layer1_2_conv2_pad_1 * 34))] = layer1_2_conv2_LB[((xx_reuse6 + (layer1_2_conv2_pad_1 * 34)) + 34)];
          }
          uint16_t layer1_2_conv2_pad_temp1;
          layer1_2_conv2_pad_temp1 = read_channel_intel(layer1_2_conv2_pad_pipe_35);
          layer1_2_conv2_LB[(xx_reuse6 + 68)] = layer1_2_conv2_pad_temp1;
          if (2 <= yy_reuse6) {
            for (int32_t layer1_2_conv2_LB_1 = 0; layer1_2_conv2_LB_1 < 3; ++layer1_2_conv2_LB_1) {
              for (int32_t layer1_2_conv2_LB_0 = 0; layer1_2_conv2_LB_0 < 2; ++layer1_2_conv2_LB_0) {
                layer1_2_conv2_WB[(layer1_2_conv2_LB_0 + (layer1_2_conv2_LB_1 * 3))] = layer1_2_conv2_WB[((layer1_2_conv2_LB_0 + (layer1_2_conv2_LB_1 * 3)) + 1)];
              }
              layer1_2_conv2_WB[((layer1_2_conv2_LB_1 * 3) + 2)] = layer1_2_conv2_LB[(xx_reuse6 + (layer1_2_conv2_LB_1 * 34))];
            }
            for (int32_t ff6 = 0; ff6 < 16; ++ff6) {
              if (2 <= xx_reuse6) {
                int8_t layer1_2_conv2_sum;
                for (int32_t layer1_2_conv2_rc = 0; layer1_2_conv2_rc < 1; ++layer1_2_conv2_rc) {
                  for (int32_t layer1_2_conv2_ry = 0; layer1_2_conv2_ry < 3; ++layer1_2_conv2_ry) {
                    for (int32_t layer1_2_conv2_rx = 0; layer1_2_conv2_rx < 3; ++layer1_2_conv2_rx) {
                      for (int32_t layer1_2_conv2_rb = 0; layer1_2_conv2_rb < 16; ++layer1_2_conv2_rb) {
                        layer1_2_conv2_sum = ((int8_t)(((int32_t)(((layer1_2_conv2_WB[(layer1_2_conv2_rx + (layer1_2_conv2_ry * 3))] ^ w_layer1_2_conv2[((layer1_2_conv2_rx + (layer1_2_conv2_ry * 3)) + (ff6 * 9))]) & (1L << layer1_2_conv2_rb)) >> layer1_2_conv2_rb)) + ((int32_t)layer1_2_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer1_2_conv2_temp;
                layer1_2_conv2_temp = ((int8_t)(144 - ((int32_t)(layer1_2_conv2_sum << 1))));
                write_channel_intel(layer1_2_conv2_pipe_36, layer1_2_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer1_2_bn2[16384];
        for (int32_t n6 = 0; n6 < 1; ++n6) {
      for (int32_t c6 = 0; c6 < 16; ++c6) {
        for (int32_t h6 = 0; h6 < 32; ++h6) {
          for (int32_t w6 = 0; w6 < 32; ++w6) {
            int8_t layer1_2_conv2_temp1;
            layer1_2_conv2_temp1 = read_channel_intel(layer1_2_conv2_pipe_36);
            int32_t layer1_2_bn2_temp;
            layer1_2_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer1_2_conv2_temp1) * ((int40_t)w_layer1_2_bn2_14[c6]))) + ((int41_t)w_layer1_2_bn2_15[c6])));
            write_channel_intel(layer1_2_bn2_pipe_37, layer1_2_bn2_temp);
          }
        }
      }
    }
    int32_t layer1_2_residual2[16384];
        for (int32_t nn23 = 0; nn23 < 1; ++nn23) {
      for (int32_t cc22 = 0; cc22 < 16; ++cc22) {
        for (int32_t ww22 = 0; ww22 < 32; ++ww22) {
          for (int32_t hh22 = 0; hh22 < 32; ++hh22) {
            int32_t layer1_2_rprelu1_temp2;
            layer1_2_rprelu1_temp2 = read_channel_intel(layer1_2_rprelu1_pipe_120);
            int32_t layer1_2_bn2_temp1;
            layer1_2_bn2_temp1 = read_channel_intel(layer1_2_bn2_pipe_37);
            int32_t layer1_2_residual2_temp;
            layer1_2_residual2_temp = ((int32_t)(((int33_t)layer1_2_bn2_temp1) + ((int33_t)layer1_2_rprelu1_temp2)));
            write_channel_intel(layer1_2_residual2_pipe_38, layer1_2_residual2_temp);
          }
        }
      }
    }
    int32_t layer1_2_rprelu2[16384];
            for (int32_t nn24 = 0; nn24 < 1; ++nn24) {
      for (int32_t cc23 = 0; cc23 < 16; ++cc23) {
        for (int32_t ww23 = 0; ww23 < 32; ++ww23) {
          for (int32_t hh23 = 0; hh23 < 32; ++hh23) {
            int32_t layer1_2_residual2_temp1;
            layer1_2_residual2_temp1 = read_channel_intel(layer1_2_residual2_pipe_38);
            int32_t layer1_2_rprelu2_temp;
            layer1_2_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_2_residual2_temp1) + ((int33_t)w_layer1_2_rprelu2_3[cc23])))) ? (((int64_t)(((int33_t)layer1_2_residual2_temp1) + ((int33_t)w_layer1_2_rprelu2_3[cc23])))) : ((int64_t)(((int64_t)w_layer1_2_rprelu2_5[cc23]) * ((int64_t)(((int33_t)layer1_2_residual2_temp1) + ((int33_t)w_layer1_2_rprelu2_3[cc23]))))))) + ((int64_t)w_layer1_2_rprelu2_4[cc23])));
            write_channel_intel(layer1_2_rprelu2_pipe_121, layer1_2_rprelu2_temp);
            write_channel_intel(layer1_2_rprelu2_pipe_39, layer1_2_rprelu2_temp);
          }
        }
      }
    }
    uint16_t layer2_0_rsign1[1024];
    for (int32_t nn25 = 0; nn25 < 1; ++nn25) {
      for (int32_t cc24 = 0; cc24 < 1; ++cc24) {
        for (int32_t hh24 = 0; hh24 < 32; ++hh24) {
          for (int32_t ww24 = 0; ww24 < 32; ++ww24) {
            uint16_t layer2_0_rsign1_pack;
            for (int32_t i7 = 0; i7 < 16; ++i7) {
              int32_t layer1_2_rprelu2_temp1;
              layer1_2_rprelu2_temp1 = read_channel_intel(layer1_2_rprelu2_pipe_39);
              layer2_0_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer1_2_rprelu2_temp1) + ((int33_t)w_layer2_0_rsign1[((cc24 * 16) + i7)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint16_t layer2_0_rsign1_temp;
            layer2_0_rsign1_temp = layer2_0_rsign1_pack;
            write_channel_intel(layer2_0_rsign1_pipe_40, layer2_0_rsign1_temp);
          }
        }
      }
    }
    uint16_t layer2_0_conv1_pad[1156];
    for (int32_t ii6 = 0; ii6 < 1; ++ii6) {
      for (int32_t cc25 = 0; cc25 < 1; ++cc25) {
        for (int32_t hh25 = 0; hh25 < 34; ++hh25) {
          for (int32_t ww25 = 0; ww25 < 34; ++ww25) {
            bool cond = ((((1 <= ww25) && (ww25 < 33)) && (1 <= hh25)) && (hh25 < 33));
            uint16_t layer2_0_rsign1_temp1 = 0;
            if (cond) {
              layer2_0_rsign1_temp1 = read_channel_intel(layer2_0_rsign1_pipe_40);
            }
            uint16_t layer2_0_conv1_pad_temp;
            layer2_0_conv1_pad_temp = ((uint16_t)(uint32_t)(cond ? (((uint32_t)layer2_0_rsign1_temp1)) : ((uint32_t)0U)));
            write_channel_intel(layer2_0_conv1_pad_pipe_41, layer2_0_conv1_pad_temp);
          }
        }
      }
    }

    int8_t layer2_0_conv1[1][32][16][16];
    int8_t layer2_0_conv1_LB[1][1][3][18];
    int8_t layer2_0_conv1_WB[1][1][3][3];

    for (int yy_reuse = 0; yy_reuse < 34; ++yy_reuse) {
      for (int xx_reuse = 0; xx_reuse < 34; ++xx_reuse) {
        for (int layer2_0_conv1_pad_1 = 0; layer2_0_conv1_pad_1 < 2; ++layer2_0_conv1_pad_1) {
            layer2_0_conv1_LB[0][0][layer2_0_conv1_pad_1][xx_reuse] = layer2_0_conv1_LB[0][0][(layer2_0_conv1_pad_1 + 1)][xx_reuse];
          }
          int8_t layer2_0_conv1_temp1 = read_channel_intel(layer2_0_conv1_pad_pipe_41);
          layer2_0_conv1_LB[0][0][2][xx_reuse] = layer2_0_conv1_temp1;
          if (2 <= yy_reuse && ((yy_reuse - 2) % 2 == 0)) { // not so correct
            for (int layer2_0_conv1_LB_1 = 0; layer2_0_conv1_LB_1 < 3; ++layer2_0_conv1_LB_1) {
              for (int layer2_0_conv1_LB_0 = 0; layer2_0_conv1_LB_0 < 2; ++layer2_0_conv1_LB_0) {
                layer2_0_conv1_WB[0][0][layer2_0_conv1_LB_1][layer2_0_conv1_LB_0] = layer2_0_conv1_WB[0][0][layer2_0_conv1_LB_1][(layer2_0_conv1_LB_0 + 1)];
              }
              layer2_0_conv1_WB[0][0][layer2_0_conv1_LB_1][2] = layer2_0_conv1_LB[0][0][layer2_0_conv1_LB_1][xx_reuse];
            }
            if (2 <= xx_reuse && ((xx_reuse - 2) % 2 == 0)) {
              int16_t layer2_0_conv1_sum;
              layer2_0_conv1_sum = 0;
    for (int ff = 0; ff < 32; ++ff) {
              for (int layer2_0_conv1_ry = 0; layer2_0_conv1_ry < 3; ++layer2_0_conv1_ry) {
                for (int layer2_0_conv1_rx = 0; layer2_0_conv1_rx < 3; ++layer2_0_conv1_rx) {
                  for (int layer2_0_conv1_rb = 0; layer2_0_conv1_rb < 16; ++layer2_0_conv1_rb) {
                    layer2_0_conv1_sum = ((((layer2_0_conv1_WB[0][0][layer2_0_conv1_ry][layer2_0_conv1_rx] ^ w_layer2_0_conv1[ff][0][layer2_0_conv1_ry][layer2_0_conv1_rx]) & (1 << layer2_0_conv1_rb)) + (layer2_0_conv1_sum)));
                  }
                }
              }
              int8_t layer2_0_conv1_temp;
              layer2_0_conv1_temp = ((int8_t)(144 - ((int32_t)(layer2_0_conv1_sum << 1))));
              write_channel_intel(layer2_0_conv1_pipe_42, layer2_0_conv1_temp);
            }
          }
        }
      }
    }

    int32_t layer2_0_bn1[8192];
    for (int32_t n7 = 0; n7 < 1; ++n7) {
      for (int32_t c7 = 0; c7 < 32; ++c7) {
        for (int32_t h7 = 0; h7 < 16; ++h7) {
          for (int32_t w7 = 0; w7 < 16; ++w7) {
            int8_t layer2_0_conv1_temp1;
            layer2_0_conv1_temp1 = read_channel_intel(layer2_0_conv1_pipe_42);
            int32_t layer2_0_bn1_temp;
            layer2_0_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer2_0_conv1_temp1) * ((int40_t)w_layer2_0_bn1_9[c7]))) + ((int41_t)w_layer2_0_bn1_10[c7])));
            write_channel_intel(layer2_0_bn1_pipe_43, layer2_0_bn1_temp);
          }
        }
      }
    }
    int32_t layer2_0_avgpool_res[4096];
    int32_t layer2_0_avgpool_LB[64];
    int32_t layer2_0_avgpool;
    for (int32_t ii7 = 0; ii7 < 1; ++ii7) {
      for (int32_t cc26 = 0; cc26 < 16; ++cc26) {
        for (int32_t hh26 = 0; hh26 < 16; ++hh26) {
          for (int32_t layer2_0_avgpool_LB_i = 0; layer2_0_avgpool_LB_i < 2; ++layer2_0_avgpool_LB_i) {
            for (int32_t layer2_0_avgpool_LB_j = 0; layer2_0_avgpool_LB_j < 32; ++layer2_0_avgpool_LB_j) {
              int32_t layer1_2_rprelu2_temp2;
              layer1_2_rprelu2_temp2 = read_channel_intel(layer1_2_rprelu2_pipe_121);
              layer2_0_avgpool_LB[(layer2_0_avgpool_LB_j + (layer2_0_avgpool_LB_i * 32))] = layer1_2_rprelu2_temp2;
            }
          }
          for (int32_t layer2_0_avgpool_ww = 0; layer2_0_avgpool_ww < 16; ++layer2_0_avgpool_ww) {
            int32_t layer2_0_avgpool_val;
            for (int32_t layer2_0_avgpool_ry = 0; layer2_0_avgpool_ry < 2; ++layer2_0_avgpool_ry) {
              for (int32_t layer2_0_avgpool_rx = 0; layer2_0_avgpool_rx < 2; ++layer2_0_avgpool_rx) {
                layer2_0_avgpool_val = ((int32_t)(((int33_t)layer2_0_avgpool_val) + ((int33_t)layer2_0_avgpool_LB[(((layer2_0_avgpool_ww * 2) + layer2_0_avgpool_rx) + (layer2_0_avgpool_ry * 32))])));
              }
            }
            int32_t layer2_0_avgpool_res_temp;
            layer2_0_avgpool_res_temp = ((int32_t)(((int64_t)layer2_0_avgpool_val) / (int64_t)4));
            write_channel_intel(layer2_0_avgpool_res_pipe_122, layer2_0_avgpool_res_temp);
            layer2_0_avgpool_res[(((layer2_0_avgpool_ww + (hh26 * 16)) + (cc26 * 256)) + (ii7 * 4096))] = layer2_0_avgpool_res_temp;
          }
        }
      }
    }
    int32_t layer2_0_concat[8192];
        for (int32_t nn27 = 0; nn27 < 1; ++nn27) {
      for (int32_t cc27 = 0; cc27 < 32; ++cc27) {
        for (int32_t ww26 = 0; ww26 < 16; ++ww26) {
          for (int32_t hh27 = 0; hh27 < 16; ++hh27) {
            int32_t layer2_0_avgpool_res_temp1;
            layer2_0_avgpool_res_temp1 = read_channel_intel(layer2_0_avgpool_res_pipe_122);
            int32_t layer2_0_concat_temp;
            layer2_0_concat_temp = layer2_0_avgpool_res_temp1;
            write_channel_intel(layer2_0_concat_pipe_123, layer2_0_concat_temp);
          }
        }
      }
    }
    int32_t layer2_0_residual1[8192];
        for (int32_t nn28 = 0; nn28 < 1; ++nn28) {
      for (int32_t cc28 = 0; cc28 < 32; ++cc28) {
        for (int32_t ww27 = 0; ww27 < 16; ++ww27) {
          for (int32_t hh28 = 0; hh28 < 16; ++hh28) {
            int32_t layer2_0_bn1_temp1;
            layer2_0_bn1_temp1 = read_channel_intel(layer2_0_bn1_pipe_43);
            int32_t layer2_0_concat_temp1;
            layer2_0_concat_temp1 = read_channel_intel(layer2_0_concat_pipe_123);
            int32_t layer2_0_residual1_temp;
            layer2_0_residual1_temp = ((int32_t)(((int33_t)layer2_0_bn1_temp1) + ((int33_t)layer2_0_concat_temp1)));
            write_channel_intel(layer2_0_residual1_pipe_44, layer2_0_residual1_temp);
          }
        }
      }
    }
    int32_t layer2_0_rprelu1[8192];
            for (int32_t nn29 = 0; nn29 < 1; ++nn29) {
      for (int32_t cc29 = 0; cc29 < 32; ++cc29) {
        for (int32_t ww28 = 0; ww28 < 16; ++ww28) {
          for (int32_t hh29 = 0; hh29 < 16; ++hh29) {
            int32_t layer2_0_residual1_temp1;
            layer2_0_residual1_temp1 = read_channel_intel(layer2_0_residual1_pipe_44);
            int32_t layer2_0_rprelu1_temp;
            layer2_0_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_0_residual1_temp1) + ((int33_t)w_layer2_0_rprelu1_0[cc29])))) ? (((int64_t)(((int33_t)layer2_0_residual1_temp1) + ((int33_t)w_layer2_0_rprelu1_0[cc29])))) : ((int64_t)(((int64_t)w_layer2_0_rprelu1_2[cc29]) * ((int64_t)(((int33_t)layer2_0_residual1_temp1) + ((int33_t)w_layer2_0_rprelu1_0[cc29]))))))) + ((int64_t)w_layer2_0_rprelu1_1[cc29])));
            write_channel_intel(layer2_0_rprelu1_pipe_124, layer2_0_rprelu1_temp);
            write_channel_intel(layer2_0_rprelu1_pipe_45, layer2_0_rprelu1_temp);
          }
        }
      }
    }
    uint32_t layer2_0_rsign2[256];
        for (int32_t nn30 = 0; nn30 < 1; ++nn30) {
      for (int32_t cc30 = 0; cc30 < 1; ++cc30) {
        for (int32_t hh30 = 0; hh30 < 16; ++hh30) {
          for (int32_t ww29 = 0; ww29 < 16; ++ww29) {
            uint32_t layer2_0_rsign2_pack;
            for (int32_t i8 = 0; i8 < 32; ++i8) {
              int32_t layer2_0_rprelu1_temp1;
              layer2_0_rprelu1_temp1 = read_channel_intel(layer2_0_rprelu1_pipe_45);
              layer2_0_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_0_rprelu1_temp1) + ((int33_t)w_layer2_0_rsign2[((cc30 * 32) + i8)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer2_0_rsign2_temp;
            layer2_0_rsign2_temp = layer2_0_rsign2_pack;
            write_channel_intel(layer2_0_rsign2_pipe_46, layer2_0_rsign2_temp);
          }
        }
      }
    }
    uint32_t layer2_0_conv2_pad[324];
        for (int32_t ii8 = 0; ii8 < 1; ++ii8) {
      for (int32_t cc31 = 0; cc31 < 1; ++cc31) {
        for (int32_t hh31 = 0; hh31 < 18; ++hh31) {
          for (int32_t ww30 = 0; ww30 < 18; ++ww30) {
            bool cond = ((((1 <= ww30) && (ww30 < 17)) && (1 <= hh31)) && (hh31 < 17));
            uint32_t layer2_0_rsign2_temp1 = 0;
            if (cond) {
              layer2_0_rsign2_temp1 = read_channel_intel(layer2_0_rsign2_pipe_46);
            }
            uint32_t layer2_0_conv2_pad_temp;
            layer2_0_conv2_pad_temp = (uint32_t)(cond ? ((uint32_t)layer2_0_rsign2_temp1) : ((uint32_t)0U));
            write_channel_intel(layer2_0_conv2_pad_pipe_47, layer2_0_conv2_pad_temp);
            layer2_0_conv2_pad[(((ww30 + (hh31 * 18)) + (cc31 * 324)) + (ii8 * 324))] = layer2_0_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer2_0_conv2[8192];
    uint32_t layer2_0_conv2_LB[54];
    uint32_t layer2_0_conv2_WB[9];
        for (int32_t nn31 = 0; nn31 < 1; ++nn31) {
      for (int32_t yy_reuse7 = 0; yy_reuse7 < 18; ++yy_reuse7) {
        for (int32_t xx_reuse7 = 0; xx_reuse7 < 18; ++xx_reuse7) {
          for (int32_t layer2_0_conv2_pad_1 = 0; layer2_0_conv2_pad_1 < 2; ++layer2_0_conv2_pad_1) {
            layer2_0_conv2_LB[(xx_reuse7 + (layer2_0_conv2_pad_1 * 18))] = layer2_0_conv2_LB[((xx_reuse7 + (layer2_0_conv2_pad_1 * 18)) + 18)];
          }
          uint32_t layer2_0_conv2_pad_temp1;
          layer2_0_conv2_pad_temp1 = read_channel_intel(layer2_0_conv2_pad_pipe_47);
          layer2_0_conv2_LB[(xx_reuse7 + 36)] = layer2_0_conv2_pad_temp1;
          if (2 <= yy_reuse7) {
            for (int32_t layer2_0_conv2_LB_1 = 0; layer2_0_conv2_LB_1 < 3; ++layer2_0_conv2_LB_1) {
              for (int32_t layer2_0_conv2_LB_0 = 0; layer2_0_conv2_LB_0 < 2; ++layer2_0_conv2_LB_0) {
                layer2_0_conv2_WB[(layer2_0_conv2_LB_0 + (layer2_0_conv2_LB_1 * 3))] = layer2_0_conv2_WB[((layer2_0_conv2_LB_0 + (layer2_0_conv2_LB_1 * 3)) + 1)];
              }
              layer2_0_conv2_WB[((layer2_0_conv2_LB_1 * 3) + 2)] = layer2_0_conv2_LB[(xx_reuse7 + (layer2_0_conv2_LB_1 * 18))];
            }
            for (int32_t ff8 = 0; ff8 < 32; ++ff8) {
              if (2 <= xx_reuse7) {
                int8_t layer2_0_conv2_sum;
                for (int32_t layer2_0_conv2_rc = 0; layer2_0_conv2_rc < 1; ++layer2_0_conv2_rc) {
                  for (int32_t layer2_0_conv2_ry = 0; layer2_0_conv2_ry < 3; ++layer2_0_conv2_ry) {
                    for (int32_t layer2_0_conv2_rx = 0; layer2_0_conv2_rx < 3; ++layer2_0_conv2_rx) {
                      for (int32_t layer2_0_conv2_rb = 0; layer2_0_conv2_rb < 32; ++layer2_0_conv2_rb) {
                        layer2_0_conv2_sum = ((int8_t)(((int64_t)(((layer2_0_conv2_WB[(layer2_0_conv2_rx + (layer2_0_conv2_ry * 3))] ^ w_layer2_0_conv2[((layer2_0_conv2_rx + (layer2_0_conv2_ry * 3)) + (ff8 * 9))]) & (1L << layer2_0_conv2_rb)) >> layer2_0_conv2_rb)) + ((int64_t)layer2_0_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer2_0_conv2_temp;
                layer2_0_conv2_temp = ((int8_t)(288 - ((int32_t)(layer2_0_conv2_sum << 1))));
                write_channel_intel(layer2_0_conv2_pipe_48, layer2_0_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer2_0_bn2[8192];
        for (int32_t n8 = 0; n8 < 1; ++n8) {
      for (int32_t c8 = 0; c8 < 32; ++c8) {
        for (int32_t h8 = 0; h8 < 16; ++h8) {
          for (int32_t w8 = 0; w8 < 16; ++w8) {
            int8_t layer2_0_conv2_temp1;
            layer2_0_conv2_temp1 = read_channel_intel(layer2_0_conv2_pipe_48);
            int32_t layer2_0_bn2_temp;
            layer2_0_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer2_0_conv2_temp1) * ((int40_t)w_layer2_0_bn2_14[c8]))) + ((int41_t)w_layer2_0_bn2_15[c8])));
            write_channel_intel(layer2_0_bn2_pipe_49, layer2_0_bn2_temp);
          }
        }
      }
    }
    int32_t layer2_0_residual2[8192];
        for (int32_t nn32 = 0; nn32 < 1; ++nn32) {
      for (int32_t cc32 = 0; cc32 < 32; ++cc32) {
        for (int32_t ww31 = 0; ww31 < 16; ++ww31) {
          for (int32_t hh32 = 0; hh32 < 16; ++hh32) {
            int32_t layer2_0_bn2_temp1;
            layer2_0_bn2_temp1 = read_channel_intel(layer2_0_bn2_pipe_49);
            int32_t layer2_0_rprelu1_temp2;
            layer2_0_rprelu1_temp2 = read_channel_intel(layer2_0_rprelu1_pipe_124);
            int32_t layer2_0_residual2_temp;
            layer2_0_residual2_temp = ((int32_t)(((int33_t)layer2_0_bn2_temp1) + ((int33_t)layer2_0_rprelu1_temp2)));
            write_channel_intel(layer2_0_residual2_pipe_50, layer2_0_residual2_temp);
          }
        }
      }
    }
    int32_t layer2_0_rprelu2[8192];
            for (int32_t nn33 = 0; nn33 < 1; ++nn33) {
      for (int32_t cc33 = 0; cc33 < 32; ++cc33) {
        for (int32_t ww32 = 0; ww32 < 16; ++ww32) {
          for (int32_t hh33 = 0; hh33 < 16; ++hh33) {
            int32_t layer2_0_residual2_temp1;
            layer2_0_residual2_temp1 = read_channel_intel(layer2_0_residual2_pipe_50);
            int32_t layer2_0_rprelu2_temp;
            layer2_0_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_0_residual2_temp1) + ((int33_t)w_layer2_0_rprelu2_3[cc33])))) ? (((int64_t)(((int33_t)layer2_0_residual2_temp1) + ((int33_t)w_layer2_0_rprelu2_3[cc33])))) : ((int64_t)(((int64_t)w_layer2_0_rprelu2_5[cc33]) * ((int64_t)(((int33_t)layer2_0_residual2_temp1) + ((int33_t)w_layer2_0_rprelu2_3[cc33]))))))) + ((int64_t)w_layer2_0_rprelu2_4[cc33])));
            write_channel_intel(layer2_0_rprelu2_pipe_125, layer2_0_rprelu2_temp);
            write_channel_intel(layer2_0_rprelu2_pipe_51, layer2_0_rprelu2_temp);
          }
        }
      }
    }
    uint32_t layer2_1_rsign1[256];
        for (int32_t nn34 = 0; nn34 < 1; ++nn34) {
      for (int32_t cc34 = 0; cc34 < 1; ++cc34) {
        for (int32_t hh34 = 0; hh34 < 16; ++hh34) {
          for (int32_t ww33 = 0; ww33 < 16; ++ww33) {
            uint32_t layer2_1_rsign1_pack;
            for (int32_t i9 = 0; i9 < 32; ++i9) {
              int32_t layer2_0_rprelu2_temp1;
              layer2_0_rprelu2_temp1 = read_channel_intel(layer2_0_rprelu2_pipe_51);
              layer2_1_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_0_rprelu2_temp1) + ((int33_t)w_layer2_1_rsign1[((cc34 * 32) + i9)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer2_1_rsign1_temp;
            layer2_1_rsign1_temp = layer2_1_rsign1_pack;
            write_channel_intel(layer2_1_rsign1_pipe_52, layer2_1_rsign1_temp);
          }
        }
      }
    }
    uint32_t layer2_1_conv1_pad[324];
        for (int32_t ii9 = 0; ii9 < 1; ++ii9) {
      for (int32_t cc35 = 0; cc35 < 1; ++cc35) {
        for (int32_t hh35 = 0; hh35 < 18; ++hh35) {
          for (int32_t ww34 = 0; ww34 < 18; ++ww34) {
            bool cond = ((((1 <= ww34) && (ww34 < 17)) && (1 <= hh35)) && (hh35 < 17));
            uint32_t layer2_1_rsign1_temp1 = 0;
            if (cond) {
              layer2_1_rsign1_temp1 = read_channel_intel(layer2_1_rsign1_pipe_52);
            }
            uint32_t layer2_1_conv1_pad_temp;
            layer2_1_conv1_pad_temp = (uint32_t)(cond ? ((uint32_t)layer2_1_rsign1_temp1) : ((uint32_t)0U));
            write_channel_intel(layer2_1_conv1_pad_pipe_53, layer2_1_conv1_pad_temp);
            layer2_1_conv1_pad[(((ww34 + (hh35 * 18)) + (cc35 * 324)) + (ii9 * 324))] = layer2_1_conv1_pad_temp;
          }
        }
      }
    }
    int8_t layer2_1_conv1[8192];
    uint32_t layer2_1_conv1_LB[54];
    uint32_t layer2_1_conv1_WB[9];
        for (int32_t nn35 = 0; nn35 < 1; ++nn35) {
      for (int32_t yy_reuse8 = 0; yy_reuse8 < 18; ++yy_reuse8) {
        for (int32_t xx_reuse8 = 0; xx_reuse8 < 18; ++xx_reuse8) {
          for (int32_t layer2_1_conv1_pad_1 = 0; layer2_1_conv1_pad_1 < 2; ++layer2_1_conv1_pad_1) {
            layer2_1_conv1_LB[(xx_reuse8 + (layer2_1_conv1_pad_1 * 18))] = layer2_1_conv1_LB[((xx_reuse8 + (layer2_1_conv1_pad_1 * 18)) + 18)];
          }
          uint32_t layer2_1_conv1_pad_temp1;
          layer2_1_conv1_pad_temp1 = read_channel_intel(layer2_1_conv1_pad_pipe_53);
          layer2_1_conv1_LB[(xx_reuse8 + 36)] = layer2_1_conv1_pad_temp1;
          if (2 <= yy_reuse8) {
            for (int32_t layer2_1_conv1_LB_1 = 0; layer2_1_conv1_LB_1 < 3; ++layer2_1_conv1_LB_1) {
              for (int32_t layer2_1_conv1_LB_0 = 0; layer2_1_conv1_LB_0 < 2; ++layer2_1_conv1_LB_0) {
                layer2_1_conv1_WB[(layer2_1_conv1_LB_0 + (layer2_1_conv1_LB_1 * 3))] = layer2_1_conv1_WB[((layer2_1_conv1_LB_0 + (layer2_1_conv1_LB_1 * 3)) + 1)];
              }
              layer2_1_conv1_WB[((layer2_1_conv1_LB_1 * 3) + 2)] = layer2_1_conv1_LB[(xx_reuse8 + (layer2_1_conv1_LB_1 * 18))];
            }
            for (int32_t ff9 = 0; ff9 < 32; ++ff9) {
              if (2 <= xx_reuse8) {
                int8_t layer2_1_conv1_sum;
                for (int32_t layer2_1_conv1_rc = 0; layer2_1_conv1_rc < 1; ++layer2_1_conv1_rc) {
                  for (int32_t layer2_1_conv1_ry = 0; layer2_1_conv1_ry < 3; ++layer2_1_conv1_ry) {
                    for (int32_t layer2_1_conv1_rx = 0; layer2_1_conv1_rx < 3; ++layer2_1_conv1_rx) {
                      for (int32_t layer2_1_conv1_rb = 0; layer2_1_conv1_rb < 32; ++layer2_1_conv1_rb) {
                        layer2_1_conv1_sum = ((int8_t)(((int64_t)(((layer2_1_conv1_WB[(layer2_1_conv1_rx + (layer2_1_conv1_ry * 3))] ^ w_layer2_1_conv1[((layer2_1_conv1_rx + (layer2_1_conv1_ry * 3)) + (ff9 * 9))]) & (1L << layer2_1_conv1_rb)) >> layer2_1_conv1_rb)) + ((int64_t)layer2_1_conv1_sum)));
                      }
                    }
                  }
                }
                int8_t layer2_1_conv1_temp;
                layer2_1_conv1_temp = ((int8_t)(288 - ((int32_t)(layer2_1_conv1_sum << 1))));
                write_channel_intel(layer2_1_conv1_pipe_54, layer2_1_conv1_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer2_1_bn1[8192];
        for (int32_t n9 = 0; n9 < 1; ++n9) {
      for (int32_t c9 = 0; c9 < 32; ++c9) {
        for (int32_t h9 = 0; h9 < 16; ++h9) {
          for (int32_t w9 = 0; w9 < 16; ++w9) {
            int8_t layer2_1_conv1_temp1;
            layer2_1_conv1_temp1 = read_channel_intel(layer2_1_conv1_pipe_54);
            int32_t layer2_1_bn1_temp;
            layer2_1_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer2_1_conv1_temp1) * ((int40_t)w_layer2_1_bn1_9[c9]))) + ((int41_t)w_layer2_1_bn1_10[c9])));
            write_channel_intel(layer2_1_bn1_pipe_55, layer2_1_bn1_temp);
          }
        }
      }
    }
    int32_t layer2_1_residual1[8192];
        for (int32_t nn36 = 0; nn36 < 1; ++nn36) {
      for (int32_t cc36 = 0; cc36 < 32; ++cc36) {
        for (int32_t ww35 = 0; ww35 < 16; ++ww35) {
          for (int32_t hh36 = 0; hh36 < 16; ++hh36) {
            int32_t layer2_1_bn1_temp1;
            layer2_1_bn1_temp1 = read_channel_intel(layer2_1_bn1_pipe_55);
            int32_t layer2_0_rprelu2_temp2;
            layer2_0_rprelu2_temp2 = read_channel_intel(layer2_0_rprelu2_pipe_125);
            int32_t layer2_1_residual1_temp;
            layer2_1_residual1_temp = ((int32_t)(((int33_t)layer2_1_bn1_temp1) + ((int33_t)layer2_0_rprelu2_temp2)));
            write_channel_intel(layer2_1_residual1_pipe_56, layer2_1_residual1_temp);
          }
        }
      }
    }
    int32_t layer2_1_rprelu1[8192];
            for (int32_t nn37 = 0; nn37 < 1; ++nn37) {
      for (int32_t cc37 = 0; cc37 < 32; ++cc37) {
        for (int32_t ww36 = 0; ww36 < 16; ++ww36) {
          for (int32_t hh37 = 0; hh37 < 16; ++hh37) {
            int32_t layer2_1_residual1_temp1;
            layer2_1_residual1_temp1 = read_channel_intel(layer2_1_residual1_pipe_56);
            int32_t layer2_1_rprelu1_temp;
            layer2_1_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_1_residual1_temp1) + ((int33_t)w_layer2_1_rprelu1_0[cc37])))) ? (((int64_t)(((int33_t)layer2_1_residual1_temp1) + ((int33_t)w_layer2_1_rprelu1_0[cc37])))) : ((int64_t)(((int64_t)w_layer2_1_rprelu1_2[cc37]) * ((int64_t)(((int33_t)layer2_1_residual1_temp1) + ((int33_t)w_layer2_1_rprelu1_0[cc37]))))))) + ((int64_t)w_layer2_1_rprelu1_1[cc37])));
            write_channel_intel(layer2_1_rprelu1_pipe_126, layer2_1_rprelu1_temp);
            write_channel_intel(layer2_1_rprelu1_pipe_57, layer2_1_rprelu1_temp);
          }
        }
      }
    }
    uint32_t layer2_1_rsign2[256];
        for (int32_t nn38 = 0; nn38 < 1; ++nn38) {
      for (int32_t cc38 = 0; cc38 < 1; ++cc38) {
        for (int32_t hh38 = 0; hh38 < 16; ++hh38) {
          for (int32_t ww37 = 0; ww37 < 16; ++ww37) {
            uint32_t layer2_1_rsign2_pack;
            for (int32_t i10 = 0; i10 < 32; ++i10) {
              int32_t layer2_1_rprelu1_temp1;
              layer2_1_rprelu1_temp1 = read_channel_intel(layer2_1_rprelu1_pipe_57);
              layer2_1_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_1_rprelu1_temp1) + ((int33_t)w_layer2_1_rsign2[((cc38 * 32) + i10)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer2_1_rsign2_temp;
            layer2_1_rsign2_temp = layer2_1_rsign2_pack;
            write_channel_intel(layer2_1_rsign2_pipe_58, layer2_1_rsign2_temp);
          }
        }
      }
    }
    uint32_t layer2_1_conv2_pad[324];
        for (int32_t ii10 = 0; ii10 < 1; ++ii10) {
      for (int32_t cc39 = 0; cc39 < 1; ++cc39) {
        for (int32_t hh39 = 0; hh39 < 18; ++hh39) {
          for (int32_t ww38 = 0; ww38 < 18; ++ww38) {
            bool cond = ((((1 <= ww38) && (ww38 < 17)) && (1 <= hh39)) && (hh39 < 17));
            uint32_t layer2_1_rsign2_temp1 = 0;
            if (cond) {
              layer2_1_rsign2_temp1 = read_channel_intel(layer2_1_rsign2_pipe_58);
            }
            uint32_t layer2_1_conv2_pad_temp;
            layer2_1_conv2_pad_temp = (uint32_t)(cond ? ((uint32_t)layer2_1_rsign2_temp1) : ((uint32_t)0U));
            write_channel_intel(layer2_1_conv2_pad_pipe_59, layer2_1_conv2_pad_temp);
            layer2_1_conv2_pad[(((ww38 + (hh39 * 18)) + (cc39 * 324)) + (ii10 * 324))] = layer2_1_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer2_1_conv2[8192];
    uint32_t layer2_1_conv2_LB[54];
    uint32_t layer2_1_conv2_WB[9];
        for (int32_t nn39 = 0; nn39 < 1; ++nn39) {
      for (int32_t yy_reuse9 = 0; yy_reuse9 < 18; ++yy_reuse9) {
        for (int32_t xx_reuse9 = 0; xx_reuse9 < 18; ++xx_reuse9) {
          for (int32_t layer2_1_conv2_pad_1 = 0; layer2_1_conv2_pad_1 < 2; ++layer2_1_conv2_pad_1) {
            layer2_1_conv2_LB[(xx_reuse9 + (layer2_1_conv2_pad_1 * 18))] = layer2_1_conv2_LB[((xx_reuse9 + (layer2_1_conv2_pad_1 * 18)) + 18)];
          }
          uint32_t layer2_1_conv2_pad_temp1;
          layer2_1_conv2_pad_temp1 = read_channel_intel(layer2_1_conv2_pad_pipe_59);
          layer2_1_conv2_LB[(xx_reuse9 + 36)] = layer2_1_conv2_pad_temp1;
          if (2 <= yy_reuse9) {
            for (int32_t layer2_1_conv2_LB_1 = 0; layer2_1_conv2_LB_1 < 3; ++layer2_1_conv2_LB_1) {
              for (int32_t layer2_1_conv2_LB_0 = 0; layer2_1_conv2_LB_0 < 2; ++layer2_1_conv2_LB_0) {
                layer2_1_conv2_WB[(layer2_1_conv2_LB_0 + (layer2_1_conv2_LB_1 * 3))] = layer2_1_conv2_WB[((layer2_1_conv2_LB_0 + (layer2_1_conv2_LB_1 * 3)) + 1)];
              }
              layer2_1_conv2_WB[((layer2_1_conv2_LB_1 * 3) + 2)] = layer2_1_conv2_LB[(xx_reuse9 + (layer2_1_conv2_LB_1 * 18))];
            }
            for (int32_t ff10 = 0; ff10 < 32; ++ff10) {
              if (2 <= xx_reuse9) {
                int8_t layer2_1_conv2_sum;
                for (int32_t layer2_1_conv2_rc = 0; layer2_1_conv2_rc < 1; ++layer2_1_conv2_rc) {
                  for (int32_t layer2_1_conv2_ry = 0; layer2_1_conv2_ry < 3; ++layer2_1_conv2_ry) {
                    for (int32_t layer2_1_conv2_rx = 0; layer2_1_conv2_rx < 3; ++layer2_1_conv2_rx) {
                      for (int32_t layer2_1_conv2_rb = 0; layer2_1_conv2_rb < 32; ++layer2_1_conv2_rb) {
                        layer2_1_conv2_sum = ((int8_t)(((int64_t)(((layer2_1_conv2_WB[(layer2_1_conv2_rx + (layer2_1_conv2_ry * 3))] ^ w_layer2_1_conv2[((layer2_1_conv2_rx + (layer2_1_conv2_ry * 3)) + (ff10 * 9))]) & (1L << layer2_1_conv2_rb)) >> layer2_1_conv2_rb)) + ((int64_t)layer2_1_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer2_1_conv2_temp;
                layer2_1_conv2_temp = ((int8_t)(288 - ((int32_t)(layer2_1_conv2_sum << 1))));
                write_channel_intel(layer2_1_conv2_pipe_60, layer2_1_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer2_1_bn2[8192];
        for (int32_t n10 = 0; n10 < 1; ++n10) {
      for (int32_t c10 = 0; c10 < 32; ++c10) {
        for (int32_t h10 = 0; h10 < 16; ++h10) {
          for (int32_t w10 = 0; w10 < 16; ++w10) {
            int8_t layer2_1_conv2_temp1;
            layer2_1_conv2_temp1 = read_channel_intel(layer2_1_conv2_pipe_60);
            int32_t layer2_1_bn2_temp;
            layer2_1_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer2_1_conv2_temp1) * ((int40_t)w_layer2_1_bn2_14[c10]))) + ((int41_t)w_layer2_1_bn2_15[c10])));
            write_channel_intel(layer2_1_bn2_pipe_61, layer2_1_bn2_temp);
          }
        }
      }
    }
    int32_t layer2_1_residual2[8192];
        for (int32_t nn40 = 0; nn40 < 1; ++nn40) {
      for (int32_t cc40 = 0; cc40 < 32; ++cc40) {
        for (int32_t ww39 = 0; ww39 < 16; ++ww39) {
          for (int32_t hh40 = 0; hh40 < 16; ++hh40) {
            int32_t layer2_1_rprelu1_temp2;
            layer2_1_rprelu1_temp2 = read_channel_intel(layer2_1_rprelu1_pipe_126);
            int32_t layer2_1_bn2_temp1;
            layer2_1_bn2_temp1 = read_channel_intel(layer2_1_bn2_pipe_61);
            int32_t layer2_1_residual2_temp;
            layer2_1_residual2_temp = ((int32_t)(((int33_t)layer2_1_bn2_temp1) + ((int33_t)layer2_1_rprelu1_temp2)));
            write_channel_intel(layer2_1_residual2_pipe_62, layer2_1_residual2_temp);
          }
        }
      }
    }
    int32_t layer2_1_rprelu2[8192];
            for (int32_t nn41 = 0; nn41 < 1; ++nn41) {
      for (int32_t cc41 = 0; cc41 < 32; ++cc41) {
        for (int32_t ww40 = 0; ww40 < 16; ++ww40) {
          for (int32_t hh41 = 0; hh41 < 16; ++hh41) {
            int32_t layer2_1_residual2_temp1;
            layer2_1_residual2_temp1 = read_channel_intel(layer2_1_residual2_pipe_62);
            int32_t layer2_1_rprelu2_temp;
            layer2_1_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_1_residual2_temp1) + ((int33_t)w_layer2_1_rprelu2_3[cc41])))) ? (((int64_t)(((int33_t)layer2_1_residual2_temp1) + ((int33_t)w_layer2_1_rprelu2_3[cc41])))) : ((int64_t)(((int64_t)w_layer2_1_rprelu2_5[cc41]) * ((int64_t)(((int33_t)layer2_1_residual2_temp1) + ((int33_t)w_layer2_1_rprelu2_3[cc41]))))))) + ((int64_t)w_layer2_1_rprelu2_4[cc41])));
            write_channel_intel(layer2_1_rprelu2_pipe_127, layer2_1_rprelu2_temp);
            write_channel_intel(layer2_1_rprelu2_pipe_63, layer2_1_rprelu2_temp);
          }
        }
      }
    }
    uint32_t layer2_2_rsign1[256];
        for (int32_t nn42 = 0; nn42 < 1; ++nn42) {
      for (int32_t cc42 = 0; cc42 < 1; ++cc42) {
        for (int32_t hh42 = 0; hh42 < 16; ++hh42) {
          for (int32_t ww41 = 0; ww41 < 16; ++ww41) {
            uint32_t layer2_2_rsign1_pack;
            for (int32_t i11 = 0; i11 < 32; ++i11) {
              int32_t layer2_1_rprelu2_temp1;
              layer2_1_rprelu2_temp1 = read_channel_intel(layer2_1_rprelu2_pipe_63);
              layer2_2_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_1_rprelu2_temp1) + ((int33_t)w_layer2_2_rsign1[((cc42 * 32) + i11)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer2_2_rsign1_temp;
            layer2_2_rsign1_temp = layer2_2_rsign1_pack;
            write_channel_intel(layer2_2_rsign1_pipe_64, layer2_2_rsign1_temp);
          }
        }
      }
    }
    uint32_t layer2_2_conv1_pad[324];
        for (int32_t ii11 = 0; ii11 < 1; ++ii11) {
      for (int32_t cc43 = 0; cc43 < 1; ++cc43) {
        for (int32_t hh43 = 0; hh43 < 18; ++hh43) {
          for (int32_t ww42 = 0; ww42 < 18; ++ww42) {
            bool cond = ((((1 <= ww42) && (ww42 < 17)) && (1 <= hh43)) && (hh43 < 17));
            uint32_t layer2_2_rsign1_temp1 = 0;
            if (cond) {
              layer2_2_rsign1_temp1 = read_channel_intel(layer2_2_rsign1_pipe_64);
            }
            uint32_t layer2_2_conv1_pad_temp;
            layer2_2_conv1_pad_temp = (uint32_t)(cond ? ((uint32_t)layer2_2_rsign1_temp1) : ((uint32_t)0U));
            write_channel_intel(layer2_2_conv1_pad_pipe_65, layer2_2_conv1_pad_temp);
            layer2_2_conv1_pad[(((ww42 + (hh43 * 18)) + (cc43 * 324)) + (ii11 * 324))] = layer2_2_conv1_pad_temp;
          }
        }
      }
    }
    int8_t layer2_2_conv1[8192];
    uint32_t layer2_2_conv1_LB[54];
    uint32_t layer2_2_conv1_WB[9];
        for (int32_t nn43 = 0; nn43 < 1; ++nn43) {
      for (int32_t yy_reuse10 = 0; yy_reuse10 < 18; ++yy_reuse10) {
        for (int32_t xx_reuse10 = 0; xx_reuse10 < 18; ++xx_reuse10) {
          for (int32_t layer2_2_conv1_pad_1 = 0; layer2_2_conv1_pad_1 < 2; ++layer2_2_conv1_pad_1) {
            layer2_2_conv1_LB[(xx_reuse10 + (layer2_2_conv1_pad_1 * 18))] = layer2_2_conv1_LB[((xx_reuse10 + (layer2_2_conv1_pad_1 * 18)) + 18)];
          }
          uint32_t layer2_2_conv1_pad_temp1;
          layer2_2_conv1_pad_temp1 = read_channel_intel(layer2_2_conv1_pad_pipe_65);
          layer2_2_conv1_LB[(xx_reuse10 + 36)] = layer2_2_conv1_pad_temp1;
          if (2 <= yy_reuse10) {
            for (int32_t layer2_2_conv1_LB_1 = 0; layer2_2_conv1_LB_1 < 3; ++layer2_2_conv1_LB_1) {
              for (int32_t layer2_2_conv1_LB_0 = 0; layer2_2_conv1_LB_0 < 2; ++layer2_2_conv1_LB_0) {
                layer2_2_conv1_WB[(layer2_2_conv1_LB_0 + (layer2_2_conv1_LB_1 * 3))] = layer2_2_conv1_WB[((layer2_2_conv1_LB_0 + (layer2_2_conv1_LB_1 * 3)) + 1)];
              }
              layer2_2_conv1_WB[((layer2_2_conv1_LB_1 * 3) + 2)] = layer2_2_conv1_LB[(xx_reuse10 + (layer2_2_conv1_LB_1 * 18))];
            }
            for (int32_t ff11 = 0; ff11 < 32; ++ff11) {
              if (2 <= xx_reuse10) {
                int8_t layer2_2_conv1_sum;
                for (int32_t layer2_2_conv1_rc = 0; layer2_2_conv1_rc < 1; ++layer2_2_conv1_rc) {
                  for (int32_t layer2_2_conv1_ry = 0; layer2_2_conv1_ry < 3; ++layer2_2_conv1_ry) {
                    for (int32_t layer2_2_conv1_rx = 0; layer2_2_conv1_rx < 3; ++layer2_2_conv1_rx) {
                      for (int32_t layer2_2_conv1_rb = 0; layer2_2_conv1_rb < 32; ++layer2_2_conv1_rb) {
                        layer2_2_conv1_sum = ((int8_t)(((int64_t)(((layer2_2_conv1_WB[(layer2_2_conv1_rx + (layer2_2_conv1_ry * 3))] ^ w_layer2_2_conv1[((layer2_2_conv1_rx + (layer2_2_conv1_ry * 3)) + (ff11 * 9))]) & (1L << layer2_2_conv1_rb)) >> layer2_2_conv1_rb)) + ((int64_t)layer2_2_conv1_sum)));
                      }
                    }
                  }
                }
                int8_t layer2_2_conv1_temp;
                layer2_2_conv1_temp = ((int8_t)(288 - ((int32_t)(layer2_2_conv1_sum << 1))));
                write_channel_intel(layer2_2_conv1_pipe_66, layer2_2_conv1_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer2_2_bn1[8192];
        for (int32_t n11 = 0; n11 < 1; ++n11) {
      for (int32_t c11 = 0; c11 < 32; ++c11) {
        for (int32_t h11 = 0; h11 < 16; ++h11) {
          for (int32_t w11 = 0; w11 < 16; ++w11) {
            int8_t layer2_2_conv1_temp1;
            layer2_2_conv1_temp1 = read_channel_intel(layer2_2_conv1_pipe_66);
            int32_t layer2_2_bn1_temp;
            layer2_2_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer2_2_conv1_temp1) * ((int40_t)w_layer2_2_bn1_9[c11]))) + ((int41_t)w_layer2_2_bn1_10[c11])));
            write_channel_intel(layer2_2_bn1_pipe_67, layer2_2_bn1_temp);
          }
        }
      }
    }
    int32_t layer2_2_residual1[8192];
        for (int32_t nn44 = 0; nn44 < 1; ++nn44) {
      for (int32_t cc44 = 0; cc44 < 32; ++cc44) {
        for (int32_t ww43 = 0; ww43 < 16; ++ww43) {
          for (int32_t hh44 = 0; hh44 < 16; ++hh44) {
            int32_t layer2_2_bn1_temp1;
            layer2_2_bn1_temp1 = read_channel_intel(layer2_2_bn1_pipe_67);
            int32_t layer2_1_rprelu2_temp2;
            layer2_1_rprelu2_temp2 = read_channel_intel(layer2_1_rprelu2_pipe_127);
            int32_t layer2_2_residual1_temp;
            layer2_2_residual1_temp = ((int32_t)(((int33_t)layer2_2_bn1_temp1) + ((int33_t)layer2_1_rprelu2_temp2)));
            write_channel_intel(layer2_2_residual1_pipe_68, layer2_2_residual1_temp);
          }
        }
      }
    }
    int32_t layer2_2_rprelu1[8192];
            for (int32_t nn45 = 0; nn45 < 1; ++nn45) {
      for (int32_t cc45 = 0; cc45 < 32; ++cc45) {
        for (int32_t ww44 = 0; ww44 < 16; ++ww44) {
          for (int32_t hh45 = 0; hh45 < 16; ++hh45) {
            int32_t layer2_2_residual1_temp1;
            layer2_2_residual1_temp1 = read_channel_intel(layer2_2_residual1_pipe_68);
            int32_t layer2_2_rprelu1_temp;
            layer2_2_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_2_residual1_temp1) + ((int33_t)w_layer2_2_rprelu1_0[cc45])))) ? (((int64_t)(((int33_t)layer2_2_residual1_temp1) + ((int33_t)w_layer2_2_rprelu1_0[cc45])))) : ((int64_t)(((int64_t)w_layer2_2_rprelu1_2[cc45]) * ((int64_t)(((int33_t)layer2_2_residual1_temp1) + ((int33_t)w_layer2_2_rprelu1_0[cc45]))))))) + ((int64_t)w_layer2_2_rprelu1_1[cc45])));
            write_channel_intel(layer2_2_rprelu1_pipe_128, layer2_2_rprelu1_temp);
            write_channel_intel(layer2_2_rprelu1_pipe_69, layer2_2_rprelu1_temp);
          }
        }
      }
    }
    uint32_t layer2_2_rsign2[256];
        for (int32_t nn46 = 0; nn46 < 1; ++nn46) {
      for (int32_t cc46 = 0; cc46 < 1; ++cc46) {
        for (int32_t hh46 = 0; hh46 < 16; ++hh46) {
          for (int32_t ww45 = 0; ww45 < 16; ++ww45) {
            uint32_t layer2_2_rsign2_pack;
            for (int32_t i12 = 0; i12 < 32; ++i12) {
              int32_t layer2_2_rprelu1_temp1;
              layer2_2_rprelu1_temp1 = read_channel_intel(layer2_2_rprelu1_pipe_69);
              layer2_2_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_2_rprelu1_temp1) + ((int33_t)w_layer2_2_rsign2[((cc46 * 32) + i12)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer2_2_rsign2_temp;
            layer2_2_rsign2_temp = layer2_2_rsign2_pack;
            write_channel_intel(layer2_2_rsign2_pipe_70, layer2_2_rsign2_temp);
          }
        }
      }
    }
    uint32_t layer2_2_conv2_pad[324];
        for (int32_t ii12 = 0; ii12 < 1; ++ii12) {
      for (int32_t cc47 = 0; cc47 < 1; ++cc47) {
        for (int32_t hh47 = 0; hh47 < 18; ++hh47) {
          for (int32_t ww46 = 0; ww46 < 18; ++ww46) {
            bool cond = ((((1 <= ww46) && (ww46 < 17)) && (1 <= hh47)) && (hh47 < 17));
            uint32_t layer2_2_rsign2_temp1 = 0;
            if (cond) {
              layer2_2_rsign2_temp1 = read_channel_intel(layer2_2_rsign2_pipe_70);
            }
            uint32_t layer2_2_conv2_pad_temp;
            layer2_2_conv2_pad_temp = (uint32_t)(cond ? ((uint32_t)layer2_2_rsign2_temp1) : ((uint32_t)0U));
            write_channel_intel(layer2_2_conv2_pad_pipe_71, layer2_2_conv2_pad_temp);
            layer2_2_conv2_pad[(((ww46 + (hh47 * 18)) + (cc47 * 324)) + (ii12 * 324))] = layer2_2_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer2_2_conv2[8192];
    uint32_t layer2_2_conv2_LB[54];
    uint32_t layer2_2_conv2_WB[9];
        for (int32_t nn47 = 0; nn47 < 1; ++nn47) {
      for (int32_t yy_reuse11 = 0; yy_reuse11 < 18; ++yy_reuse11) {
        for (int32_t xx_reuse11 = 0; xx_reuse11 < 18; ++xx_reuse11) {
          for (int32_t layer2_2_conv2_pad_1 = 0; layer2_2_conv2_pad_1 < 2; ++layer2_2_conv2_pad_1) {
            layer2_2_conv2_LB[(xx_reuse11 + (layer2_2_conv2_pad_1 * 18))] = layer2_2_conv2_LB[((xx_reuse11 + (layer2_2_conv2_pad_1 * 18)) + 18)];
          }
          uint32_t layer2_2_conv2_pad_temp1;
          layer2_2_conv2_pad_temp1 = read_channel_intel(layer2_2_conv2_pad_pipe_71);
          layer2_2_conv2_LB[(xx_reuse11 + 36)] = layer2_2_conv2_pad_temp1;
          if (2 <= yy_reuse11) {
            for (int32_t layer2_2_conv2_LB_1 = 0; layer2_2_conv2_LB_1 < 3; ++layer2_2_conv2_LB_1) {
              for (int32_t layer2_2_conv2_LB_0 = 0; layer2_2_conv2_LB_0 < 2; ++layer2_2_conv2_LB_0) {
                layer2_2_conv2_WB[(layer2_2_conv2_LB_0 + (layer2_2_conv2_LB_1 * 3))] = layer2_2_conv2_WB[((layer2_2_conv2_LB_0 + (layer2_2_conv2_LB_1 * 3)) + 1)];
              }
              layer2_2_conv2_WB[((layer2_2_conv2_LB_1 * 3) + 2)] = layer2_2_conv2_LB[(xx_reuse11 + (layer2_2_conv2_LB_1 * 18))];
            }
            for (int32_t ff12 = 0; ff12 < 32; ++ff12) {
              if (2 <= xx_reuse11) {
                int8_t layer2_2_conv2_sum;
                for (int32_t layer2_2_conv2_rc = 0; layer2_2_conv2_rc < 1; ++layer2_2_conv2_rc) {
                  for (int32_t layer2_2_conv2_ry = 0; layer2_2_conv2_ry < 3; ++layer2_2_conv2_ry) {
                    for (int32_t layer2_2_conv2_rx = 0; layer2_2_conv2_rx < 3; ++layer2_2_conv2_rx) {
                      for (int32_t layer2_2_conv2_rb = 0; layer2_2_conv2_rb < 32; ++layer2_2_conv2_rb) {
                        layer2_2_conv2_sum = ((int8_t)(((int64_t)(((layer2_2_conv2_WB[(layer2_2_conv2_rx + (layer2_2_conv2_ry * 3))] ^ w_layer2_2_conv2[((layer2_2_conv2_rx + (layer2_2_conv2_ry * 3)) + (ff12 * 9))]) & (1L << layer2_2_conv2_rb)) >> layer2_2_conv2_rb)) + ((int64_t)layer2_2_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer2_2_conv2_temp;
                layer2_2_conv2_temp = ((int8_t)(288 - ((int32_t)(layer2_2_conv2_sum << 1))));
                write_channel_intel(layer2_2_conv2_pipe_72, layer2_2_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer2_2_bn2[8192];
        for (int32_t n12 = 0; n12 < 1; ++n12) {
      for (int32_t c12 = 0; c12 < 32; ++c12) {
        for (int32_t h12 = 0; h12 < 16; ++h12) {
          for (int32_t w12 = 0; w12 < 16; ++w12) {
            int8_t layer2_2_conv2_temp1;
            layer2_2_conv2_temp1 = read_channel_intel(layer2_2_conv2_pipe_72);
            int32_t layer2_2_bn2_temp;
            layer2_2_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer2_2_conv2_temp1) * ((int40_t)w_layer2_2_bn2_14[c12]))) + ((int41_t)w_layer2_2_bn2_15[c12])));
            write_channel_intel(layer2_2_bn2_pipe_73, layer2_2_bn2_temp);
          }
        }
      }
    }
    int32_t layer2_2_residual2[8192];
        for (int32_t nn48 = 0; nn48 < 1; ++nn48) {
      for (int32_t cc48 = 0; cc48 < 32; ++cc48) {
        for (int32_t ww47 = 0; ww47 < 16; ++ww47) {
          for (int32_t hh48 = 0; hh48 < 16; ++hh48) {
            int32_t layer2_2_bn2_temp1;
            layer2_2_bn2_temp1 = read_channel_intel(layer2_2_bn2_pipe_73);
            int32_t layer2_2_rprelu1_temp2;
            layer2_2_rprelu1_temp2 = read_channel_intel(layer2_2_rprelu1_pipe_128);
            int32_t layer2_2_residual2_temp;
            layer2_2_residual2_temp = ((int32_t)(((int33_t)layer2_2_bn2_temp1) + ((int33_t)layer2_2_rprelu1_temp2)));
            write_channel_intel(layer2_2_residual2_pipe_74, layer2_2_residual2_temp);
          }
        }
      }
    }
    int32_t layer2_2_rprelu2[8192];
            for (int32_t nn49 = 0; nn49 < 1; ++nn49) {
      for (int32_t cc49 = 0; cc49 < 32; ++cc49) {
        for (int32_t ww48 = 0; ww48 < 16; ++ww48) {
          for (int32_t hh49 = 0; hh49 < 16; ++hh49) {
            int32_t layer2_2_residual2_temp1;
            layer2_2_residual2_temp1 = read_channel_intel(layer2_2_residual2_pipe_74);
            int32_t layer2_2_rprelu2_temp;
            layer2_2_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_2_residual2_temp1) + ((int33_t)w_layer2_2_rprelu2_3[cc49])))) ? (((int64_t)(((int33_t)layer2_2_residual2_temp1) + ((int33_t)w_layer2_2_rprelu2_3[cc49])))) : ((int64_t)(((int64_t)w_layer2_2_rprelu2_5[cc49]) * ((int64_t)(((int33_t)layer2_2_residual2_temp1) + ((int33_t)w_layer2_2_rprelu2_3[cc49]))))))) + ((int64_t)w_layer2_2_rprelu2_4[cc49])));
            write_channel_intel(layer2_2_rprelu2_pipe_129, layer2_2_rprelu2_temp);
            write_channel_intel(layer2_2_rprelu2_pipe_75, layer2_2_rprelu2_temp);
          }
        }
      }
    }
    uint32_t layer3_0_rsign1[256];
        for (int32_t nn50 = 0; nn50 < 1; ++nn50) {
      for (int32_t cc50 = 0; cc50 < 1; ++cc50) {
        for (int32_t hh50 = 0; hh50 < 16; ++hh50) {
          for (int32_t ww49 = 0; ww49 < 16; ++ww49) {
            uint32_t layer3_0_rsign1_pack;
            for (int32_t i13 = 0; i13 < 32; ++i13) {
              int32_t layer2_2_rprelu2_temp1;
              layer2_2_rprelu2_temp1 = read_channel_intel(layer2_2_rprelu2_pipe_75);
              layer3_0_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer2_2_rprelu2_temp1) + ((int33_t)w_layer3_0_rsign1[((cc50 * 32) + i13)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer3_0_rsign1_temp;
            layer3_0_rsign1_temp = layer3_0_rsign1_pack;
            write_channel_intel(layer3_0_rsign1_pipe_76, layer3_0_rsign1_temp);
          }
        }
      }
    }
    uint32_t layer3_0_conv1_pad[324];
        for (int32_t ii13 = 0; ii13 < 1; ++ii13) {
      for (int32_t cc51 = 0; cc51 < 1; ++cc51) {
        for (int32_t hh51 = 0; hh51 < 18; ++hh51) {
          for (int32_t ww50 = 0; ww50 < 18; ++ww50) {
            bool cond = ((((1 <= ww50) && (ww50 < 17)) && (1 <= hh51)) && (hh51 < 17));
            uint32_t layer3_0_rsign1_temp1 = 0;
            if (cond) {
              layer3_0_rsign1_temp1 = read_channel_intel(layer3_0_rsign1_pipe_76);
            }
            uint32_t layer3_0_conv1_pad_temp;
            layer3_0_conv1_pad_temp = (uint32_t)(cond ? ((uint32_t)layer3_0_rsign1_temp1) : ((uint32_t)0U));
            write_channel_intel(layer3_0_conv1_pad_pipe_77, layer3_0_conv1_pad_temp);
          }
        }
      }
    }

    int32_t layer3_0_conv1_LB[1][1][3][18];
    int32_t layer3_0_conv1_WB[1][1][3][3];
      for (int yy_reuse = 0; yy_reuse < 18; ++yy_reuse) {
        for (int xx_reuse = 0; xx_reuse < 18; ++xx_reuse) {
          for (int layer3_0_conv1_pad_1 = 0; layer3_0_conv1_pad_1 < 2; ++layer3_0_conv1_pad_1) {
            layer3_0_conv1_LB[0][0][layer3_0_conv1_pad_1][xx_reuse] = layer3_0_conv1_LB[0][0][(layer3_0_conv1_pad_1 + 1)][xx_reuse];
          }
          int32_t layer3_0_conv1_pad_temp1 = read_channel_intel(layer3_0_conv1_pad_pipe_77);
          layer3_0_conv1_LB[0][0][2][xx_reuse] = layer3_0_conv1_pad_temp1;
          if (2 <= yy_reuse && (yy_reuse - 2) % 2 == 0) {
            for (int layer3_0_conv1_LB_1 = 0; layer3_0_conv1_LB_1 < 3; ++layer3_0_conv1_LB_1) {
              for (int layer3_0_conv1_LB_0 = 0; layer3_0_conv1_LB_0 < 2; ++layer3_0_conv1_LB_0) {
                layer3_0_conv1_WB[0][0][layer3_0_conv1_LB_1][layer3_0_conv1_LB_0] = layer3_0_conv1_WB[0][0][layer3_0_conv1_LB_1][(layer3_0_conv1_LB_0 + 1)];
              }
              layer3_0_conv1_WB[0][0][layer3_0_conv1_LB_1][2] = layer3_0_conv1_LB[0][0][layer3_0_conv1_LB_1][xx_reuse];
            }
            if (2 <= xx_reuse && (xx_reuse - 2) % 2 == 0) {
    for (int ff = 0; ff < 64; ++ff) {
              int16_t layer3_0_conv1_sum;
              layer3_0_conv1_sum = 0;
              for (int layer3_0_conv1_ry = 0; layer3_0_conv1_ry < 3; ++layer3_0_conv1_ry) {
                for (int layer3_0_conv1_rx = 0; layer3_0_conv1_rx < 3; ++layer3_0_conv1_rx) {
                  for (int layer3_0_conv1_rb = 0; layer3_0_conv1_rb < 32; ++layer3_0_conv1_rb) {
                    layer3_0_conv1_sum = ((((layer3_0_conv1_WB[0][0][layer3_0_conv1_ry][layer3_0_conv1_rx] ^ w_layer3_0_conv1[ff][0][layer3_0_conv1_ry][layer3_0_conv1_rx]) & (1 << layer3_0_conv1_rb)) + (layer3_0_conv1_sum)));
                  }
                }
              }
              int8_t layer3_0_conv1_temp;
              layer3_0_conv1_temp = ((int8_t)(288 - ((int32_t)(layer3_0_conv1_sum << 1))));
              write_channel_intel(layer3_0_conv1_pipe_78, layer3_0_conv1_temp);
            }
          }
        }
      }
    }

    int32_t layer3_0_bn1[4096];
    for (int32_t n13 = 0; n13 < 1; ++n13) {
      for (int32_t c13 = 0; c13 < 64; ++c13) {
        for (int32_t h13 = 0; h13 < 8; ++h13) {
          for (int32_t w13 = 0; w13 < 8; ++w13) {
            int8_t layer3_0_conv1_temp1;
            layer3_0_conv1_temp1 = read_channel_intel(layer3_0_conv1_pipe_78);
            int32_t layer3_0_bn1_temp;
            layer3_0_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer3_0_conv1_temp1) * ((int40_t)w_layer3_0_bn1_9[c13]))) + ((int41_t)w_layer3_0_bn1_10[c13])));
            write_channel_intel(layer3_0_bn1_pipe_79, layer3_0_bn1_temp);
          }
        }
      }
    }
    int32_t layer3_0_avgpool_res[2048];
    int32_t layer3_0_avgpool_LB[32];
    int32_t layer3_0_avgpool;
        for (int32_t ii14 = 0; ii14 < 1; ++ii14) {
      for (int32_t cc52 = 0; cc52 < 32; ++cc52) {
        for (int32_t hh52 = 0; hh52 < 8; ++hh52) {
          for (int32_t layer3_0_avgpool_LB_i = 0; layer3_0_avgpool_LB_i < 2; ++layer3_0_avgpool_LB_i) {
            for (int32_t layer3_0_avgpool_LB_j = 0; layer3_0_avgpool_LB_j < 16; ++layer3_0_avgpool_LB_j) {
              int32_t layer2_2_rprelu2_temp2;
              layer2_2_rprelu2_temp2 = read_channel_intel(layer2_2_rprelu2_pipe_129);
              layer3_0_avgpool_LB[(layer3_0_avgpool_LB_j + (layer3_0_avgpool_LB_i * 16))] = layer2_2_rprelu2_temp2;
            }
          }
          for (int32_t layer3_0_avgpool_ww = 0; layer3_0_avgpool_ww < 8; ++layer3_0_avgpool_ww) {
            int32_t layer3_0_avgpool_val;
            for (int32_t layer3_0_avgpool_ry = 0; layer3_0_avgpool_ry < 2; ++layer3_0_avgpool_ry) {
              for (int32_t layer3_0_avgpool_rx = 0; layer3_0_avgpool_rx < 2; ++layer3_0_avgpool_rx) {
                layer3_0_avgpool_val = ((int32_t)(((int33_t)layer3_0_avgpool_val) + ((int33_t)layer3_0_avgpool_LB[(((layer3_0_avgpool_ww * 2) + layer3_0_avgpool_rx) + (layer3_0_avgpool_ry * 16))])));
              }
            }
            int32_t layer3_0_avgpool_res_temp;
            layer3_0_avgpool_res_temp = ((int32_t)(((int64_t)layer3_0_avgpool_val) / (int64_t)4));
            write_channel_intel(layer3_0_avgpool_res_pipe_130, layer3_0_avgpool_res_temp);
            layer3_0_avgpool_res[(((layer3_0_avgpool_ww + (hh52 * 8)) + (cc52 * 64)) + (ii14 * 2048))] = layer3_0_avgpool_res_temp;
          }
        }
      }
    }
    int32_t layer3_0_concat[4096];
        for (int32_t nn52 = 0; nn52 < 1; ++nn52) {
      for (int32_t cc53 = 0; cc53 < 64; ++cc53) {
        for (int32_t ww51 = 0; ww51 < 8; ++ww51) {
          for (int32_t hh53 = 0; hh53 < 8; ++hh53) {
            int32_t layer3_0_avgpool_res_temp1;
            layer3_0_avgpool_res_temp1 = read_channel_intel(layer3_0_avgpool_res_pipe_130);
            int32_t layer3_0_concat_temp;
            layer3_0_concat_temp = layer3_0_avgpool_res_temp1;
            write_channel_intel(layer3_0_concat_pipe_131, layer3_0_concat_temp);
          }
        }
      }
    }
    int32_t layer3_0_residual1[4096];
        for (int32_t nn53 = 0; nn53 < 1; ++nn53) {
      for (int32_t cc54 = 0; cc54 < 64; ++cc54) {
        for (int32_t ww52 = 0; ww52 < 8; ++ww52) {
          for (int32_t hh54 = 0; hh54 < 8; ++hh54) {
            int32_t layer3_0_bn1_temp1;
            layer3_0_bn1_temp1 = read_channel_intel(layer3_0_bn1_pipe_79);
            int32_t layer3_0_concat_temp1;
            layer3_0_concat_temp1 = read_channel_intel(layer3_0_concat_pipe_131);
            int32_t layer3_0_residual1_temp;
            layer3_0_residual1_temp = ((int32_t)(((int33_t)layer3_0_bn1_temp1) + ((int33_t)layer3_0_concat_temp1)));
            write_channel_intel(layer3_0_residual1_pipe_80, layer3_0_residual1_temp);
          }
        }
      }
    }
    int32_t layer3_0_rprelu1[4096];
            for (int32_t nn54 = 0; nn54 < 1; ++nn54) {
      for (int32_t cc55 = 0; cc55 < 64; ++cc55) {
        for (int32_t ww53 = 0; ww53 < 8; ++ww53) {
          for (int32_t hh55 = 0; hh55 < 8; ++hh55) {
            int32_t layer3_0_residual1_temp1;
            layer3_0_residual1_temp1 = read_channel_intel(layer3_0_residual1_pipe_80);
            int32_t layer3_0_rprelu1_temp;
            layer3_0_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_0_residual1_temp1) + ((int33_t)w_layer3_0_rprelu1_0[cc55])))) ? (((int64_t)(((int33_t)layer3_0_residual1_temp1) + ((int33_t)w_layer3_0_rprelu1_0[cc55])))) : ((int64_t)(((int64_t)w_layer3_0_rprelu1_2[cc55]) * ((int64_t)(((int33_t)layer3_0_residual1_temp1) + ((int33_t)w_layer3_0_rprelu1_0[cc55]))))))) + ((int64_t)w_layer3_0_rprelu1_1[cc55])));
            write_channel_intel(layer3_0_rprelu1_pipe_132, layer3_0_rprelu1_temp);
            write_channel_intel(layer3_0_rprelu1_pipe_81, layer3_0_rprelu1_temp);
          }
        }
      }
    }
    uint32_t layer3_0_rsign2[128];
        for (int32_t nn55 = 0; nn55 < 1; ++nn55) {
      for (int32_t cc56 = 0; cc56 < 2; ++cc56) {
        for (int32_t hh56 = 0; hh56 < 8; ++hh56) {
          for (int32_t ww54 = 0; ww54 < 8; ++ww54) {
            uint32_t layer3_0_rsign2_pack;
            for (int32_t i14 = 0; i14 < 32; ++i14) {
              int32_t layer3_0_rprelu1_temp1;
              layer3_0_rprelu1_temp1 = read_channel_intel(layer3_0_rprelu1_pipe_81);
              layer3_0_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_0_rprelu1_temp1) + ((int33_t)w_layer3_0_rsign2[((cc56 * 32) + i14)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer3_0_rsign2_temp;
            layer3_0_rsign2_temp = layer3_0_rsign2_pack;
            write_channel_intel(layer3_0_rsign2_pipe_82, layer3_0_rsign2_temp);
          }
        }
      }
    }
    uint32_t layer3_0_conv2_pad[200];
        for (int32_t ii15 = 0; ii15 < 1; ++ii15) {
      for (int32_t cc57 = 0; cc57 < 2; ++cc57) {
        for (int32_t hh57 = 0; hh57 < 10; ++hh57) {
          for (int32_t ww55 = 0; ww55 < 10; ++ww55) {
            bool cond = ((((1 <= ww55) && (ww55 < 9)) && (1 <= hh57)) && (hh57 < 9));
            uint32_t layer3_0_rsign2_temp1 = 0;
            if (cond) {
              layer3_0_rsign2_temp1 = read_channel_intel(layer3_0_rsign2_pipe_82);
            }
            uint32_t layer3_0_conv2_pad_temp;
            layer3_0_conv2_pad_temp = (uint32_t)(cond ? ((uint32_t)layer3_0_rsign2_temp1) : ((uint32_t)0U));
            write_channel_intel(layer3_0_conv2_pad_pipe_83, layer3_0_conv2_pad_temp);
            layer3_0_conv2_pad[(((ww55 + (hh57 * 10)) + (cc57 * 100)) + (ii15 * 200))] = layer3_0_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer3_0_conv2[4096];
    uint32_t layer3_0_conv2_LB[60];
    uint32_t layer3_0_conv2_WB[18];
        for (int32_t nn56 = 0; nn56 < 1; ++nn56) {
      for (int32_t yy_reuse12 = 0; yy_reuse12 < 10; ++yy_reuse12) {
        for (int32_t xx_reuse12 = 0; xx_reuse12 < 10; ++xx_reuse12) {
          for (int32_t layer3_0_conv2_pad_2 = 0; layer3_0_conv2_pad_2 < 2; ++layer3_0_conv2_pad_2) {
            for (int32_t layer3_0_conv2_pad_1 = 0; layer3_0_conv2_pad_1 < 2; ++layer3_0_conv2_pad_1) {
              layer3_0_conv2_LB[((xx_reuse12 + (layer3_0_conv2_pad_1 * 10)) + (layer3_0_conv2_pad_2 * 30))] = layer3_0_conv2_LB[(((xx_reuse12 + (layer3_0_conv2_pad_1 * 10)) + (layer3_0_conv2_pad_2 * 30)) + 10)];
            }
            uint32_t layer3_0_conv2_pad_temp1;
            layer3_0_conv2_pad_temp1 = read_channel_intel(layer3_0_conv2_pad_pipe_83);
            layer3_0_conv2_LB[((xx_reuse12 + (layer3_0_conv2_pad_2 * 30)) + 20)] = layer3_0_conv2_pad_temp1;
          }
          if (2 <= yy_reuse12) {
            for (int32_t layer3_0_conv2_LB_1 = 0; layer3_0_conv2_LB_1 < 3; ++layer3_0_conv2_LB_1) {
              for (int32_t layer3_0_conv2_LB_2 = 0; layer3_0_conv2_LB_2 < 2; ++layer3_0_conv2_LB_2) {
                for (int32_t layer3_0_conv2_LB_0 = 0; layer3_0_conv2_LB_0 < 2; ++layer3_0_conv2_LB_0) {
                  layer3_0_conv2_WB[((layer3_0_conv2_LB_0 + (layer3_0_conv2_LB_1 * 3)) + (layer3_0_conv2_LB_2 * 9))] = layer3_0_conv2_WB[(((layer3_0_conv2_LB_0 + (layer3_0_conv2_LB_1 * 3)) + (layer3_0_conv2_LB_2 * 9)) + 1)];
                }
                layer3_0_conv2_WB[(((layer3_0_conv2_LB_1 + (layer3_0_conv2_LB_2 * 3)) * 3) + 2)] = layer3_0_conv2_LB[((xx_reuse12 + (layer3_0_conv2_LB_1 * 10)) + (layer3_0_conv2_LB_2 * 30))];
              }
            }
            for (int32_t ff14 = 0; ff14 < 64; ++ff14) {
              if (2 <= xx_reuse12) {
                int8_t layer3_0_conv2_sum;
                for (int32_t layer3_0_conv2_rc = 0; layer3_0_conv2_rc < 2; ++layer3_0_conv2_rc) {
                  for (int32_t layer3_0_conv2_ry = 0; layer3_0_conv2_ry < 3; ++layer3_0_conv2_ry) {
                    for (int32_t layer3_0_conv2_rx = 0; layer3_0_conv2_rx < 3; ++layer3_0_conv2_rx) {
                      for (int32_t layer3_0_conv2_rb = 0; layer3_0_conv2_rb < 32; ++layer3_0_conv2_rb) {
                        layer3_0_conv2_sum = ((int8_t)(((int64_t)(((layer3_0_conv2_WB[((layer3_0_conv2_rx + (layer3_0_conv2_ry * 3)) + (layer3_0_conv2_rc * 9))] ^ w_layer3_0_conv2[(((layer3_0_conv2_rx + (layer3_0_conv2_ry * 3)) + (layer3_0_conv2_rc * 9)) + (ff14 * 18))]) & (1L << layer3_0_conv2_rb)) >> layer3_0_conv2_rb)) + ((int64_t)layer3_0_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer3_0_conv2_temp;
                layer3_0_conv2_temp = ((int8_t)(576 - ((int32_t)(layer3_0_conv2_sum << 1))));
                write_channel_intel(layer3_0_conv2_pipe_84, layer3_0_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer3_0_bn2[4096];
        for (int32_t n14 = 0; n14 < 1; ++n14) {
      for (int32_t c14 = 0; c14 < 64; ++c14) {
        for (int32_t h14 = 0; h14 < 8; ++h14) {
          for (int32_t w14 = 0; w14 < 8; ++w14) {
            int8_t layer3_0_conv2_temp1;
            layer3_0_conv2_temp1 = read_channel_intel(layer3_0_conv2_pipe_84);
            int32_t layer3_0_bn2_temp;
            layer3_0_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer3_0_conv2_temp1) * ((int40_t)w_layer3_0_bn2_14[c14]))) + ((int41_t)w_layer3_0_bn2_15[c14])));
            write_channel_intel(layer3_0_bn2_pipe_85, layer3_0_bn2_temp);
          }
        }
      }
    }
    int32_t layer3_0_residual2[4096];
        for (int32_t nn57 = 0; nn57 < 1; ++nn57) {
      for (int32_t cc58 = 0; cc58 < 64; ++cc58) {
        for (int32_t ww56 = 0; ww56 < 8; ++ww56) {
          for (int32_t hh58 = 0; hh58 < 8; ++hh58) {
            int32_t layer3_0_bn2_temp1;
            layer3_0_bn2_temp1 = read_channel_intel(layer3_0_bn2_pipe_85);
            int32_t layer3_0_rprelu1_temp2;
            layer3_0_rprelu1_temp2 = read_channel_intel(layer3_0_rprelu1_pipe_132);
            int32_t layer3_0_residual2_temp;
            layer3_0_residual2_temp = ((int32_t)(((int33_t)layer3_0_bn2_temp1) + ((int33_t)layer3_0_rprelu1_temp2)));
            write_channel_intel(layer3_0_residual2_pipe_86, layer3_0_residual2_temp);
          }
        }
      }
    }
    int32_t layer3_0_rprelu2[4096];
            for (int32_t nn58 = 0; nn58 < 1; ++nn58) {
      for (int32_t cc59 = 0; cc59 < 64; ++cc59) {
        for (int32_t ww57 = 0; ww57 < 8; ++ww57) {
          for (int32_t hh59 = 0; hh59 < 8; ++hh59) {
            int32_t layer3_0_residual2_temp1;
            layer3_0_residual2_temp1 = read_channel_intel(layer3_0_residual2_pipe_86);
            int32_t layer3_0_rprelu2_temp;
            layer3_0_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_0_residual2_temp1) + ((int33_t)w_layer3_0_rprelu2_3[cc59])))) ? (((int64_t)(((int33_t)layer3_0_residual2_temp1) + ((int33_t)w_layer3_0_rprelu2_3[cc59])))) : ((int64_t)(((int64_t)w_layer3_0_rprelu2_5[cc59]) * ((int64_t)(((int33_t)layer3_0_residual2_temp1) + ((int33_t)w_layer3_0_rprelu2_3[cc59]))))))) + ((int64_t)w_layer3_0_rprelu2_4[cc59])));
            write_channel_intel(layer3_0_rprelu2_pipe_133, layer3_0_rprelu2_temp);
            write_channel_intel(layer3_0_rprelu2_pipe_87, layer3_0_rprelu2_temp);
          }
        }
      }
    }
    uint32_t layer3_1_rsign1[128];
        for (int32_t nn59 = 0; nn59 < 1; ++nn59) {
      for (int32_t cc60 = 0; cc60 < 2; ++cc60) {
        for (int32_t hh60 = 0; hh60 < 8; ++hh60) {
          for (int32_t ww58 = 0; ww58 < 8; ++ww58) {
            uint32_t layer3_1_rsign1_pack;
            for (int32_t i15 = 0; i15 < 32; ++i15) {
              int32_t layer3_0_rprelu2_temp1;
              layer3_0_rprelu2_temp1 = read_channel_intel(layer3_0_rprelu2_pipe_87);
              layer3_1_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_0_rprelu2_temp1) + ((int33_t)w_layer3_1_rsign1[((cc60 * 32) + i15)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer3_1_rsign1_temp;
            layer3_1_rsign1_temp = layer3_1_rsign1_pack;
            write_channel_intel(layer3_1_rsign1_pipe_88, layer3_1_rsign1_temp);
          }
        }
      }
    }
    uint32_t layer3_1_conv1_pad[200];
        for (int32_t ii16 = 0; ii16 < 1; ++ii16) {
      for (int32_t cc61 = 0; cc61 < 2; ++cc61) {
        for (int32_t hh61 = 0; hh61 < 10; ++hh61) {
          for (int32_t ww59 = 0; ww59 < 10; ++ww59) {
            bool cond = ((((1 <= ww59) && (ww59 < 9)) && (1 <= hh61)) && (hh61 < 9));
            uint32_t layer3_1_rsign1_temp1 = 0;
            if (cond) {
              layer3_1_rsign1_temp1 = read_channel_intel(layer3_1_rsign1_pipe_88);
            }
            uint32_t layer3_1_conv1_pad_temp;
            layer3_1_conv1_pad_temp = (uint32_t)(cond ? ((uint32_t)layer3_1_rsign1_temp1) : ((uint32_t)0U));
            write_channel_intel(layer3_1_conv1_pad_pipe_89, layer3_1_conv1_pad_temp);
            layer3_1_conv1_pad[(((ww59 + (hh61 * 10)) + (cc61 * 100)) + (ii16 * 200))] = layer3_1_conv1_pad_temp;
          }
        }
      }
    }
    int8_t layer3_1_conv1[4096];
    uint32_t layer3_1_conv1_LB[60];
    uint32_t layer3_1_conv1_WB[18];
        for (int32_t nn60 = 0; nn60 < 1; ++nn60) {
      for (int32_t yy_reuse13 = 0; yy_reuse13 < 10; ++yy_reuse13) {
        for (int32_t xx_reuse13 = 0; xx_reuse13 < 10; ++xx_reuse13) {
          for (int32_t layer3_1_conv1_pad_2 = 0; layer3_1_conv1_pad_2 < 2; ++layer3_1_conv1_pad_2) {
            for (int32_t layer3_1_conv1_pad_1 = 0; layer3_1_conv1_pad_1 < 2; ++layer3_1_conv1_pad_1) {
              layer3_1_conv1_LB[((xx_reuse13 + (layer3_1_conv1_pad_1 * 10)) + (layer3_1_conv1_pad_2 * 30))] = layer3_1_conv1_LB[(((xx_reuse13 + (layer3_1_conv1_pad_1 * 10)) + (layer3_1_conv1_pad_2 * 30)) + 10)];
            }
            uint32_t layer3_1_conv1_pad_temp1;
            layer3_1_conv1_pad_temp1 = read_channel_intel(layer3_1_conv1_pad_pipe_89);
            layer3_1_conv1_LB[((xx_reuse13 + (layer3_1_conv1_pad_2 * 30)) + 20)] = layer3_1_conv1_pad_temp1;
          }
          if (2 <= yy_reuse13) {
            for (int32_t layer3_1_conv1_LB_1 = 0; layer3_1_conv1_LB_1 < 3; ++layer3_1_conv1_LB_1) {
              for (int32_t layer3_1_conv1_LB_2 = 0; layer3_1_conv1_LB_2 < 2; ++layer3_1_conv1_LB_2) {
                for (int32_t layer3_1_conv1_LB_0 = 0; layer3_1_conv1_LB_0 < 2; ++layer3_1_conv1_LB_0) {
                  layer3_1_conv1_WB[((layer3_1_conv1_LB_0 + (layer3_1_conv1_LB_1 * 3)) + (layer3_1_conv1_LB_2 * 9))] = layer3_1_conv1_WB[(((layer3_1_conv1_LB_0 + (layer3_1_conv1_LB_1 * 3)) + (layer3_1_conv1_LB_2 * 9)) + 1)];
                }
                layer3_1_conv1_WB[(((layer3_1_conv1_LB_1 + (layer3_1_conv1_LB_2 * 3)) * 3) + 2)] = layer3_1_conv1_LB[((xx_reuse13 + (layer3_1_conv1_LB_1 * 10)) + (layer3_1_conv1_LB_2 * 30))];
              }
            }
            for (int32_t ff15 = 0; ff15 < 64; ++ff15) {
              if (2 <= xx_reuse13) {
                int8_t layer3_1_conv1_sum;
                for (int32_t layer3_1_conv1_rc = 0; layer3_1_conv1_rc < 2; ++layer3_1_conv1_rc) {
                  for (int32_t layer3_1_conv1_ry = 0; layer3_1_conv1_ry < 3; ++layer3_1_conv1_ry) {
                    for (int32_t layer3_1_conv1_rx = 0; layer3_1_conv1_rx < 3; ++layer3_1_conv1_rx) {
                      for (int32_t layer3_1_conv1_rb = 0; layer3_1_conv1_rb < 32; ++layer3_1_conv1_rb) {
                        layer3_1_conv1_sum = ((int8_t)(((int64_t)(((layer3_1_conv1_WB[((layer3_1_conv1_rx + (layer3_1_conv1_ry * 3)) + (layer3_1_conv1_rc * 9))] ^ w_layer3_1_conv1[(((layer3_1_conv1_rx + (layer3_1_conv1_ry * 3)) + (layer3_1_conv1_rc * 9)) + (ff15 * 18))]) & (1L << layer3_1_conv1_rb)) >> layer3_1_conv1_rb)) + ((int64_t)layer3_1_conv1_sum)));
                      }
                    }
                  }
                }
                int8_t layer3_1_conv1_temp;
                layer3_1_conv1_temp = ((int8_t)(576 - ((int32_t)(layer3_1_conv1_sum << 1))));
                write_channel_intel(layer3_1_conv1_pipe_90, layer3_1_conv1_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer3_1_bn1[4096];
        for (int32_t n15 = 0; n15 < 1; ++n15) {
      for (int32_t c15 = 0; c15 < 64; ++c15) {
        for (int32_t h15 = 0; h15 < 8; ++h15) {
          for (int32_t w15 = 0; w15 < 8; ++w15) {
            int8_t layer3_1_conv1_temp1;
            layer3_1_conv1_temp1 = read_channel_intel(layer3_1_conv1_pipe_90);
            int32_t layer3_1_bn1_temp;
            layer3_1_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer3_1_conv1_temp1) * ((int40_t)w_layer3_1_bn1_9[c15]))) + ((int41_t)w_layer3_1_bn1_10[c15])));
            write_channel_intel(layer3_1_bn1_pipe_91, layer3_1_bn1_temp);
          }
        }
      }
    }
    int32_t layer3_1_residual1[4096];
        for (int32_t nn61 = 0; nn61 < 1; ++nn61) {
      for (int32_t cc62 = 0; cc62 < 64; ++cc62) {
        for (int32_t ww60 = 0; ww60 < 8; ++ww60) {
          for (int32_t hh62 = 0; hh62 < 8; ++hh62) {
            int32_t layer3_0_rprelu2_temp2;
            layer3_0_rprelu2_temp2 = read_channel_intel(layer3_0_rprelu2_pipe_133);
            int32_t layer3_1_bn1_temp1;
            layer3_1_bn1_temp1 = read_channel_intel(layer3_1_bn1_pipe_91);
            int32_t layer3_1_residual1_temp;
            layer3_1_residual1_temp = ((int32_t)(((int33_t)layer3_1_bn1_temp1) + ((int33_t)layer3_0_rprelu2_temp2)));
            write_channel_intel(layer3_1_residual1_pipe_92, layer3_1_residual1_temp);
          }
        }
      }
    }
    int32_t layer3_1_rprelu1[4096];
            for (int32_t nn62 = 0; nn62 < 1; ++nn62) {
      for (int32_t cc63 = 0; cc63 < 64; ++cc63) {
        for (int32_t ww61 = 0; ww61 < 8; ++ww61) {
          for (int32_t hh63 = 0; hh63 < 8; ++hh63) {
            int32_t layer3_1_residual1_temp1;
            layer3_1_residual1_temp1 = read_channel_intel(layer3_1_residual1_pipe_92);
            int32_t layer3_1_rprelu1_temp;
            layer3_1_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_1_residual1_temp1) + ((int33_t)w_layer3_1_rprelu1_0[cc63])))) ? (((int64_t)(((int33_t)layer3_1_residual1_temp1) + ((int33_t)w_layer3_1_rprelu1_0[cc63])))) : ((int64_t)(((int64_t)w_layer3_1_rprelu1_2[cc63]) * ((int64_t)(((int33_t)layer3_1_residual1_temp1) + ((int33_t)w_layer3_1_rprelu1_0[cc63]))))))) + ((int64_t)w_layer3_1_rprelu1_1[cc63])));
            write_channel_intel(layer3_1_rprelu1_pipe_134, layer3_1_rprelu1_temp);
            write_channel_intel(layer3_1_rprelu1_pipe_93, layer3_1_rprelu1_temp);
          }
        }
      }
    }
    uint32_t layer3_1_rsign2[128];
        for (int32_t nn63 = 0; nn63 < 1; ++nn63) {
      for (int32_t cc64 = 0; cc64 < 2; ++cc64) {
        for (int32_t hh64 = 0; hh64 < 8; ++hh64) {
          for (int32_t ww62 = 0; ww62 < 8; ++ww62) {
            uint32_t layer3_1_rsign2_pack;
            for (int32_t i16 = 0; i16 < 32; ++i16) {
              int32_t layer3_1_rprelu1_temp1;
              layer3_1_rprelu1_temp1 = read_channel_intel(layer3_1_rprelu1_pipe_93);
              layer3_1_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_1_rprelu1_temp1) + ((int33_t)w_layer3_1_rsign2[((cc64 * 32) + i16)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer3_1_rsign2_temp;
            layer3_1_rsign2_temp = layer3_1_rsign2_pack;
            write_channel_intel(layer3_1_rsign2_pipe_94, layer3_1_rsign2_temp);
          }
        }
      }
    }
    uint32_t layer3_1_conv2_pad[200];
        for (int32_t ii17 = 0; ii17 < 1; ++ii17) {
      for (int32_t cc65 = 0; cc65 < 2; ++cc65) {
        for (int32_t hh65 = 0; hh65 < 10; ++hh65) {
          for (int32_t ww63 = 0; ww63 < 10; ++ww63) {
            bool cond = ((((1 <= ww63) && (ww63 < 9)) && (1 <= hh65)) && (hh65 < 9));
            uint32_t layer3_1_rsign2_temp1 = 0;
            if (cond) {
              layer3_1_rsign2_temp1 = read_channel_intel(layer3_1_rsign2_pipe_94);
            }
            uint32_t layer3_1_conv2_pad_temp;
            layer3_1_conv2_pad_temp = (uint32_t)(cond ? ((uint32_t)layer3_1_rsign2_temp1) : ((uint32_t)0U));
            write_channel_intel(layer3_1_conv2_pad_pipe_95, layer3_1_conv2_pad_temp);
            layer3_1_conv2_pad[(((ww63 + (hh65 * 10)) + (cc65 * 100)) + (ii17 * 200))] = layer3_1_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer3_1_conv2[4096];
    uint32_t layer3_1_conv2_LB[60];
    uint32_t layer3_1_conv2_WB[18];
        for (int32_t nn64 = 0; nn64 < 1; ++nn64) {
      for (int32_t yy_reuse14 = 0; yy_reuse14 < 10; ++yy_reuse14) {
        for (int32_t xx_reuse14 = 0; xx_reuse14 < 10; ++xx_reuse14) {
          for (int32_t layer3_1_conv2_pad_2 = 0; layer3_1_conv2_pad_2 < 2; ++layer3_1_conv2_pad_2) {
            for (int32_t layer3_1_conv2_pad_1 = 0; layer3_1_conv2_pad_1 < 2; ++layer3_1_conv2_pad_1) {
              layer3_1_conv2_LB[((xx_reuse14 + (layer3_1_conv2_pad_1 * 10)) + (layer3_1_conv2_pad_2 * 30))] = layer3_1_conv2_LB[(((xx_reuse14 + (layer3_1_conv2_pad_1 * 10)) + (layer3_1_conv2_pad_2 * 30)) + 10)];
            }
            uint32_t layer3_1_conv2_pad_temp1;
            layer3_1_conv2_pad_temp1 = read_channel_intel(layer3_1_conv2_pad_pipe_95);
            layer3_1_conv2_LB[((xx_reuse14 + (layer3_1_conv2_pad_2 * 30)) + 20)] = layer3_1_conv2_pad_temp1;
          }
          if (2 <= yy_reuse14) {
            for (int32_t layer3_1_conv2_LB_1 = 0; layer3_1_conv2_LB_1 < 3; ++layer3_1_conv2_LB_1) {
              for (int32_t layer3_1_conv2_LB_2 = 0; layer3_1_conv2_LB_2 < 2; ++layer3_1_conv2_LB_2) {
                for (int32_t layer3_1_conv2_LB_0 = 0; layer3_1_conv2_LB_0 < 2; ++layer3_1_conv2_LB_0) {
                  layer3_1_conv2_WB[((layer3_1_conv2_LB_0 + (layer3_1_conv2_LB_1 * 3)) + (layer3_1_conv2_LB_2 * 9))] = layer3_1_conv2_WB[(((layer3_1_conv2_LB_0 + (layer3_1_conv2_LB_1 * 3)) + (layer3_1_conv2_LB_2 * 9)) + 1)];
                }
                layer3_1_conv2_WB[(((layer3_1_conv2_LB_1 + (layer3_1_conv2_LB_2 * 3)) * 3) + 2)] = layer3_1_conv2_LB[((xx_reuse14 + (layer3_1_conv2_LB_1 * 10)) + (layer3_1_conv2_LB_2 * 30))];
              }
            }
            for (int32_t ff16 = 0; ff16 < 64; ++ff16) {
              if (2 <= xx_reuse14) {
                int8_t layer3_1_conv2_sum;
                for (int32_t layer3_1_conv2_rc = 0; layer3_1_conv2_rc < 2; ++layer3_1_conv2_rc) {
                  for (int32_t layer3_1_conv2_ry = 0; layer3_1_conv2_ry < 3; ++layer3_1_conv2_ry) {
                    for (int32_t layer3_1_conv2_rx = 0; layer3_1_conv2_rx < 3; ++layer3_1_conv2_rx) {
                      for (int32_t layer3_1_conv2_rb = 0; layer3_1_conv2_rb < 32; ++layer3_1_conv2_rb) {
                        layer3_1_conv2_sum = ((int8_t)(((int64_t)(((layer3_1_conv2_WB[((layer3_1_conv2_rx + (layer3_1_conv2_ry * 3)) + (layer3_1_conv2_rc * 9))] ^ w_layer3_1_conv2[(((layer3_1_conv2_rx + (layer3_1_conv2_ry * 3)) + (layer3_1_conv2_rc * 9)) + (ff16 * 18))]) & (1L << layer3_1_conv2_rb)) >> layer3_1_conv2_rb)) + ((int64_t)layer3_1_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer3_1_conv2_temp;
                layer3_1_conv2_temp = ((int8_t)(576 - ((int32_t)(layer3_1_conv2_sum << 1))));
                write_channel_intel(layer3_1_conv2_pipe_96, layer3_1_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer3_1_bn2[4096];
        for (int32_t n16 = 0; n16 < 1; ++n16) {
      for (int32_t c16 = 0; c16 < 64; ++c16) {
        for (int32_t h16 = 0; h16 < 8; ++h16) {
          for (int32_t w16 = 0; w16 < 8; ++w16) {
            int8_t layer3_1_conv2_temp1;
            layer3_1_conv2_temp1 = read_channel_intel(layer3_1_conv2_pipe_96);
            int32_t layer3_1_bn2_temp;
            layer3_1_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer3_1_conv2_temp1) * ((int40_t)w_layer3_1_bn2_14[c16]))) + ((int41_t)w_layer3_1_bn2_15[c16])));
            write_channel_intel(layer3_1_bn2_pipe_97, layer3_1_bn2_temp);
          }
        }
      }
    }
    int32_t layer3_1_residual2[4096];
        for (int32_t nn65 = 0; nn65 < 1; ++nn65) {
      for (int32_t cc66 = 0; cc66 < 64; ++cc66) {
        for (int32_t ww64 = 0; ww64 < 8; ++ww64) {
          for (int32_t hh66 = 0; hh66 < 8; ++hh66) {
            int32_t layer3_1_bn2_temp1;
            layer3_1_bn2_temp1 = read_channel_intel(layer3_1_bn2_pipe_97);
            int32_t layer3_1_rprelu1_temp2;
            layer3_1_rprelu1_temp2 = read_channel_intel(layer3_1_rprelu1_pipe_134);
            int32_t layer3_1_residual2_temp;
            layer3_1_residual2_temp = ((int32_t)(((int33_t)layer3_1_bn2_temp1) + ((int33_t)layer3_1_rprelu1_temp2)));
            write_channel_intel(layer3_1_residual2_pipe_98, layer3_1_residual2_temp);
          }
        }
      }
    }
    int32_t layer3_1_rprelu2[4096];
            for (int32_t nn66 = 0; nn66 < 1; ++nn66) {
      for (int32_t cc67 = 0; cc67 < 64; ++cc67) {
        for (int32_t ww65 = 0; ww65 < 8; ++ww65) {
          for (int32_t hh67 = 0; hh67 < 8; ++hh67) {
            int32_t layer3_1_residual2_temp1;
            layer3_1_residual2_temp1 = read_channel_intel(layer3_1_residual2_pipe_98);
            int32_t layer3_1_rprelu2_temp;
            layer3_1_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_1_residual2_temp1) + ((int33_t)w_layer3_1_rprelu2_3[cc67])))) ? (((int64_t)(((int33_t)layer3_1_residual2_temp1) + ((int33_t)w_layer3_1_rprelu2_3[cc67])))) : ((int64_t)(((int64_t)w_layer3_1_rprelu2_5[cc67]) * ((int64_t)(((int33_t)layer3_1_residual2_temp1) + ((int33_t)w_layer3_1_rprelu2_3[cc67]))))))) + ((int64_t)w_layer3_1_rprelu2_4[cc67])));
            write_channel_intel(layer3_1_rprelu2_pipe_135, layer3_1_rprelu2_temp);
            write_channel_intel(layer3_1_rprelu2_pipe_99, layer3_1_rprelu2_temp);
          }
        }
      }
    }
    uint32_t layer3_2_rsign1[128];
        for (int32_t nn67 = 0; nn67 < 1; ++nn67) {
      for (int32_t cc68 = 0; cc68 < 2; ++cc68) {
        for (int32_t hh68 = 0; hh68 < 8; ++hh68) {
          for (int32_t ww66 = 0; ww66 < 8; ++ww66) {
            uint32_t layer3_2_rsign1_pack;
            for (int32_t i17 = 0; i17 < 32; ++i17) {
              int32_t layer3_1_rprelu2_temp1;
              layer3_1_rprelu2_temp1 = read_channel_intel(layer3_1_rprelu2_pipe_99);
              layer3_2_rsign1_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_1_rprelu2_temp1) + ((int33_t)w_layer3_2_rsign1[((cc68 * 32) + i17)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer3_2_rsign1_temp;
            layer3_2_rsign1_temp = layer3_2_rsign1_pack;
            write_channel_intel(layer3_2_rsign1_pipe_100, layer3_2_rsign1_temp);
          }
        }
      }
    }
    uint32_t layer3_2_conv1_pad[200];
        for (int32_t ii18 = 0; ii18 < 1; ++ii18) {
      for (int32_t cc69 = 0; cc69 < 2; ++cc69) {
        for (int32_t hh69 = 0; hh69 < 10; ++hh69) {
          for (int32_t ww67 = 0; ww67 < 10; ++ww67) {
            bool cond = ((((1 <= ww67) && (ww67 < 9)) && (1 <= hh69)) && (hh69 < 9));
            uint32_t layer3_2_rsign1_temp1 = 0;
            if (cond) {
              layer3_2_rsign1_temp1 = read_channel_intel(layer3_2_rsign1_pipe_100);
            }
            uint32_t layer3_2_conv1_pad_temp;
            layer3_2_conv1_pad_temp = (uint32_t)(cond ? ((uint32_t)layer3_2_rsign1_temp1) : ((uint32_t)0U));
            write_channel_intel(layer3_2_conv1_pad_pipe_101, layer3_2_conv1_pad_temp);
            layer3_2_conv1_pad[(((ww67 + (hh69 * 10)) + (cc69 * 100)) + (ii18 * 200))] = layer3_2_conv1_pad_temp;
          }
        }
      }
    }
    int8_t layer3_2_conv1[4096];
    uint32_t layer3_2_conv1_LB[60];
    uint32_t layer3_2_conv1_WB[18];
        for (int32_t nn68 = 0; nn68 < 1; ++nn68) {
      for (int32_t yy_reuse15 = 0; yy_reuse15 < 10; ++yy_reuse15) {
        for (int32_t xx_reuse15 = 0; xx_reuse15 < 10; ++xx_reuse15) {
          for (int32_t layer3_2_conv1_pad_2 = 0; layer3_2_conv1_pad_2 < 2; ++layer3_2_conv1_pad_2) {
            for (int32_t layer3_2_conv1_pad_1 = 0; layer3_2_conv1_pad_1 < 2; ++layer3_2_conv1_pad_1) {
              layer3_2_conv1_LB[((xx_reuse15 + (layer3_2_conv1_pad_1 * 10)) + (layer3_2_conv1_pad_2 * 30))] = layer3_2_conv1_LB[(((xx_reuse15 + (layer3_2_conv1_pad_1 * 10)) + (layer3_2_conv1_pad_2 * 30)) + 10)];
            }
            uint32_t layer3_2_conv1_pad_temp1;
            layer3_2_conv1_pad_temp1 = read_channel_intel(layer3_2_conv1_pad_pipe_101);
            layer3_2_conv1_LB[((xx_reuse15 + (layer3_2_conv1_pad_2 * 30)) + 20)] = layer3_2_conv1_pad_temp1;
          }
          if (2 <= yy_reuse15) {
            for (int32_t layer3_2_conv1_LB_1 = 0; layer3_2_conv1_LB_1 < 3; ++layer3_2_conv1_LB_1) {
              for (int32_t layer3_2_conv1_LB_2 = 0; layer3_2_conv1_LB_2 < 2; ++layer3_2_conv1_LB_2) {
                for (int32_t layer3_2_conv1_LB_0 = 0; layer3_2_conv1_LB_0 < 2; ++layer3_2_conv1_LB_0) {
                  layer3_2_conv1_WB[((layer3_2_conv1_LB_0 + (layer3_2_conv1_LB_1 * 3)) + (layer3_2_conv1_LB_2 * 9))] = layer3_2_conv1_WB[(((layer3_2_conv1_LB_0 + (layer3_2_conv1_LB_1 * 3)) + (layer3_2_conv1_LB_2 * 9)) + 1)];
                }
                layer3_2_conv1_WB[(((layer3_2_conv1_LB_1 + (layer3_2_conv1_LB_2 * 3)) * 3) + 2)] = layer3_2_conv1_LB[((xx_reuse15 + (layer3_2_conv1_LB_1 * 10)) + (layer3_2_conv1_LB_2 * 30))];
              }
            }
            for (int32_t ff17 = 0; ff17 < 64; ++ff17) {
              if (2 <= xx_reuse15) {
                int8_t layer3_2_conv1_sum;
                for (int32_t layer3_2_conv1_rc = 0; layer3_2_conv1_rc < 2; ++layer3_2_conv1_rc) {
                  for (int32_t layer3_2_conv1_ry = 0; layer3_2_conv1_ry < 3; ++layer3_2_conv1_ry) {
                    for (int32_t layer3_2_conv1_rx = 0; layer3_2_conv1_rx < 3; ++layer3_2_conv1_rx) {
                      for (int32_t layer3_2_conv1_rb = 0; layer3_2_conv1_rb < 32; ++layer3_2_conv1_rb) {
                        layer3_2_conv1_sum = ((int8_t)(((int64_t)(((layer3_2_conv1_WB[((layer3_2_conv1_rx + (layer3_2_conv1_ry * 3)) + (layer3_2_conv1_rc * 9))] ^ w_layer3_2_conv1[(((layer3_2_conv1_rx + (layer3_2_conv1_ry * 3)) + (layer3_2_conv1_rc * 9)) + (ff17 * 18))]) & (1L << layer3_2_conv1_rb)) >> layer3_2_conv1_rb)) + ((int64_t)layer3_2_conv1_sum)));
                      }
                    }
                  }
                }
                int8_t layer3_2_conv1_temp;
                layer3_2_conv1_temp = ((int8_t)(576 - ((int32_t)(layer3_2_conv1_sum << 1))));
                write_channel_intel(layer3_2_conv1_pipe_102, layer3_2_conv1_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer3_2_bn1[4096];
        for (int32_t n17 = 0; n17 < 1; ++n17) {
      for (int32_t c17 = 0; c17 < 64; ++c17) {
        for (int32_t h17 = 0; h17 < 8; ++h17) {
          for (int32_t w17 = 0; w17 < 8; ++w17) {
            int8_t layer3_2_conv1_temp1;
            layer3_2_conv1_temp1 = read_channel_intel(layer3_2_conv1_pipe_102);
            int32_t layer3_2_bn1_temp;
            layer3_2_bn1_temp = ((int32_t)(((int41_t)(((int64_t)layer3_2_conv1_temp1) * ((int40_t)w_layer3_2_bn1_9[c17]))) + ((int41_t)w_layer3_2_bn1_10[c17])));
            write_channel_intel(layer3_2_bn1_pipe_103, layer3_2_bn1_temp);
          }
        }
      }
    }
    int32_t layer3_2_residual1[4096];
        for (int32_t nn69 = 0; nn69 < 1; ++nn69) {
      for (int32_t cc70 = 0; cc70 < 64; ++cc70) {
        for (int32_t ww68 = 0; ww68 < 8; ++ww68) {
          for (int32_t hh70 = 0; hh70 < 8; ++hh70) {
            int32_t layer3_2_bn1_temp1;
            layer3_2_bn1_temp1 = read_channel_intel(layer3_2_bn1_pipe_103);
            int32_t layer3_1_rprelu2_temp2;
            layer3_1_rprelu2_temp2 = read_channel_intel(layer3_1_rprelu2_pipe_135);
            int32_t layer3_2_residual1_temp;
            layer3_2_residual1_temp = ((int32_t)(((int33_t)layer3_2_bn1_temp1) + ((int33_t)layer3_1_rprelu2_temp2)));
            write_channel_intel(layer3_2_residual1_pipe_104, layer3_2_residual1_temp);
          }
        }
      }
    }
    int32_t layer3_2_rprelu1[4096];
    for (int32_t nn70 = 0; nn70 < 1; ++nn70) {
      for (int32_t cc71 = 0; cc71 < 64; ++cc71) {
        for (int32_t ww69 = 0; ww69 < 8; ++ww69) {
          for (int32_t hh71 = 0; hh71 < 8; ++hh71) {
            int32_t layer3_2_residual1_temp1;
            layer3_2_residual1_temp1 = read_channel_intel(layer3_2_residual1_pipe_104);
            int32_t layer3_2_rprelu1_temp;
            layer3_2_rprelu1_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_2_residual1_temp1) + ((int33_t)w_layer3_2_rprelu1_0[cc71])))) ? (((int64_t)(((int33_t)layer3_2_residual1_temp1) + ((int33_t)w_layer3_2_rprelu1_0[cc71])))) : ((int64_t)(((int64_t)w_layer3_2_rprelu1_2[cc71]) * ((int64_t)(((int33_t)layer3_2_residual1_temp1) + ((int33_t)w_layer3_2_rprelu1_0[cc71]))))))) + ((int64_t)w_layer3_2_rprelu1_1[cc71])));
            write_channel_intel(layer3_2_rprelu1_pipe_136, layer3_2_rprelu1_temp);
            write_channel_intel(layer3_2_rprelu1_pipe_105, layer3_2_rprelu1_temp);
          }
        }
      }
    }
    uint32_t layer3_2_rsign2[128];
        for (int32_t nn71 = 0; nn71 < 1; ++nn71) {
      for (int32_t cc72 = 0; cc72 < 2; ++cc72) {
        for (int32_t hh72 = 0; hh72 < 8; ++hh72) {
          for (int32_t ww70 = 0; ww70 < 8; ++ww70) {
            uint32_t layer3_2_rsign2_pack;
            for (int32_t i18 = 0; i18 < 32; ++i18) {
              int32_t layer3_2_rprelu1_temp1;
              layer3_2_rprelu1_temp1 = read_channel_intel(layer3_2_rprelu1_pipe_105);
              layer3_2_rsign2_pack |= ((bool)(int32_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_2_rprelu1_temp1) + ((int33_t)w_layer3_2_rsign2[((cc72 * 32) + i18)])))) ? ((int32_t)1) : ((int32_t)0)));
            }
            uint32_t layer3_2_rsign2_temp;
            layer3_2_rsign2_temp = layer3_2_rsign2_pack;
            write_channel_intel(layer3_2_rsign2_pipe_106, layer3_2_rsign2_temp);
          }
        }
      }
    }
    uint32_t layer3_2_conv2_pad[200];
        for (int32_t ii19 = 0; ii19 < 1; ++ii19) {
      for (int32_t cc73 = 0; cc73 < 2; ++cc73) {
        for (int32_t hh73 = 0; hh73 < 10; ++hh73) {
          for (int32_t ww71 = 0; ww71 < 10; ++ww71) {
            bool cond = ((((1 <= ww71) && (ww71 < 9)) && (1 <= hh73)) && (hh73 < 9));
            uint32_t layer3_2_rsign2_temp1 = 0;
            if (cond) {
              layer3_2_rsign2_temp1 = read_channel_intel(layer3_2_rsign2_pipe_106);
            }
            uint32_t layer3_2_conv2_pad_temp;
            layer3_2_conv2_pad_temp = (uint32_t)(cond ? ((uint32_t)layer3_2_rsign2_temp1) : ((uint32_t)0U));
            write_channel_intel(layer3_2_conv2_pad_pipe_107, layer3_2_conv2_pad_temp);
            layer3_2_conv2_pad[(((ww71 + (hh73 * 10)) + (cc73 * 100)) + (ii19 * 200))] = layer3_2_conv2_pad_temp;
          }
        }
      }
    }
    int8_t layer3_2_conv2[4096];
    uint32_t layer3_2_conv2_LB[60];
    uint32_t layer3_2_conv2_WB[18];
        for (int32_t nn72 = 0; nn72 < 1; ++nn72) {
      for (int32_t yy_reuse16 = 0; yy_reuse16 < 10; ++yy_reuse16) {
        for (int32_t xx_reuse16 = 0; xx_reuse16 < 10; ++xx_reuse16) {
          for (int32_t layer3_2_conv2_pad_2 = 0; layer3_2_conv2_pad_2 < 2; ++layer3_2_conv2_pad_2) {
            for (int32_t layer3_2_conv2_pad_1 = 0; layer3_2_conv2_pad_1 < 2; ++layer3_2_conv2_pad_1) {
              layer3_2_conv2_LB[((xx_reuse16 + (layer3_2_conv2_pad_1 * 10)) + (layer3_2_conv2_pad_2 * 30))] = layer3_2_conv2_LB[(((xx_reuse16 + (layer3_2_conv2_pad_1 * 10)) + (layer3_2_conv2_pad_2 * 30)) + 10)];
            }
            uint32_t layer3_2_conv2_pad_temp1;
            layer3_2_conv2_pad_temp1 = read_channel_intel(layer3_2_conv2_pad_pipe_107);
            layer3_2_conv2_LB[((xx_reuse16 + (layer3_2_conv2_pad_2 * 30)) + 20)] = layer3_2_conv2_pad_temp1;
          }
          if (2 <= yy_reuse16) {
            for (int32_t layer3_2_conv2_LB_1 = 0; layer3_2_conv2_LB_1 < 3; ++layer3_2_conv2_LB_1) {
              for (int32_t layer3_2_conv2_LB_2 = 0; layer3_2_conv2_LB_2 < 2; ++layer3_2_conv2_LB_2) {
                for (int32_t layer3_2_conv2_LB_0 = 0; layer3_2_conv2_LB_0 < 2; ++layer3_2_conv2_LB_0) {
                  layer3_2_conv2_WB[((layer3_2_conv2_LB_0 + (layer3_2_conv2_LB_1 * 3)) + (layer3_2_conv2_LB_2 * 9))] = layer3_2_conv2_WB[(((layer3_2_conv2_LB_0 + (layer3_2_conv2_LB_1 * 3)) + (layer3_2_conv2_LB_2 * 9)) + 1)];
                }
                layer3_2_conv2_WB[(((layer3_2_conv2_LB_1 + (layer3_2_conv2_LB_2 * 3)) * 3) + 2)] = layer3_2_conv2_LB[((xx_reuse16 + (layer3_2_conv2_LB_1 * 10)) + (layer3_2_conv2_LB_2 * 30))];
              }
            }
            for (int32_t ff18 = 0; ff18 < 64; ++ff18) {
              if (2 <= xx_reuse16) {
                int8_t layer3_2_conv2_sum;
                for (int32_t layer3_2_conv2_rc = 0; layer3_2_conv2_rc < 2; ++layer3_2_conv2_rc) {
                  for (int32_t layer3_2_conv2_ry = 0; layer3_2_conv2_ry < 3; ++layer3_2_conv2_ry) {
                    for (int32_t layer3_2_conv2_rx = 0; layer3_2_conv2_rx < 3; ++layer3_2_conv2_rx) {
                      for (int32_t layer3_2_conv2_rb = 0; layer3_2_conv2_rb < 32; ++layer3_2_conv2_rb) {
                        layer3_2_conv2_sum = ((int8_t)(((int64_t)(((layer3_2_conv2_WB[((layer3_2_conv2_rx + (layer3_2_conv2_ry * 3)) + (layer3_2_conv2_rc * 9))] ^ w_layer3_2_conv2[(((layer3_2_conv2_rx + (layer3_2_conv2_ry * 3)) + (layer3_2_conv2_rc * 9)) + (ff18 * 18))]) & (1L << layer3_2_conv2_rb)) >> layer3_2_conv2_rb)) + ((int64_t)layer3_2_conv2_sum)));
                      }
                    }
                  }
                }
                int8_t layer3_2_conv2_temp;
                layer3_2_conv2_temp = ((int8_t)(576 - ((int32_t)(layer3_2_conv2_sum << 1))));
                write_channel_intel(layer3_2_conv2_pipe_108, layer3_2_conv2_temp);
              }
            }
          }
        }
      }
    }
    int32_t layer3_2_bn2[4096];
        for (int32_t n18 = 0; n18 < 1; ++n18) {
      for (int32_t c18 = 0; c18 < 64; ++c18) {
        for (int32_t h18 = 0; h18 < 8; ++h18) {
          for (int32_t w18 = 0; w18 < 8; ++w18) {
            int8_t layer3_2_conv2_temp1;
            layer3_2_conv2_temp1 = read_channel_intel(layer3_2_conv2_pipe_108);
            int32_t layer3_2_bn2_temp;
            layer3_2_bn2_temp = ((int32_t)(((int41_t)(((int64_t)layer3_2_conv2_temp1) * ((int40_t)w_layer3_2_bn2_14[c18]))) + ((int41_t)w_layer3_2_bn2_15[c18])));
            write_channel_intel(layer3_2_bn2_pipe_109, layer3_2_bn2_temp);
          }
        }
      }
    }
    int32_t layer3_2_residual2[4096];
        for (int32_t nn73 = 0; nn73 < 1; ++nn73) {
      for (int32_t cc74 = 0; cc74 < 64; ++cc74) {
        for (int32_t ww72 = 0; ww72 < 8; ++ww72) {
          for (int32_t hh74 = 0; hh74 < 8; ++hh74) {
            int32_t layer3_2_bn2_temp1;
            layer3_2_bn2_temp1 = read_channel_intel(layer3_2_bn2_pipe_109);
            int32_t layer3_2_rprelu1_temp2;
            layer3_2_rprelu1_temp2 = read_channel_intel(layer3_2_rprelu1_pipe_136);
            int32_t layer3_2_residual2_temp;
            layer3_2_residual2_temp = ((int32_t)(((int33_t)layer3_2_bn2_temp1) + ((int33_t)layer3_2_rprelu1_temp2)));
            write_channel_intel(layer3_2_residual2_pipe_110, layer3_2_residual2_temp);
          }
        }
      }
    }
    int32_t layer3_2_rprelu2[4096];
        for (int32_t nn74 = 0; nn74 < 1; ++nn74) {
      for (int32_t cc75 = 0; cc75 < 64; ++cc75) {
        for (int32_t ww73 = 0; ww73 < 8; ++ww73) {
          for (int32_t hh75 = 0; hh75 < 8; ++hh75) {
            int32_t layer3_2_residual2_temp1;
            layer3_2_residual2_temp1 = read_channel_intel(layer3_2_residual2_pipe_110);
            int32_t layer3_2_rprelu2_temp;
            layer3_2_rprelu2_temp = ((int32_t)(((int64_t)(int64_t)(((int44_t)0 < ((int44_t)(((int33_t)layer3_2_residual2_temp1) + ((int33_t)w_layer3_2_rprelu2_3[cc75])))) ? (((int64_t)(((int33_t)layer3_2_residual2_temp1) + ((int33_t)w_layer3_2_rprelu2_3[cc75])))) : ((int64_t)(((int64_t)w_layer3_2_rprelu2_5[cc75]) * ((int64_t)(((int33_t)layer3_2_residual2_temp1) + ((int33_t)w_layer3_2_rprelu2_3[cc75]))))))) + ((int64_t)w_layer3_2_rprelu2_4[cc75])));
            write_channel_intel(layer3_2_rprelu2_pipe_111, layer3_2_rprelu2_temp);
          }
        }
      }
    }
    int32_t avgpool_res[64];
    int32_t avgpool_LB[64];
    int32_t avgpool;
        for (int32_t ii20 = 0; ii20 < 1; ++ii20) {
      for (int32_t cc76 = 0; cc76 < 64; ++cc76) {
        for (int32_t hh76 = 0; hh76 < 1; ++hh76) {
          for (int32_t avgpool_LB_i = 0; avgpool_LB_i < 8; ++avgpool_LB_i) {
            for (int32_t avgpool_LB_j = 0; avgpool_LB_j < 8; ++avgpool_LB_j) {
              int32_t layer3_2_rprelu2_temp1;
              layer3_2_rprelu2_temp1 = read_channel_intel(layer3_2_rprelu2_pipe_111);
              avgpool_LB[(avgpool_LB_j + (avgpool_LB_i * 8))] = layer3_2_rprelu2_temp1;
            }
          }
          int32_t avgpool_val;
          for (int32_t avgpool_ry = 0; avgpool_ry < 8; ++avgpool_ry) {
            for (int32_t avgpool_rx = 0; avgpool_rx < 8; ++avgpool_rx) {
              avgpool_val = ((int32_t)(((int33_t)avgpool_val) + ((int33_t)avgpool_LB[(avgpool_rx + (avgpool_ry * 8))])));
            }
          }
          int32_t avgpool_res_temp;
          avgpool_res_temp = ((int32_t)(((int64_t)avgpool_val) / (int64_t)64));
          write_channel_intel(avgpool_res_pipe_112, avgpool_res_temp);
          avgpool_res[((hh76 + cc76) + (ii20 * 64))] = avgpool_res_temp;
        }
      }
    }
    int32_t flatten[64];
        for (int32_t i19 = 0; i19 < 1; ++i19) {
      for (int32_t j = 0; j < 64; ++j) {
        int32_t avgpool_res_temp1;
        avgpool_res_temp1 = read_channel_intel(avgpool_res_pipe_112);
        int32_t flatten_temp;
        flatten_temp = avgpool_res_temp1;
        write_channel_intel(flatten_pipe_113, flatten_temp);
      }
    }
    int32_t fc_matmul[10];
    for (int32_t i20 = 0; i20 < 1; ++i20) {
      for (int32_t j1 = 0; j1 < 10; ++j1) {
        float reducer0;

        if (j1 == 0) { // avoid reading multiple times
          for (int i = 0; i < 64; ++i)
            fc_matmul[i] = read_channel_intel(flatten_pipe_113);
        }
        for (int32_t ra6 = 0; ra6 < 64; ++ra6) {
          int32_t flatten_temp1;
          flatten_temp1 = fc_matmul[ra6];
          reducer0 = (((float)(((int64_t)flatten_temp1) * ((int64_t)w_fc_167[(ra6 + (j1 * 64))]))) + reducer0);
        }
        int32_t fc_matmul_temp;
        fc_matmul_temp = ((int32_t)reducer0 * 1000000000000);
        write_channel_intel(fc_matmul_pipe_114, fc_matmul_temp);
      }
    }
    for (int32_t i21 = 0; i21 < 1; ++i21) {
      for (int32_t j2 = 0; j2 < 10; ++j2) {
        int32_t fc_matmul_temp1;
        fc_matmul_temp1 = read_channel_intel(fc_matmul_pipe_114);
        fc[(j2 + (i21 * 10))] = ((int32_t)(((int33_t)fc_matmul_temp1) + ((int33_t)w_fc_168[j2])));
      }
    }
}
