use std::borrow::Cow;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct BlockQ6K {
    ql: [u8; 128],    // quants, lower 4 bits
    qh: [u8; 64],     // quants, upper 2 bits
    scales: [i8; 16], // scales, quantized with 8 bits
    d: f16,           // super-block scale
}

impl BlockQ6K {
    pub fn dequantize(&self, buf: &mut [f32]) {
        for n in (0..256).step_by(128) {
            let delta = self.d.to_f32();
            let idx = n / 128;
            let buf = &mut buf[n..];
            let scales = &self.scales[8 * idx..];
            let ql = &self.ql[64 * idx..];
            let qh = &self.qh[32 * idx..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                buf[l] = delta * scales[is] as f32 * q1 as f32;
                buf[l + 32] = delta * scales[is + 2] as f32 * q2 as f32;
                buf[l + 64] = delta * scales[is + 4] as f32 * q3 as f32;
                buf[l + 96] = delta * scales[is + 6] as f32 * q4 as f32;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ6K<'a> {
    pub blocks: Cow<'a, [BlockQ6K]>,
}

impl<'a> QuantBufQ6K<'_> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ6K>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ6_K size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ6K, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }
    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q6_k(data);
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ6K] {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len() * 32
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn dequantize(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert!(start % 32 == 0);

        let block_start = start / 32;
        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0f32; 256];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &QuantBufQ8K, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks[b_offset / 32..(b_offset + len) / 32];

        vec_dot_q6_k_q8_k(abs, bbs)
    }
}

mod impl_fallback {
    use half::f16;

    use super::BlockQ6K;
    use crate::backends::cpu::buf::buf_q8_k::BlockQ8K;
    use crate::backends::cpu::buf::make_qx_quants;
    use crate::backends::cpu::buf::nearest_i32;
    pub fn quantize_f32_q6_k(data: &[f32]) -> Vec<BlockQ6K> {
        let mut bs = Vec::with_capacity(data.len() / 32);

        for chunk in data.chunks(256) {
            let mut l = [0_i8; 256];
            let mut max_scale = 0.0;
            let mut max_abs_scale = 0.0;
            let mut scales = [0f32; 16];
            let mut block_scales = [0_i8; 16];
            let mut ql = [0_u8; 128];
            let mut qh = [0_u8; 64];

            // Find the maximum absolute scale in the chunk
            for (ib, scale_) in scales.iter_mut().enumerate() {
                let scale = make_qx_quants(
                    16,
                    32,
                    &chunk[16 * ib..16 * ib + 16],
                    &mut l[16 * ib..16 * ib + 16],
                    1,
                );
                *scale_ = scale;
                let abs_scale = scale.abs();
                if abs_scale > max_abs_scale {
                    max_abs_scale = abs_scale;
                    max_scale = scale
                }
            }

            let iscale = -128f32 / max_scale;
            let d = 1.0 / iscale;

            // Quantize the chunk
            for (block_scale, scale) in block_scales.iter_mut().zip(scales.iter()) {
                *block_scale = nearest_i32(iscale * scale).min(127) as i8
            }

            for (j, &block_scale) in block_scales.iter().enumerate() {
                let d = d * block_scale as f32;
                if d == 0.0 {
                    continue;
                }
                for ii in 0..16 {
                    let index = 16 * j + ii;
                    let ll = nearest_i32(chunk[index] / d).clamp(-32, 31);
                    l[index] = (ll + 32) as i8;
                }
            }

            for j in (0..256).step_by(128) {
                for l_idx in 0..32 {
                    let q1 = l[j + l_idx] & 0xF;
                    let q2 = l[j + l_idx + 32] & 0xF;
                    let q3 = l[j + l_idx + 64] & 0xF;
                    let q4 = l[j + l_idx + 96] & 0xF;
                    ql[l_idx] = (q1 | (q3 << 4)) as u8;
                    ql[l_idx + 32] = (q2 | (q4 << 4)) as u8;
                    qh[l_idx] = ((l[j + l_idx] >> 4)
                        | ((l[j + l_idx + 32] >> 4) << 2)
                        | ((l[j + l_idx + 64] >> 4) << 4)
                        | ((l[j + l_idx + 96] >> 4) << 6)) as u8;
                }
            }

            // Store the block with the scaling factor, quantized values
            bs.push(BlockQ6K {
                ql,
                qh,
                scales: block_scales,
                d: f16::from_f32(d),
            });
        }

        bs
    }

    pub fn vec_dot_q6_k_q8_k(abs: &[BlockQ6K], bbs: &[BlockQ8K]) -> f32 {
        let mut aux8 = [0i8; 256];
        let mut aux16 = [0i16; 8];
        let mut sums = [0f32; 8];
        let mut aux32 = [0f32; 8];

        for (abs, bbs) in abs.iter().zip(bbs.iter()) {
            let q4 = &abs.ql;
            let qh = &abs.qh;
            let q8 = &bbs.qs;
            aux32.fill(0f32);

            for j in (0..256).step_by(128) {
                let aux8 = &mut aux8[j..];
                let q4 = &q4[j / 2..];
                let qh = &qh[j / 4..];
                for l in 0..32 {
                    aux8[l] = (((q4[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 32] =
                        (((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 64] = (((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 96] =
                        (((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
                }
            }

            for (j, &scale) in abs.scales.iter().enumerate() {
                let scale = scale as f32;
                let q8 = &q8[16 * j..];
                let aux8 = &aux8[16 * j..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
                let q8 = &q8[8..];
                let aux8 = &aux8[8..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
            }

            let d = abs.d.to_f32() * bbs.d;
            for (sum, &a) in sums.iter_mut().zip(aux32.iter()) {
                *sum += a * d;
            }
        }

        let sumf = sums.iter().sum();
        sumf
    }
}
use half::f16;
use impl_fallback::*;

use super::QuantBufQ8K;

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;

    #[test]
    fn test_q6_k_block() {
        assert_eq!(
            std::mem::size_of::<BlockQ6K>(),
            std::mem::size_of::<f16>() + 128 + 64 + 16,
            "wrong q6_k block size/padding"
        );

        let mut buf: [u8; 210] = [0x1; 210];
        let delta = f16::from_f32(3.0).to_bits().to_le_bytes();
        let min_val = f16::from_f32(1.0).to_bits().to_le_bytes();

        buf[0] = delta[0];
        buf[1] = delta[1];
        buf[2] = min_val[0];
        buf[3] = min_val[1];
        buf[4] = 2;
        buf[5] = 3;
        buf[6] = 4;
        buf[4 + 15] = 7;
        buf[208] = 10;

        let blocks = QuantBufQ6K::from_bytes(&buf[0..210]).blocks;
        assert_eq!(blocks[0].d.to_f32(), 1.5854836e-5);
        assert_eq!(blocks[0].ql[0..16], [
            0, 66, 0, 60, 2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]);
        assert_eq!(blocks[0].qh[48..64], [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]);
    }

    #[test]
    fn test_q6_k_quantize() {
        let data = vec![
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];
        let bs = QuantBufQ6K::quantize(&data);

        assert_eq!(bs.blocks.len(), 1);
        assert_eq!(bs.blocks[0].d.to_f32(), -0.001953125);
        let mut dequantize = [0.0f32; 256];

        bs.blocks[0].dequantize(&mut dequantize);
        assert_eq!(dequantize[0..128], data[0..128]);
    }
}
