use std::borrow::Cow;

use bytemuck::Pod;
use bytemuck::Zeroable;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use half::f16;

use super::QuantBufQ8_1;
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_1 {
    pub d: f16, // delta
    pub m: f16, // min
    pub qh: [u8; 4],
    pub qs: [u8; 16], // nibbles / quants
}

impl BlockQ5_1 {
    pub fn dequantize(&self, buf: &mut [f32]) {
        let delta = self.d.to_f32();
        let min_val = self.m.to_f32();
        let qh: u32 = LittleEndian::read_u32(&self.qh);

        for (i, &quantized) in self.qs.iter().enumerate() {
            let xh_0 = (((qh >> i) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (i + 12)) & 0x10) as u8;

            let x0 = (quantized & 0x0F) | xh_0;
            let x1 = (quantized >> 4) | xh_1;

            buf[i] = (x0 as f32) * delta + min_val;
            buf[i + 16] = (x1 as f32) * delta + min_val;
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantBufQ5_1<'a> {
    pub blocks: Cow<'a, [BlockQ5_1]>,
}

impl<'a> QuantBufQ5_1<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ5_1>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ8_0 size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ5_1, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q5_1(data);
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ5_1] {
        &self.blocks
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.blocks)
    }

    pub fn len(&self) -> usize {
        self.blocks.len() * 32
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn dequantize(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert_eq!(start % 32, 0);

        let block_start = start / 32;
        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0.0; 32];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &QuantBufQ8_1, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks[b_offset / 32..(b_offset + len) / 32];

        vec_dot_q5_1_q8_1(abs, bbs)
    }
}

use crate::cpu::buf::buf_q8_1::BlockQ8_1;

pub fn quantize_f32_q5_1(data: &[f32]) -> Vec<BlockQ5_1> {
    let mut bs = Vec::with_capacity(data.len() / 32);
    for chunk in data.chunks(32) {
        // Find the maximum and minimum value in the chunk
        let (min_val, max_val) = chunk
            .iter()
            .fold((f32::MAX, f32::MIN), |(min_val, max_val), &v| {
                (v.min(min_val), v.max(max_val))
            });

        let d = (max_val - min_val) / ((1 << 5) - 1) as f32; // Compute the scaling factor
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qh = [0u8; 4];
        let mut iqh = 0u32;

        let mut qs = [0u8; 16]; // Initialize the quantized values array

        for (i, q) in qs.iter_mut().take(16).enumerate() {
            // Scale the value and convert to u8
            let x0 = (chunk[i] - min_val) * id;
            let x1 = (chunk[i + 16] - min_val) * id;

            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;

            *q = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // get the 5-th bit and store it in qh at the right position
            iqh |= ((xi0 as u32 & 0x10) >> 4) << i;
            iqh |= ((xi1 as u32 & 0x10) >> 4) << (i + 16);
        }
        LittleEndian::write_u32(&mut qh, iqh);
        bs.push(BlockQ5_1 {
            d: f16::from_f32(d),
            m: f16::from_f32(min_val),
            qh,
            qs,
        })
    }
    bs
}

pub fn vec_dot_q5_1_q8_1(abs: &[BlockQ5_1], bbs: &[BlockQ8_1]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        vec_dot_q5_1_q8_1_neon(abs, bbs)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        vec_dot_q5_1_q8_1_avx2(abs, bbs)
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    vec_dot_q5_1_q8_1_fallback(abs, bbs)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn vec_dot_q5_1_q8_1_neon(abs: &[BlockQ5_1], bbs: &[BlockQ8_1]) -> f32 {
    use std::arch::aarch64::*;

    let mut sumf: f32 = 0.0;
    unsafe {
        for i in 0..bbs.len() {
            let qh: u32 = LittleEndian::read_u32(&abs[i].qh);
            let mut sumi = vdupq_n_s32(0);

            for j in (0..16).step_by(8) {
                // Load Q5_1 quantized values
                let q5_1 = vld1_u8(abs[i].qs[j..].as_ptr());
                let q5_1_low = vmovl_u8(q5_1);
                let q5_1_high = vmovl_high_u8(vld1q_u8(abs[i].qs[j..].as_ptr()));

                // Extract high bits from qh
                let qh_low = vdupq_n_u16(((qh >> j) & 0xFF) as u16);
                let qh_high = vdupq_n_u16(((qh >> (j + 8)) & 0xFF) as u16);

                // Combine low and high bits
                let q5_1_low = vorrq_u16(q5_1_low, vshlq_n_u16(qh_low, 4));
                let q5_1_high = vorrq_u16(q5_1_high, vshlq_n_u16(qh_high, 4));

                // Load Q8_1 values
                let q8_1_low = vld1q_s8(bbs[i].qs[j..].as_ptr());
                let q8_1_high = vld1q_s8(bbs[i].qs[j + 16..].as_ptr());

                // Multiply and accumulate
                let prod_low = vmull_s16(
                    vget_low_s16(vreinterpretq_s16_u16(q5_1_low)),
                    vget_low_s16(vreinterpretq_s16_s8(q8_1_low)),
                );
                let prod_high = vmull_s16(
                    vget_low_s16(vreinterpretq_s16_u16(q5_1_high)),
                    vget_low_s16(vreinterpretq_s16_s8(q8_1_high)),
                );

                sumi = vaddq_s32(sumi, vaddq_s32(prod_low, prod_high));
            }

            // Horizontal sum
            let tmp = vpadd_s32(vget_low_s32(sumi), vget_high_s32(sumi));
            let sum = vget_lane_s32(tmp, 0) + vget_lane_s32(tmp, 1);

            sumf += sum as f32 * (abs[i].d * bbs[i].d).to_f32() + (abs[i].m * bbs[i].s).to_f32();
        }
    }
    sumf
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn vec_dot_q5_1_q8_1_avx2(abs: &[BlockQ5_1], bbs: &[BlockQ8_1]) -> f32 {
    use std::arch::x86_64::*;
    let mut sumf: f32 = 0.0;

    unsafe {
        for i in 0..abs.len() {
            let qh: u32 = LittleEndian::read_u32(&abs[i].qh);
            let mut sumi = _mm256_setzero_si256();

            for j in (0..16).step_by(8) {
                // Load Q5_1 quantized values
                let q5_1 = _mm_loadu_si128(abs[i].qs[j..].as_ptr() as *const __m128i);
                let q5_1_low = _mm256_cvtepu8_epi16(q5_1);
                let q5_1_high = _mm256_cvtepu8_epi16(_mm_srli_si128(q5_1, 8));

                // Extract high bits from qh
                let qh_low = _mm256_set1_epi16(((qh >> j) & 0xFF) as i16);
                let qh_high = _mm256_set1_epi16(((qh >> (j + 8)) & 0xFF) as i16);

                // Combine low and high bits
                let q5_1_low = _mm256_or_si256(q5_1_low, _mm256_slli_epi16(qh_low, 4));
                let q5_1_high = _mm256_or_si256(q5_1_high, _mm256_slli_epi16(qh_high, 4));

                // Load Q8_1 quantized values
                let q8_1_low = _mm256_loadu_si256(bbs[i].qs[j..].as_ptr() as *const __m256i);
                let q8_1_high = _mm256_loadu_si256(bbs[i].qs[j + 16..].as_ptr() as *const __m256i);

                // Multiply Q5_1 and Q8_1 values
                let prod_low = _mm256_madd_epi16(q5_1_low, q8_1_low);
                let prod_high = _mm256_madd_epi16(q5_1_high, q8_1_high);

                // Accumulate the results
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(prod_low, prod_high));
            }

            // Horizontal sum of the 256-bit register
            let sumi = _mm256_extract_epi32(sumi, 0)
                + _mm256_extract_epi32(sumi, 1)
                + _mm256_extract_epi32(sumi, 2)
                + _mm256_extract_epi32(sumi, 3)
                + _mm256_extract_epi32(sumi, 4)
                + _mm256_extract_epi32(sumi, 5)
                + _mm256_extract_epi32(sumi, 6)
                + _mm256_extract_epi32(sumi, 7);

            // Add the final result to sumf
            sumf += sumi as f32 * (abs[i].d * bbs[i].d).to_f32() + (abs[i].m * bbs[i].s).to_f32();
        }
    }

    sumf
}

pub fn vec_dot_q5_1_q8_1_fallback(abs: &[BlockQ5_1], bbs: &[BlockQ8_1]) -> f32 {
    let mut sumf = 0f32;

    for i in 0..abs.len() {
        let qh = LittleEndian::read_u32(&abs[i].qh);
        let mut sumi: i32 = 0;
        for j in 0..16 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;

            let x0 = (abs[i].qs[j] as i32 & 0xF) | xh_0 as i32;
            let x1 = (abs[i].qs[j] as i32 >> 4) | xh_1 as i32;

            sumi += (x0 * bbs[i].qs[j] as i32) + (x1 * bbs[i].qs[j + 16] as i32);
        }
        sumf += sumi as f32 * (abs[i].d * bbs[i].d).to_f32() + (abs[i].m * bbs[i].s).to_f32();
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_vector(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|x| (1. * x).round() / 1.)
            .collect::<Vec<_>>()
    }

    #[test]
    fn test_q5_1_block() {
        assert_eq!(
            std::mem::size_of::<BlockQ5_1>(),
            2 * std::mem::size_of::<f16>() + 4 + 16,
            "wrong q5_1 block size/padding"
        );

        let mut buf: [u8; 24] = [0x1; 24];
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

        let blocks = QuantBufQ5_1::from_bytes(&buf[0..24]).blocks;
        assert_eq!(blocks[0].d.to_f32(), 3.0);
        assert_eq!(blocks[0].m.to_f32(), 1.0);
        assert_eq!(blocks[0].qh, [2, 3, 4, 1]);
        assert_eq!(blocks[0].qs, [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1
        ])
    }

    #[test]
    fn test_q5_1_quantize() {
        let data = vec![
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];
        let bs = QuantBufQ5_1::quantize(&data);

        assert_eq!(bs.blocks.len(), 1);
        assert_eq!(bs.blocks[0].d.to_f32(), 0.48388672);
        assert_eq!(bs.blocks[0].m.to_f32(), -8.0);
        assert_eq!(bs.blocks[0].qs, [
            0, 34, 68, 102, 136, 170, 204, 238, 17, 51, 85, 119, 153, 187, 221, 255
        ]);

        let mut dequantize = [0.0f32; 32];
        bs.blocks[0].dequantize(&mut dequantize);
        assert_eq!(round_vector(&dequantize), *data);
    }
}
